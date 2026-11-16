"""Synthetic, offline batch/report and resource-policy regressions."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from benchmark import manifest, runner, reporting
from benchmark.scoring import score_response
from tools import pipeline_check as pc


class Batch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # Small distinct PPM images require no image library to create.
        self.records = []
        for i in range(4):
            path = self.root / f'{i}.ppm'
            path.write_bytes(f'P6\n1 1\n255\n'.encode() + bytes([i, 0, 0]))
            self.records.append(dict(id=f'img-{i}', path=path.name, sha256=manifest.sha256_file(path),
                bib='007', split='development' if i == 0 else 'evaluation', review_status='reviewed',
                target_identity='central foreground runner', provenance='synthetic fixture'))
        self.path = self.root / 'batch.json'
        self.path.write_text(json.dumps(dict(records=self.records)))

    def test_non_three_order_splits_and_leading_zeros(self):
        batch, _ = manifest.validated_manifest(self.path, self.root)
        self.assertEqual(batch['records'], self.records)
        self.assertEqual(batch['records'][0]['bib'], '007')
        self.assertEqual([r['split'] for r in batch['records']], ['development'] + ['evaluation'] * 3)

    def test_missing_changed_duplicate_review_and_invalid_records(self):
        for replacement, expected in [
            (dict(self.records[0], path='missing.ppm'), 'not found'),
            (dict(self.records[0], sha256='0'*64), 'sha256 mismatch'),
            (dict(self.records[1], id='other'), 'duplicate'),
            (dict(self.records[0], review_status='ambiguous'), 'review_status'),
            (dict(self.records[0], bib=7), 'non-string'),
            (42, 'object')]:
            records = [replacement] + self.records[1:]
            errors, _ = manifest.validate(dict(records=records), self.root)
            self.assertTrue(any(expected in e for e in errors), errors)
        errors, _ = manifest.validate(dict(records=[]), self.root)
        self.assertTrue(errors)

    def test_duplicate_content_with_same_id_and_alias_path(self):
        copy = self.root / 'copy.ppm'
        copy.write_bytes((self.root / '0.ppm').read_bytes())
        records = [self.records[0], dict(self.records[0], path='copy.ppm')]
        errors, _ = manifest.validate(dict(records=records), self.root)
        self.assertTrue(any('duplicate image' in e for e in errors))
        records[1]['path'] = './0.ppm'
        errors, _ = manifest.validate(dict(records=records), self.root)
        self.assertTrue(any('duplicate path' in e for e in errors))

    def test_rows_scores_failures_and_csv_roundtrip(self):
        report = dict(systems=dict(smolvlm=dict(result=dict(trials=[
            dict(image='0.ppm', status='ok', score=score_response('007', '007'), pipeline_sec=2),
            dict(image='1.ppm', status='failed', error='bad, "input"\nline', pipeline_sec=1)]))))
        config = dict(model_id='test', model_revision='a'*40, device='cpu', quantization=None)
        rows = reporting.detail_rows('run', self.records, 'smolvlm', report, config)
        self.assertEqual(len(rows), 4)
        self.assertEqual([r['status'] for r in rows], ['ok', 'failed', 'unattempted', 'unattempted'])
        summary = reporting.summaries(rows, dict(smolvlm=report))
        self.assertEqual(summary[0]['normalized_exact_accuracy'], 1)
        self.assertEqual(summary[1]['normalized_exact_accuracy'], 0)
        self.assertEqual(summary[1]['attempted_images'], 1)
        path = self.root / 'out.csv'
        reporting.write_csv(path, rows, reporting.FIELDS)
        with path.open(newline='') as f:
            saved = list(csv.DictReader(f))
        self.assertEqual(saved[0]['ground_truth'], '007')
        self.assertEqual(saved[1]['error_message'], 'bad, "input"\nline')
        self.assertEqual(saved[2]['pipeline_sec'], '')

    def test_runner_propagates_frozen_manifest_and_outputs_12_rows(self):
        out = self.root / 'run'
        def fake_run(command, **kwargs):
            if command[0] == 'git':
                return subprocess.CompletedProcess(command, 0, 'a'*40, '')
            self.assertEqual(Path(command[command.index('--manifest')+1]), out / 'manifest.json')
            worker_out = Path(command[command.index('--output')+1])
            worker_out.mkdir()
            worker_out.joinpath('report.json').write_text(json.dumps(dict(status='failed', errors=['mock worker failure'])))
            return subprocess.CompletedProcess(command, 1)
        with patch.object(runner, 'ROOT', self.root), patch.object(runner.subprocess, 'run', side_effect=fake_run):
            result = runner.main(['--manifest', str(self.path), '--device', 'cuda', '--output', str(out)])
        self.assertEqual(result, 1)
        with (out / 'comparison.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(r['status'] == 'unattempted' for r in rows))
        self.assertEqual((out / 'manifest.json').read_bytes(), self.path.read_bytes())
        for name in ('summary.csv', 'REPORT.md', 'metadata.json', 'gpu-preflight.json'):
            self.assertTrue((out / name).exists())

    def test_verifier_custom_manifest_overrides_default_single_image(self):
        out = self.root / 'verify'
        def fake_launch(command, env, log, timeout):
            pc.write_json(out / 'probe.json', dict(device='cpu'))
            return dict(returncode=0)
        with patch.object(sys, 'argv', ['verify', '--manifest', str(self.path), '--systems', 'cascade',
                    '--check-only', '--output', str(out)]), patch.object(pc, 'ROOT', self.root), \
             patch.object(pc, 'cascade_inputs', return_value=[]), \
             patch.object(pc.importlib.metadata, 'version', return_value='mock'), \
             patch.object(pc, 'launch', side_effect=fake_launch):
            with self.assertRaises(SystemExit) as exit:
                pc.main('pc')
        self.assertEqual(exit.exception.code, 0)
        self.assertEqual(json.loads((out/'report.json').read_text())['records'], self.records)

    def test_workers_receive_identical_ordered_records(self):
        out = self.root / 'verify-all'
        requests = []
        def fake_launch(command, env, log, timeout):
            if command[1] == '-c':
                pc.write_json(out / 'probe.json', dict(device='cpu'))
            else:
                request = json.loads(Path(command[-1]).read_text())
                requests.append(request)
                pc.write_json(Path(request['output']) / 'result.json', dict(pipeline_passed=False,
                    trials=[dict(image=self.records[0]['path'], status='failed', error='mock inference failure')]))
            return dict(returncode=0 if command[1] == '-c' else 7)
        with patch.object(sys, 'argv', ['verify', '--manifest', str(self.path), '--systems', 'cascade', 'smolvlm',
                    '--output', str(out)]), patch.object(pc, 'ROOT', self.root), \
             patch.object(pc, 'cascade_inputs', return_value=[]), \
             patch.object(pc.importlib.metadata, 'version', return_value='mock'), \
             patch.object(pc, 'check_model', return_value={}), \
             patch.object(pc, 'launch', side_effect=fake_launch):
            with self.assertRaises(SystemExit) as exit:
                pc.main('pc')
        self.assertEqual(exit.exception.code, 1)
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(r['records'] == self.records for r in requests))
        report = json.loads((out / 'report.json').read_text())
        self.assertTrue(all(not state['passed'] for state in report['systems'].values()))

    def test_default_smoke_snapshot_is_forwarded_explicitly(self):
        out = self.root / 'smoke-run'
        def fake_run(command, **kwargs):
            if command[0] != 'git':
                self.assertIn('--legacy-smoke', command)
                self.assertIn('--manifest', command)
            return subprocess.CompletedProcess(command, 1, '', '')
        with patch.object(runner, 'ROOT', self.root):
            # Validation is imported inside main; patch the owning module.
            with patch.object(manifest, 'validated_manifest', return_value=(dict(records=self.records), [])), \
                 patch.object(runner.subprocess, 'run', side_effect=fake_run):
                source = self.root / 'benchmark/manifests/smoke3.json'
                source.parent.mkdir(parents=True)
                source.write_bytes(self.path.read_bytes())
                result = runner.main(['--systems', 'cascade', '--output', str(out)])
        self.assertEqual(result, 1)



class Retry(unittest.TestCase):
    def setUp(self):
        self.gpu = dict(uuid='GPU-test', index=0, name='L4', total_mib=23034, used_mib=0, utilization_pct=0)
        self.observations = []

    def run_screen(self, measurements, wait=10):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-test'}), \
             patch.object(pc, 'gpu_facts', side_effect=measurements), \
             patch.object(pc, 'compute_processes', return_value=[]), \
             patch.object(pc.time, 'sleep'), patch.object(pc.time, 'monotonic', side_effect=[0, 0, 11, 11]):
            return pc.screen_gpu('GPU-test', True, wait, 5, self.observations)

    def test_busy_then_available(self):
        gpu = self.run_screen([[dict(self.gpu, used_mib=2000)], [self.gpu]])
        self.assertEqual(gpu, self.gpu)
        self.assertEqual([o['passed'] for o in self.observations], [False, True])
        self.assertEqual(self.observations[1]['free_mib'], 23034)

    def test_exhaustion_keeps_measured_values(self):
        with self.assertRaisesRegex(ValueError, '2000'):
            self.run_screen([[dict(self.gpu, used_mib=2000)]]*3)
        self.assertEqual(len(self.observations), 2)
        self.assertTrue(all(not o['passed'] for o in self.observations))

    def test_identity_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'identity unavailable'):
            self.run_screen([[dict(self.gpu, uuid='GPU-other')]])

    def test_mask_change_during_retry(self):
        def change_mask(_):
            os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU-other'
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-test'}), \
             patch.object(pc, 'gpu_facts', return_value=[dict(self.gpu, used_mib=2000)]), \
             patch.object(pc, 'compute_processes', return_value=None), \
             patch.object(pc.time, 'sleep', side_effect=change_mask):
            with self.assertRaisesRegex(ValueError, 'mask changed'):
                pc.screen_gpu('GPU-test', False, 10, 1, self.observations)
        self.assertIsNone(self.observations[0]['compute_processes'])
        self.assertFalse(self.observations[-1]['passed'])

    def test_unknown_measurements_are_not_zero(self):
        with self.assertRaises(ValueError):
            self.run_screen([[dict(self.gpu, used_mib=None)]])
        self.assertFalse(self.observations[0]['passed'])
        self.assertIsNone(self.observations[0]['free_mib'])

    def test_worker_nonzero_is_preserved_and_reaped(self):
        process = MagicMock(pid=123)
        process.wait.return_value = 7
        with tempfile.TemporaryDirectory() as tmp, patch.object(pc.subprocess, 'Popen') as popen, \
             patch.object(pc.os, 'killpg'):
            popen.return_value.__enter__.return_value = process
            state = pc.launch(['worker'], {}, Path(tmp)/'log', 1)
        self.assertEqual(state['returncode'], 7)
        self.assertEqual(process.wait.call_count, 2)
