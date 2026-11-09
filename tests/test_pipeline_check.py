import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import pipeline_check as pc


class Checkpoints(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'config.json').write_text(json.dumps({'model_type': 'idefics3'}))
        for name in ['tokenizer_config.json', 'tokenizer.json', 'preprocessor_config.json']:
            (self.root / name).write_text('{}')
        header = json.dumps({'x': {'data_offsets': [0, 4], 'dtype': 'F32', 'shape': [1]}}).encode()
        self.payload = struct.pack('<Q', len(header)) + header + b'1234'
        (self.root / 'model.safetensors').write_bytes(self.payload)

    def test_complete_and_truncated(self):
        self.assertEqual(pc.check_model(self.root, 'smolvlm')['shards'], 1)
        (self.root / 'model.safetensors').write_bytes(self.payload[:-1])
        with self.assertRaisesRegex(ValueError, 'payload'):
            pc.check_model(self.root, 'smolvlm')

    def test_index_path_escape(self):
        (self.root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x': '../outside'}}))
        with self.assertRaisesRegex(ValueError, 'escapes'):
            pc.check_model(self.root, 'smolvlm')

    def test_hub_style_blob_symlink(self):
        blob = self.root / 'blobs'
        blob.mkdir()
        (self.root / 'model.safetensors').rename(blob / 'data')
        (self.root / 'model.safetensors').symlink_to(blob / 'data')
        self.assertEqual(pc.check_model(self.root, 'smolvlm')['shards'], 1)

    def test_missing_tokenizer_or_wrong_architecture(self):
        with self.assertRaisesRegex(ValueError, 'architecture'):
            pc.check_model(self.root, 'pixtral')
        (self.root / 'tokenizer.json').unlink()
        with self.assertRaisesRegex(ValueError, 'tokenizer'):
            pc.check_model(self.root, 'smolvlm')


class GPU(unittest.TestCase):
    def setUp(self):
        self.gpus = [{'index': 0, 'uuid': 'GPU-a'}, {'index': 1, 'uuid': 'GPU-b'}]

    def test_mask_numeric_uuid_and_selection(self):
        self.assertEqual(pc.gpu_environment('1', None, self.gpus), 'GPU-b')
        self.assertEqual(pc.gpu_environment('GPU-a', None, self.gpus), 'GPU-a')
        self.assertEqual(pc.gpu_environment('0,1', 'GPU-b', self.gpus), 'GPU-b')
        self.assertEqual(pc.gpu_environment(None, 'GPU-a', self.gpus), 'GPU-a')

    def test_refuse_mask_escape_or_missing_allocation(self):
        for mask, selected in [('0', 'GPU-b'), ('', 'GPU-a'), ('-1', None),
                               (None, None), ('0,1', None), (None, 'GPU-placeholder')]:
            with self.subTest(mask=mask, selected=selected), self.assertRaises(ValueError):
                pc.gpu_environment(mask, selected, self.gpus)


class ProcessAndTarget(unittest.TestCase):
    def test_timeout_is_recorded(self):
        with tempfile.TemporaryDirectory() as d, patch.object(pc.subprocess, 'run',
                side_effect=subprocess.TimeoutExpired('worker', 1)):
            status = pc.launch(['worker'], {}, Path(d) / 'log', 1)
        self.assertIsNone(status['returncode'])
        self.assertIn('Timed out', status['error'])

    def test_target_selection_has_no_label_input(self):
        left = dict(x1=0, y1=0, x2=15, y2=80)
        small = dict(x1=40, y1=0, x2=60, y2=20)
        big = dict(x1=30, y1=0, x2=70, y2=80)
        self.assertIs(pc.select_person([left, small, big], 100), big)
        self.assertIsNone(pc.select_person([], 100))
        self.assertIs(pc.select_person([left], 100), left)

    def test_help_imports_no_ml(self):
        code = "import tools.verify_pc, tools.verify_renn, sys; assert not {'torch', 'transformers', 'ultralytics'} & set(sys.modules)"
        subprocess.run([sys.executable, '-c', code], cwd=pc.ROOT, check=True)
        for tool in ['tools.verify_pc', 'tools.verify_renn']:
            r = subprocess.run([sys.executable, '-m', tool, '--help'], cwd=pc.ROOT,
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)


class Orchestration(unittest.TestCase):
    def test_cpu_check_only_is_not_inference_success(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'run'
            def fake_launch(command, env, log, timeout):
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
                self.assertEqual(env['HF_HUB_OFFLINE'], '1')
                pc.write_json(out / 'probe.json', {'device': 'cpu'})
                return {'returncode': 0}
            with patch.object(sys, 'argv', ['verify', '--systems', 'cascade', '--check-only', '--output', str(out)]), \
                 patch.object(pc.importlib.metadata, 'version', return_value='test'), \
                 patch.object(pc, 'cascade_inputs', return_value=[]), \
                 patch.object(pc, 'launch', side_effect=fake_launch):
                with self.assertRaises(SystemExit) as exit:
                    pc.main('pc')
            self.assertEqual(exit.exception.code, 0)
            report = json.loads((out / 'report.json').read_text())
            self.assertEqual(report['status'], 'preflight_passed')
            self.assertFalse(report['pipeline_verified'])
            self.assertEqual(report['systems'], {})

    def test_gpu_check_only_uses_selected_uuid_and_matches_probe_identity(self):
        for actual, expected_exit in [('GPU-a', 0), ('GPU-wrong', 1)]:
            with self.subTest(actual=actual), tempfile.TemporaryDirectory() as d:
                out = Path(d) / 'run'
                def fake_launch(command, env, log, timeout):
                    self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-a')
                    self.assertIn("'cuda'", command[-1])
                    pc.write_json(out / 'probe.json', {'device': 'cuda', 'gpu_uuid': actual})
                    return {'returncode': 0}
                gpus = [dict(index=0, uuid='GPU-a', name='L4', total_mib=23034,
                             used_mib=0, utilization_pct=0)]
                with patch.object(sys, 'argv', ['verify', '--systems', 'cascade', '--check-only',
                        '--gpu-uuid', 'GPU-a', '--output', str(out)]), \
                     patch.dict(pc.os.environ, {}, clear=True), \
                     patch.object(pc.importlib.metadata, 'version', return_value='test'), \
                     patch.object(pc, 'cascade_inputs', return_value=[]), \
                     patch.object(pc, 'gpu_facts', return_value=gpus), \
                     patch.object(pc, 'launch', side_effect=fake_launch):
                    with self.assertRaises(SystemExit) as exit:
                        pc.main('renn')
                self.assertEqual(exit.exception.code, expected_exit)
                report = json.loads((out / 'report.json').read_text())
                self.assertFalse(report['pipeline_verified'])
                self.assertEqual(report['systems'], {})

    def test_missing_dependencies_exit_nonzero_and_save_error(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'run'
            with patch.object(sys, 'argv', ['verify', '--systems', 'cascade', '--output', str(out)]), \
                 patch.object(pc.importlib.metadata, 'version', side_effect=pc.importlib.metadata.PackageNotFoundError), \
                 patch.object(pc, 'launch') as launch:
                with self.assertRaises(SystemExit) as exit:
                    pc.main('pc')
            self.assertEqual(exit.exception.code, 1)
            self.assertEqual(json.loads((out / 'report.json').read_text())['status'], 'failed')
            launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
