"""One-command offline comparison using isolated, preflight-checked workers."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from benchmark.scoring import PROMPT

ROOT = Path(__file__).resolve().parents[1]


def summarize(system, report):
    state = report.get('systems', {}).get(system, {})
    result = state.get('result', {})
    trials = result.get('trials', [])
    planned = len(report.get('records', []))
    correct = result.get('summary', {}).get('normalized_exact', 0)
    times = [t['pipeline_sec'] for t in trials if t.get('status') == 'ok']
    row = dict(system=system, status=report.get('status', 'failed'), planned=planned,
               correct=correct, accuracy=correct / planned if planned else None,
               load_sec=result.get('load_sec'), error='; '.join(report.get('errors', [])) or result.get('error', ''))
    if times:
        row.update(mean_pipeline_sec=statistics.mean(times), median_pipeline_sec=statistics.median(times),
                   images_per_sec=len(times) / sum(times))
    for name in ('input_tokens', 'output_tokens'):
        values = [t['metrics'][name] for t in trials if t.get('status') == 'ok' and name in t.get('metrics', {})]
        row['mean_' + name] = statistics.mean(values) if values else None
    for name in ('process_peak_rss_mib', 'process_cpu_sec'):
        row[name] = result.get(name)
    peaks = [v / 1048576 for t in trials for k, v in t.items() if k.endswith('_peak_allocated_bytes')]
    row['cuda_peak_allocated_mib'] = max(peaks) if peaks else None
    return row


def main(argv=None):
    from benchmark.manifest import validated_manifest, sha256_file
    from benchmark.reporting import detail_rows, summaries, write_csv, FIELDS as DETAIL_FIELDS
    from tools.pipeline_check import REVISIONS, TARGET_POLICY, write_json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--gpu-uuid')
    p.add_argument('--manifest', type=Path)
    p.add_argument('--systems', nargs='+', choices=['cascade', 'smolvlm', 'pixtral'], default=['cascade', 'smolvlm', 'pixtral'])
    p.add_argument('--smolvlm-dir')
    p.add_argument('--pixtral-dir')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--gpu-wait-seconds', type=float, default=0)
    p.add_argument('--gpu-poll-seconds', type=float, default=5)
    p.add_argument('--output', type=Path)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args(argv)
    if args.threads < 1 or args.timeout < 1:
        p.error('threads and timeout must be positive')
    if not all(math.isfinite(x) for x in (args.gpu_wait_seconds, args.gpu_poll_seconds)) or args.gpu_wait_seconds < 0 or args.gpu_poll_seconds <= 0:
        p.error('GPU wait must be finite and nonnegative; poll must be finite and positive')
    source = (args.manifest or ROOT / 'benchmark/manifests/smoke3.json').resolve()
    try:
        manifest, warnings = validated_manifest(source, ROOT, smoke=args.manifest is None)
    except (OSError, ValueError, TypeError) as exc:
        p.error(str(exc))
    output = (args.output or ROOT / 'results' / ('comparison-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Exact byte copy freezes both ordering and original manifest digest.
    frozen_bytes = source.read_bytes()
    if json.loads(frozen_bytes) != manifest:
        p.error('Manifest changed during validation; retry with a stable file')
    (output / 'manifest.json').write_bytes(frozen_bytes)
    records = manifest['records']
    metadata = dict(run_id=output.name, started_utc=datetime.now(timezone.utc).isoformat(),
                    manifest_sha256=sha256_file(output / 'manifest.json'), image_count=len(records),
                    source_manifest=str(source), warnings=warnings, prompt=PROMPT,
                    target_selection_policy=TARGET_POLICY, target_review_required=True,
                    threads=args.threads, warmup=0, repetitions=1, configuration=vars(args).copy(),
                    inherited_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), models={},
                    timing_definitions=dict(pipeline_sec='image decode, preprocessing, inference, postprocessing; CUDA synchronized; excludes load',
                                            generation_sec='synchronized model.generate only, excluding preprocessing and decode',
                                            aggregates='successful executions only; accuracy denominator is all planned images',
                                            p95='nearest rank', cuda_peak='VLM generation or cascade pipeline; includes resident weights'))
    metadata['configuration'] = {k: str(v) if isinstance(v, Path) else v for k, v in metadata['configuration'].items()}
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    metadata['git_commit_sha'] = commit.stdout.strip() if commit.returncode == 0 else None
    reports, rows, observations = {}, [], []
    for system in dict.fromkeys(args.systems):
        model_id, revision = REVISIONS.get(system, ('YOLOv8s + number632resol + Tesseract', None))
        config = dict(model_id=model_id, model_revision=revision, device=args.device,
                      quantization='nf4' if system == 'pixtral' else None)
        metadata['models'][system] = dict(config, revision_status='expected pinned revision; local weights not cryptographically authenticated' if revision else 'local weights hashed in worker result',
            preprocessing='full RGB image; local processor configuration' if system != 'cascade' else 'YOLO defaults; selected person crop; first bib detection; OCR otsu scale=1.5 area=50 psm=11',
            max_new_tokens=32 if system != 'cascade' else None, do_sample=False,
            dtype='bfloat16' if args.device == 'cuda' and system != 'cascade' else 'float32',
            max_image_side=None)
        if system == 'pixtral' and args.device == 'cpu':
            report = dict(status='blocked', errors=['Pixtral NF4 requires CUDA'])
        else:
            command = [sys.executable, '-m', 'tools.verify_pc' if args.device == 'cpu' else 'tools.verify_renn',
                       '--systems', system, '--all-images', '--output', str(output / system),
                       '--threads', str(args.threads), '--timeout', str(args.timeout),
                       '--gpu-wait-seconds', str(args.gpu_wait_seconds), '--gpu-poll-seconds', str(args.gpu_poll_seconds)]
            command += ['--manifest', str(output / 'manifest.json')]
            if args.manifest is None:
                command.append('--legacy-smoke')
            if args.check_only:
                command.append('--check-only')
            if args.device == 'cuda' and args.gpu_uuid:
                command += ['--gpu-uuid', args.gpu_uuid]
            model_dir = getattr(args, system + '_dir', None)
            if model_dir:
                command += ['--' + system + '-dir', model_dir]
            print(f'Comparing {system} ({args.device})', flush=True)
            try:
                state = subprocess.run(command, cwd=ROOT, check=False)
            except OSError as exc:
                state = subprocess.CompletedProcess(command, -1)
                state.launch_error = str(exc)
            report_file = output / system / 'report.json'
            try:
                report = json.loads(report_file.read_text())
            except (OSError, ValueError):
                report = dict(status='failed', errors=['Verifier did not save a valid report'])
            if state.returncode != 0:
                report.setdefault('errors', []).append(getattr(state, 'launch_error', f'Verifier exited {state.returncode}'))
                report['status'] = 'failed'
        reports[system] = report
        rows.extend(detail_rows(output.name, records, system, report, config))
        observations.extend(dict(system=system, **o) for o in report.get('gpu_preflight', []))
        metadata['models'][system]['verification'] = {k: report.get(k) for k in ('models', 'runtime', 'packages', 'selected_gpu')}
        result = report.get('systems', {}).get(system, {}).get('result', {})
        metadata['models'][system]['weights_sha256'] = result.get('weights_sha256')
        metadata.setdefault('software_versions', {})[system] = report.get('packages', {})
        metadata.setdefault('gpu', {})[system] = report.get('selected_gpu')
        metadata.setdefault('cuda_runtime', {})[system] = report.get('runtime', {}).get('cuda_runtime')
        write_csv(output / 'comparison.csv', rows, DETAIL_FIELDS)
        write_csv(output / 'summary.csv', summaries(rows, reports))
        write_json(output / 'gpu-preflight.json', observations)
    metadata['ended_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(output / 'metadata.json', metadata)
    write_json(output / 'comparison.json', reports)
    lines = ['# Three-model comparison', '', f'Run: {output.name}', f'Images: {len(records)}',
             f'Manifest SHA256: {metadata["manifest_sha256"]}', '',
             'Accuracy includes all planned images, including failures and unattempted trials.',
             'Timing summaries include successful executions only. No live-video throughput claim.',
             'Target heuristic: ' + TARGET_POLICY,
             'Human target agreement requires review for every image; target descriptions do not steer the legacy heuristic or fixed VLM prompt.',
             'Cascade raw OCR is preserved separately from its legacy digit-filtered scored prediction in worker results.',
             'Local checkpoint revisions are expected pins, not authenticated by structural checks.', '',
             'See comparison.csv, summary.csv, metadata.json and per-system worker logs.', '']
    lines += [f'{name}: {r.get("status")}; ' + '; '.join(r.get('errors', [])) for name, r in reports.items()]
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(f'Results: {output}', flush=True)
    return 0 if all(r.get('status') in ('passed', 'preflight_passed') for r in reports.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
