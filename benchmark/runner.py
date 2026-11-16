"""One-command offline comparison using isolated, preflight-checked workers."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['system', 'status', 'planned', 'correct', 'accuracy', 'mean_pipeline_sec',
          'median_pipeline_sec', 'images_per_sec', 'load_sec', 'mean_input_tokens',
          'mean_output_tokens', 'process_peak_rss_mib', 'process_cpu_sec',
          'cuda_peak_allocated_mib', 'error']


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


def write_outputs(output, rows):
    with (output / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print('\nSystem       Status   Correct   Mean sec   Input tokens   Output tokens   Peak RAM MiB')
    def fmt(value):
        return 'N/A' if value is None else f'{value:.2f}'
    for row in rows:
        print(f"{row['system']:12} {row['status']:8} {row.get('correct', 0)}/{row.get('planned', 0):<7} "
              f"{fmt(row.get('mean_pipeline_sec')):>8} {fmt(row.get('mean_input_tokens')):>14} "
              f"{fmt(row.get('mean_output_tokens')):>15} {fmt(row.get('process_peak_rss_mib')):>14}")
        print(f"  Load sec: {fmt(row.get('load_sec'))}; CPU sec: {fmt(row.get('process_cpu_sec'))}; CUDA peak MiB: {fmt(row.get('cuda_peak_allocated_mib'))}; images/sec: {fmt(row.get('images_per_sec'))}")
        if row.get('error'):
            print('  ' + row['error'])
    print(f'CSV: {output / "comparison.csv"}', flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--gpu-uuid')
    p.add_argument('--systems', nargs='+', choices=['cascade', 'smolvlm', 'pixtral'], default=['cascade', 'smolvlm', 'pixtral'])
    p.add_argument('--smolvlm-dir')
    p.add_argument('--pixtral-dir')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--output', type=Path)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args(argv)
    if args.threads < 1 or args.timeout < 1:
        p.error('threads and timeout must be positive')
    output = (args.output or ROOT / 'results' / ('comparison-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for system in dict.fromkeys(args.systems):
        if system == 'pixtral' and args.device == 'cpu':
            rows.append(dict(system=system, status='blocked', planned=3, correct=None,
                             error='Pixtral NF4 requires CUDA. Run on allocated RENN GPU with --device cuda.'))
        else:
            command = [sys.executable, '-m', 'tools.verify_pc' if args.device == 'cpu' else 'tools.verify_renn',
                       '--systems', system, '--all-images', '--output', str(output / system),
                       '--threads', str(args.threads), '--timeout', str(args.timeout)]
            if args.check_only:
                command.append('--check-only')
            if args.device == 'cuda' and args.gpu_uuid:
                command += ['--gpu-uuid', args.gpu_uuid]
            model_dir = getattr(args, system + '_dir', None)
            if model_dir:
                command += ['--' + system + '-dir', model_dir]
            print(f'Comparing {system} ({args.device})', flush=True)
            subprocess.run(command, cwd=ROOT, check=False)
            report_file = output / system / 'report.json'
            report = json.loads(report_file.read_text()) if report_file.exists() else {'status': 'failed', 'errors': ['Verifier did not save a report']}
            rows.append(summarize(system, report))
        write_outputs(output, rows)
    (output / 'comparison.json').write_text(json.dumps(rows, indent=2) + '\n')
    print('Accuracy uses the same three development photos; timing excludes model loading. '
          'Tokens count processor input IDs and generated IDs, not billed tokens. '
          'Cascade tokens are N/A. RAM is worker lifetime peak RSS; CPU seconds measure process CPU time, '
          'not energy or FLOPs. GPU memory uses the recorded phase peak. No live-video throughput claim.')
    return 0 if all(r['status'] in ('passed', 'preflight_passed') for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
