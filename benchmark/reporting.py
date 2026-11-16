"""GPU-independent comparison tables from incremental worker reports."""
import csv
import math
import statistics
from benchmark.scoring import summarize

FIELDS = '''run_id image_id image_path image_sha256 split target_identity ground_truth system model_id model_revision device quantization raw_prediction extracted_bib strict_exact normalized_exact character_error_rate status error_message pipeline_sec generation_sec cuda_peak_allocated_mib cuda_peak_reserved_mib target_review_required input_tokens output_tokens process_peak_rss_mib process_cpu_sec scored_prediction image_width image_height'''.split()


def detail_rows(run_id, records, system, report, config):
    state = report.get('systems', {}).get(system, {})
    result = state.get('result', {})
    trials = {t['image']: t for t in result.get('trials', [])}
    rows = []
    for rec in records:
        trial = trials.get(rec['path'], {})
        score = trial.get('score') or {}
        metrics = trial.get('metrics', {})
        row = dict(run_id=run_id, image_id=rec['id'], image_path=rec['path'],
                   image_sha256=rec['sha256'], split=rec['split'], target_identity=rec['target_identity'],
                   ground_truth=rec['bib'], system=system, **config,
                   raw_prediction=trial.get('raw_prediction', score.get('raw')),
                   extracted_bib=score.get('normalized'), strict_exact=score.get('strict_exact'),
                   normalized_exact=score.get('normalized_exact'), character_error_rate=score.get('cer'),
                   status=trial.get('status', 'unattempted'),
                   error_message=trial.get('error') or result.get('error') or state.get('error') or
                   '; '.join(report.get('errors', [])), pipeline_sec=trial.get('pipeline_sec'),
                   generation_sec=metrics.get('latency_sec'),
                   scored_prediction=trial.get('scored_prediction', score.get('raw')),
                   image_width=(trial.get('image_resolution') or [None, None])[0],
                   image_height=(trial.get('image_resolution') or [None, None])[1],
                   target_review_required=system == 'cascade' or rec['target_identity'] != 'central foreground runner')
        for phase in ('allocated', 'reserved'):
            peaks = [v / 1048576 for k, v in trial.items() if k.endswith('_peak_' + phase + '_bytes')]
            row['cuda_peak_' + phase + '_mib'] = max(peaks) if peaks else None
        for key in ('input_tokens', 'output_tokens'):
            row[key] = metrics.get(key)
        for key in ('process_peak_rss_mib', 'process_cpu_sec'):
            row[key] = result.get(key)
        rows.append(row)
    return rows


def summaries(rows, reports):
    output = []
    for system, split in dict.fromkeys((r['system'], r['split']) for r in rows):
        group = [r for r in rows if (r['system'], r['split']) == (system, split)]
        trials = [dict(status=r['status'], score=dict(strict_exact=r['strict_exact'],
                   normalized_exact=r['normalized_exact'], cer=r['character_error_rate'],
                   **{k: False for k in ('digits_only', 'abstained', 'empty', 'no_digits', 'multiple_digit_groups')}))
                  for r in group if r['status'] != 'unattempted']
        stats = summarize(trials, len(group))
        row = dict(system=system, split=split, planned_images=len(group), attempted_images=stats['attempted'],
                   successful_executions=stats['succeeded'], failed_executions=stats['failed'],
                   unattempted_images=stats['unattempted'], exact_correct_predictions=stats['normalized_exact'],
                   normalized_exact_accuracy=stats['normalized_exact'] / len(group),
                   status=(reports.get(system, {}).get('status', 'incomplete')
                           if reports.get(system, {}).get('status') != 'passed'
                           else 'passed' if stats['succeeded'] == len(group) else 'incomplete'),
                   model_load_sec=reports.get(system, {}).get('systems', {}).get(system, {}).get('result', {}).get('load_sec'))
        for metric in ('pipeline_sec', 'generation_sec'):
            values = sorted(r[metric] for r in group if r['status'] == 'ok' and r[metric] is not None)
            row['mean_' + metric] = statistics.mean(values) if values else None
            row['median_' + metric] = statistics.median(values) if values else None
            if metric == 'pipeline_sec':
                row['p95_pipeline_sec'] = values[math.ceil(.95 * len(values)) - 1] if values else None
        peaks = [r['cuda_peak_allocated_mib'] for r in group if r['cuda_peak_allocated_mib'] is not None]
        row['peak_gpu_allocated_mib'] = max(peaks) if peaks else None
        output.append(row)
    return output


def write_csv(path, rows, fields=None):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
