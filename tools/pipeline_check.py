"""Shared implementation for PC and RENN verification, with isolated model workers."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone

from benchmark.manifest import load_manifest, sha256_file, validate
from benchmark.scoring import PROMPT, score_response, summarize

ROOT = Path(__file__).resolve().parents[1]
REVISIONS = {
    'smolvlm': ('HuggingFaceTB/SmolVLM-500M-Instruct', 'a7da5b986cb59b408707209984f360a5f4ad7e47'),
    'pixtral': ('mistral-experimental/pixtral-12b', 'c2756cbbb9422eba9f6c5c439a214b0392dfc998'),
}
TARGET_POLICY = 'largest person crossing horizontal image midpoint; else nearest horizontal person center'


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def local_model(system, supplied):
    if supplied:
        return Path(supplied).expanduser().resolve()
    conventional = ROOT / 'models' / ('smolvlm-500m' if system == 'smolvlm' else 'pixtral-12b')
    if conventional.is_dir():
        return conventional
    model_id, revision = REVISIONS[system]
    hub = Path(os.environ.get('HF_HUB_CACHE', Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface')) / 'hub'))
    return hub / ('models--' + model_id.replace('/', '--')) / 'snapshots' / revision


def check_model(path, system):
    """Offline structural checks, including indexed shards and safetensors bounds."""
    config = json.loads((path / 'config.json').read_text())
    valid = (config.get('model_type') == 'idefics3' if system == 'smolvlm' else
             config.get('model_type') == 'llava' and config.get('vision_config', {}).get('model_type') == 'pixtral')
    if not valid:
        raise ValueError(f'{system}: unexpected checkpoint architecture in {path}')
    for name in ['tokenizer_config.json', 'tokenizer.json', 'preprocessor_config.json']:
        if not (path / name).is_file():
            raise ValueError(f'Missing processor/tokenizer file: {path / name}')
    index = path / 'model.safetensors.index.json'
    names = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
    if not names:
        raise ValueError('No indexed safetensors shards')
    total = 0
    for name in names:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Weight index escapes model directory')
        # Hub snapshots legitimately symlink shards into the cache blobs directory.
        shard = path / name
        with shard.open('rb') as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError(f'Truncated weight file: {shard}')
            length = struct.unpack('<Q', raw)[0]
            if not 2 <= length <= 100_000_000:
                raise ValueError(f'Invalid safetensors header: {shard}')
            header = json.loads(f.read(length))
        ends = [v['data_offsets'][1] for k, v in header.items() if k != '__metadata__']
        if not ends or shard.stat().st_size != 8 + length + max(ends):
            raise ValueError(f'Truncated/invalid safetensors payload: {shard}')
        total += shard.stat().st_size
    return {'path': str(path), 'shards': len(names), 'weight_file_bytes': total,
            'integrity_check': 'header bounds and file sizes, not cryptographic verification'}


def cascade_inputs():
    errors = []
    for name in ['yolov8s.pt', 'number632resol.pt']:
        if not (ROOT / 'models' / name).is_file():
            errors.append(f'Missing models/{name}; copy local weights from laptop')
    if not shutil.which('tesseract'):
        errors.append('Tesseract executable missing: install native tesseract-ocr')
    return errors


def gpu_environment(inherited, selected, gpus):
    """Resolve a single allocated GPU without escaping an inherited mask."""
    by_uuid = {g['uuid']: g for g in gpus}
    by_index = {str(g['index']): g['uuid'] for g in gpus}
    if inherited is not None:
        tokens = [x.strip() for x in inherited.split(',') if x.strip()]
        allowed = [x if x in by_uuid else by_index.get(x) for x in tokens]
        if not allowed or any(x is None for x in allowed):
            raise ValueError('Inherited CUDA_VISIBLE_DEVICES is empty/invalid or uses an unsupported mask')
        if selected and selected not in allowed:
            raise ValueError('Requested GPU is outside inherited CUDA_VISIBLE_DEVICES')
        if not selected and len(allowed) != 1:
            raise ValueError('Multiple allocated GPUs: pass --gpu-uuid to select one within the mask')
        selected = selected or allowed[0]
    elif not selected:
        raise ValueError('No allocated GPU selected: supply --gpu-uuid or inherit a single-GPU allocation mask')
    if selected not in by_uuid:
        raise ValueError('GPU UUID not found; use the real UUID from nvidia-smi')
    return selected


def gpu_facts():
    r = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=20)
    rows = []
    for line in r.stdout.splitlines():
        index, uuid, name, total, used, utilization = [x.strip() for x in line.split(',')]
        rows.append(dict(index=int(index), uuid=uuid, name=name, total_mib=int(total),
                         used_mib=int(used), utilization_pct=int(utilization)))
    return rows


def runtime_probe(device, nf4):
    import torch
    import torchvision
    # Exercise native torchvision ops as well as importing package metadata.
    boxes = torch.tensor([[0., 0., 2., 2.]])
    torchvision.ops.nms(boxes, torch.tensor([1.]), 0.5)
    result = {'torch': torch.__version__, 'torchvision': torchvision.__version__, 'device': device}
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: check nvidia-smi, driver/passthrough, mask and Torch build')
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Expected exactly one allocated visible GPU')
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 unavailable; the configured CUDA model profile requires it')
        (torch.ones(2, device='cuda') + 1).cpu()
        result.update(gpu=torch.cuda.get_device_name(0), gpu_uuid=str(getattr(torch.cuda.get_device_properties(0), 'uuid', 'unknown')), cuda_runtime=torch.version.cuda,
                      bf16=True, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        if nf4:
            import bitsandbytes as bnb
            result['bitsandbytes'] = bnb.__version__
    else:
        torch.ones(2) + 1
    return result


def select_person(people, width):
    central = [p for p in people if p['x1'] <= width / 2 <= p['x2']]
    if central:
        return max(central, key=lambda p: (p['x2'] - p['x1']) * (p['y2'] - p['y1']))
    return min(people, key=lambda p: abs((p['x1'] + p['x2']) / 2 - width / 2)) if people else None


def run_worker(request):
    """A process per system: no overlapping model memory or benchmark execution."""
    import torch
    torch.set_num_threads(request['threads'])
    device = request['device']
    system = request['system']
    output = Path(request['output'])
    os.chdir(output)  # legacy OCR debug files stay inside this run
    report = {'system': system, 'device': device, 'trials': [], 'planned': len(request['records']),
              'status': 'running', 'prompt': PROMPT, 'target_policy': TARGET_POLICY}

    def sync():
        if device == 'cuda':
            torch.cuda.synchronize()

    try:
        start = time.perf_counter()
        if system == 'cascade':
            from vision import background_yolo as y
            if y.GLOBAL_PERSON_MODEL is None or y.GLOBAL_BIB_MODEL is None:
                raise RuntimeError('Cascade model loading failed')
            y.GLOBAL_PERSON_MODEL.to(device)
            y.GLOBAL_BIB_MODEL.to(device)
            report['weights_sha256'] = {Path(p).name: sha256_file(p) for p in [y.PERSON_MODEL_PATH, y.BIB_MODEL_PATH]}
        else:
            from src.vlm_vision import get_model, vision_prompt
            from PIL import Image
            model, processor = get_model(request['model_dir'], device=device,
                                         local_files_only=True, quantization='nf4' if system == 'pixtral' else None)
            report['model_dir'] = request['model_dir']
            report['quantization'] = 'nf4' if system == 'pixtral' else None
        sync()
        report['load_sec'] = time.perf_counter() - start
        write_json(output / 'result.json', report)
        for rec in request['records']:
            trial = {'image': rec['path'], 'label': rec['bib'], 'sha256': rec['sha256']}
            print(f"{system}: {rec['path']}", flush=True)
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            sync()
            start = time.perf_counter()
            try:
                if system == 'cascade':
                    frame = y.load_image(str(ROOT / rec['path']))
                    if frame is None:
                        raise ValueError('Image decode failed')
                    people = y.detect_person(frame)
                    target = select_person(people, frame.shape[1])
                    trial['target_bbox'] = [target[k] for k in ['x1', 'y1', 'x2', 'y2']] if target else None
                    crop = y.detect_bib(target['frame_crop']) if target else None
                    raw = []
                    ocr = y.pytesseract.image_to_string

                    def capture(*a, **kw):
                        text = ocr(*a, **kw)
                        raw.append(text)
                        return text

                    y.pytesseract.image_to_string = capture
                    try:
                        prediction = (y.extract_bib_number(crop, scale=1.5, area=50, border=0, psm=11)
                                      if crop is not None else '') or ''
                    finally:
                        y.pytesseract.image_to_string = ocr
                    # The legacy function swallows OCR exceptions; no captured output means execution failed.
                    if crop is not None and not raw:
                        raise RuntimeError('Legacy OCR failed before returning raw text; inspect worker.log')
                    trial.update(raw_ocr=raw, outcome='no_person' if target is None else
                                 'no_bib' if crop is None else 'ocr_empty' if not prediction else 'prediction')
                    trial['score'] = score_response(prediction, rec['bib'])
                    for name in ['ocr_target_bib_crop.jpg', 'ocr_debug_threshold.jpg']:
                        if Path(name).exists():
                            Path(name).replace(output / (rec['id'] + '-' + name))
                else:
                    with Image.open(ROOT / rec['path']) as im:
                        image = im.convert('RGB')
                    metrics = vision_prompt(model, processor, image, query=PROMPT)
                    trial.update(metrics=metrics, score=score_response(metrics['raw_output'], rec['bib']))
                sync()
                trial.update(status='ok', pipeline_sec=time.perf_counter() - start)
                if device == 'cuda':
                    phase = 'cuda_pipeline' if system == 'cascade' else 'cuda_generation'
                    trial[phase + '_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                    trial[phase + '_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            except Exception as exc:
                trial.update(status='failed', error=repr(exc), pipeline_sec=time.perf_counter() - start)
            report['trials'].append(trial)
            write_json(output / 'result.json', report)
        report['summary'] = summarize(report['trials'], report['planned'])
        report['pipeline_passed'] = all(t['status'] == 'ok' for t in report['trials'])
        report['status'] = 'passed' if report['pipeline_passed'] else 'failed'
        return 0 if report['pipeline_passed'] else 1
    except Exception as exc:
        report.update(status='failed', pipeline_passed=False, error=repr(exc))
        print(repr(exc), flush=True)
        return 1
    finally:
        report['summary'] = summarize(report['trials'], report['planned'])
        write_json(output / 'result.json', report)


def launch(command, env, log, timeout):
    with log.open('w', encoding='utf-8') as f:
        try:
            result = subprocess.run(command, env=env, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
            return {'returncode': result.returncode}
        except subprocess.TimeoutExpired:
            return {'returncode': None, 'error': f'Timed out after {timeout} seconds'}


def main(profile):
    parser = argparse.ArgumentParser(description=f'{profile.upper()} offline pipeline verification: execution success is separate from accuracy.')
    defaults = ['cascade', 'smolvlm'] if profile == 'pc' else ['cascade', 'smolvlm', 'pixtral']
    parser.add_argument('--systems', nargs='+', choices=defaults, default=defaults)
    parser.add_argument('--smolvlm-dir', help='Local checkpoint; default: models/smolvlm-500m or pinned Hub cache')
    if profile == 'renn':
        parser.add_argument('--pixtral-dir', help='Local checkpoint; default: models/pixtral-12b or pinned Hub cache')
    parser.add_argument('--all-images', action='store_true', help='Use all three known photos; default: Quinton only')
    parser.add_argument('--check-only', action='store_true', help='Files/dependencies/tiny runtime probe only; no model loading')
    parser.add_argument('--output', type=Path, help='New directory; default: timestamped results/verify-*')
    parser.add_argument('--timeout', type=int, default=900, help='Seconds per isolated worker, default 900')
    parser.add_argument('--threads', type=int, default=2)
    if profile == 'renn':
        parser.add_argument('--gpu-uuid', help='Allocated GPU UUID; must respect inherited CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    if args.timeout < 1 or args.threads < 1:
        parser.error('timeout and threads must be positive')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = (args.output or ROOT / 'results' / f'verify-{profile}-{stamp}').resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error('--output already exists; choose a new directory')
    systems = list(dict.fromkeys(args.systems))
    report = {'profile': profile, 'host': socket.gethostname(), 'python': sys.executable,
              'mode': 'check-only' if args.check_only else 'inference', 'systems': {},
              'errors': [], 'pipeline_verified': False, 'status': 'running',
              'started_utc': stamp, 'threads': args.threads, 'warmup': 0,
              'target_policy': TARGET_POLICY, 'target_policy_status': 'development heuristic; review targets',
              'note': 'A successful smoke run verifies this setup and sample only, not final accuracy or live throughput.'}
    device = 'cpu' if profile == 'pc' else 'cuda'
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
               PYTHONPATH=str(ROOT), YOLO_OFFLINE='true')
    try:
        manifest = load_manifest(ROOT / 'benchmark/manifests/smoke3.json')
        errors, _ = validate(manifest, ROOT, 3, accepted_review=('reviewed', 'user_supplied'))
        if errors:
            raise ValueError('; '.join(errors))
        records = manifest['records'] if args.all_images else [manifest['records'][2]]
        report['records'] = records
        needed = ['torch', 'torchvision', 'Pillow', 'numpy', 'requests']
        if any(s != 'cascade' for s in systems):
            needed += ['transformers', 'accelerate', 'safetensors']
        if 'cascade' in systems:
            needed += ['ultralytics', 'opencv-python', 'pytesseract', 'matplotlib']
            report['errors'].extend(cascade_inputs())
        if 'pixtral' in systems:
            needed += ['bitsandbytes']
        report['packages'] = {}
        for name in needed:
            try:
                report['packages'][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                report['errors'].append(f'Missing Python package: {name}; see requirements/ and SETUP_RENN.md')
        model_dirs = {}
        for name in systems:
            if name != 'cascade':
                path = local_model(name, getattr(args, name + '_dir'))
                model_dirs[name] = str(path)
                try:
                    report.setdefault('models', {})[name] = check_model(path, name)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    report['errors'].append(f'{name} local checkpoint invalid: {exc}; download manually using setup guide')
        if profile == 'pc':
            env['CUDA_VISIBLE_DEVICES'] = ''
        else:
            gpus = gpu_facts()
            selected = gpu_environment(os.environ.get('CUDA_VISIBLE_DEVICES'), args.gpu_uuid, gpus)
            gpu = next(g for g in gpus if g['uuid'] == selected)
            report['selected_gpu'] = gpu
            inherited_mask = os.environ.get('CUDA_VISIBLE_DEVICES')
            numeric_mask = inherited_mask is not None and any(t.strip().isdigit() for t in inherited_mask.split(','))
            if numeric_mask and len(inherited_mask.split(',')) != 1:
                raise ValueError('Multiple numeric allocation indices cannot safely be remapped to UUIDs; request a single-GPU allocation or UUID-based mask')
            # Never reinterpret a scheduler numeric index through a different CUDA ordering.
            env['CUDA_VISIBLE_DEVICES'] = inherited_mask if numeric_mask else selected
            # Read-only screening: never launch models onto an already busy device.
            if gpu['utilization_pct'] > 10 or gpu['used_mib'] > 1024:
                report['errors'].append('Selected GPU is busy (>10% utilization or >1 GiB used); wait for allocation')
            if 'pixtral' in systems and gpu['total_mib'] - gpu['used_mib'] < 18 * 1024:
                report['errors'].append('Pixtral policy requires at least 18 GiB free VRAM; not a fit guarantee')
        if report['errors']:
            raise ValueError('Preflight failed; see errors below')
        probe_path = output / 'probe.json'
        code = ('from tools.pipeline_check import runtime_probe, write_json; from pathlib import Path; '
                f'write_json(Path({str(probe_path)!r}), runtime_probe({device!r}, {"pixtral" in systems!r}))')
        probe_status = launch([sys.executable, '-c', code], env, output / 'probe.log', min(args.timeout, 120))
        if probe_status['returncode'] != 0:
            raise RuntimeError(f'Runtime probe failed: {probe_status}; read {output / "probe.log"}')
        report['runtime'] = json.loads(probe_path.read_text())
        if profile == 'renn':
            actual = report['runtime'].get('gpu_uuid', 'unknown')
            if actual == 'unknown' or actual.lower() != report['selected_gpu']['uuid'].lower():
                raise RuntimeError('Torch GPU identity could not be matched to the selected nvidia-smi UUID; use an administrator-provided UUID allocation mask')
        if args.check_only:
            report['status'] = 'preflight_passed'
        else:
            for system in systems:
                work = output / system
                work.mkdir()
                request = dict(system=system, device=device, records=records, output=str(work),
                               model_dir=model_dirs.get(system), threads=args.threads)
                write_json(work / 'request.json', request)
                worker_env = dict(env, YOLO_CONFIG_DIR=str(work), MPLCONFIGDIR=str(work / 'matplotlib'))
                command = [sys.executable, '-m', 'tools.pipeline_check', str(work / 'request.json')]
                print(f'Running {system} on {device}; log: {work / "worker.log"}', flush=True)
                state = launch(command, worker_env, work / 'worker.log', args.timeout)
                if (work / 'result.json').exists():
                    state['result'] = json.loads((work / 'result.json').read_text())
                state['passed'] = state['returncode'] == 0 and state.get('result', {}).get('pipeline_passed', False)
                report['systems'][system] = state
                print(f"{system}: {'PASS' if state['passed'] else 'FAIL'}", flush=True)
                write_json(output / 'report.json', report)
            report['pipeline_verified'] = all(s['passed'] for s in report['systems'].values())
            report['status'] = 'passed' if report['pipeline_verified'] else 'failed'
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        report['status'] = 'failed'
        report['errors'].append(str(exc))
    write_json(output / 'report.json', report)
    lines = [f"# {profile.upper()} pipeline verification", '', f"Status: {report['status']}",
             f"Mode: {report['mode']}", f"Host: {report['host']}", '',
             'PASS means execution succeeded, even if bib accuracy is poor. Check-only never verifies inference.', '']
    for system, state in report['systems'].items():
        result = state.get('result', {})
        summary = result.get('summary', {})
        lines.append(f"- {system}: {'PASS' if state['passed'] else 'FAIL'}; normalized correct "
                     f"{summary.get('normalized_exact', 0)}/{len(report.get('records', []))}; logs: {system}/worker.log")
    lines += ['', *report['errors']]
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(f"Status: {report['status']}\nReport: {output / 'REPORT.md'}")
    for error in report['errors']:
        print(error)
    raise SystemExit(0 if report['status'] in ('passed', 'preflight_passed') else 1)


if __name__ == '__main__':
    try:
        request = json.loads(Path(sys.argv[1]).read_text())
        raise SystemExit(run_worker(request))
    except Exception as exc:
        print(repr(exc), file=sys.stderr)
        raise SystemExit(1)
