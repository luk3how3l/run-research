"""Bounded development comparison; cached models only, CPU, three known photos."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system', choices=['cascade', 'vlm'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                      CUDA_VISIBLE_DEVICES='', YOLO_CONFIG_DIR=str(args.output.resolve()))
    import torch
    torch.set_num_threads(2)
    from benchmark.manifest import load_manifest, validate, sha256_file
    from benchmark.scoring import PROMPT, score_response
    manifest = load_manifest(ROOT / 'benchmark/manifests/smoke3.json')
    errors, _ = validate(manifest, ROOT, 3, accepted_review=('user_supplied', 'reviewed'))
    if errors:
        raise ValueError(errors)
    report = {'system': args.system, 'device': 'cpu', 'threads': 2, 'warmup': 0,
              'prompt': PROMPT, 'target_policy': 'largest person box crossing horizontal image midpoint; otherwise nearest horizontal box center',
              'target_policy_status': 'development heuristic, requires visual review', 'trials': []}
    import importlib.metadata as md
    report['packages'] = {x: md.version(x) for x in ['torch', 'torchvision', 'transformers', 'ultralytics', 'pillow', 'pytesseract']}
    import socket
    report['host'] = socket.gethostname()
    report['input_manifest_sha256'] = sha256_file(ROOT / 'benchmark/manifests/smoke3.json')
    t = time.perf_counter()
    try:
        if args.system == 'cascade':
            from vision import background_yolo as y
            if y.GLOBAL_PERSON_MODEL is None or y.GLOBAL_BIB_MODEL is None:
                raise RuntimeError('Cascade models not loaded')
            y.GLOBAL_PERSON_MODEL.to('cpu')
            y.GLOBAL_BIB_MODEL.to('cpu')
            report['weights'] = {str(Path(x).relative_to(ROOT)): sha256_file(x) for x in [y.PERSON_MODEL_PATH, y.BIB_MODEL_PATH]}
        else:
            from transformers import Idefics3ForConditionalGeneration, AutoProcessor
            cached = Path.home() / '.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-500M-Instruct/snapshots/a7da5b986cb59b408707209984f360a5f4ad7e47'
            report['model'] = 'HuggingFaceTB/SmolVLM-500M-Instruct'
            report['revision'] = cached.name
            processor = AutoProcessor.from_pretrained(cached, local_files_only=True, trust_remote_code=False)
            model = Idefics3ForConditionalGeneration.from_pretrained(cached, local_files_only=True, trust_remote_code=False, use_safetensors=True, dtype=torch.float32).to('cpu').eval()
        report['load_sec'] = time.perf_counter() - t
        for rec in manifest['records']:
            trial = {'image': rec['path'], 'label': rec['bib'], 'sha256': rec['sha256']}
            start = time.perf_counter()
            try:
                if args.system == 'cascade':
                    import cv2
                    frame = y.load_image(str(ROOT / rec['path']))
                    people = y.detect_person(frame)
                    w = frame.shape[1]
                    central = [b for b in people if b['x1'] <= w/2 <= b['x2']]
                    selected = (max(central, key=lambda b:(b['x2']-b['x1'])*(b['y2']-b['y1'])) if central else min(people, key=lambda b: abs((b['x1']+b['x2'])/2-w/2))) if people else None
                    trial['people_detected'] = len(people)
                    trial['target_bbox'] = [selected[k] for k in ['x1','y1','x2','y2']] if selected else None
                    text = ''
                    if selected:
                        crop = y.detect_bib(selected['frame_crop'])
                        trial['bib_detected'] = crop is not None
                        # Preserve original OCR text while executing legacy preprocessing and extraction.
                        original = y.pytesseract.image_to_string
                        raw_ocr = []
                        def capture(*a, **kw):
                            result = original(*a, **kw)
                            raw_ocr.append(result)
                            return result
                        y.pytesseract.image_to_string = capture
                        try:
                            text = y.extract_bib_number(crop, scale=1.5, area=50, border=0, psm=11) or '' if crop is not None else ''
                        finally:
                            y.pytesseract.image_to_string = original
                        trial['raw_ocr'] = raw_ocr
                    trial['pipeline_sec'] = time.perf_counter() - start
                    trial['score'] = score_response(text, rec['bib'])
                    trial['legacy_joined_digits'] = text
                    if selected:
                        cv2.rectangle(frame, (selected['x1'],selected['y1']), (selected['x2'],selected['y2']), (0,255,0), 3)
                    cv2.imwrite(str(args.output / (rec['id'] + '-target.jpg')), frame)
                    for name in ['ocr_target_bib_crop.jpg','ocr_debug_threshold.jpg']:
                        src = Path(name)
                        if src.exists():
                            shutil.move(str(src), str(args.output / (rec['id'] + '-' + name)))
                else:
                    from PIL import Image
                    with Image.open(ROOT / rec['path']) as im:
                        image = im.convert('RGB')
                    messages = [{'role':'user','content':[{'type':'image'},{'type':'text','text':PROMPT}]}]
                    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = processor(text=prompt, images=[image], return_tensors='pt').to('cpu')
                    trial['preprocess_sec'] = time.perf_counter() - start
                    gen_start = time.perf_counter()
                    with torch.inference_mode():
                        ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
                    trial['generate_sec'] = time.perf_counter() - gen_start
                    text = processor.decode(ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    trial['score'] = score_response(text, rec['bib'])
                    trial['pipeline_sec'] = time.perf_counter() - start
                trial['status'] = 'ok'
            except Exception as exc:
                trial.update(status='failed', error=repr(exc), pipeline_sec=time.perf_counter()-start)
            report['trials'].append(trial)
            (args.output / 'results.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(trial), flush=True)
    except Exception as exc:
        report['load_error'] = repr(exc)
        raise
    finally:
        report['process_high_water_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (args.output / 'results.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
