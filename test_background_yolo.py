"""Profile YOLO/OCR and optionally compare a VLM on the same full image.

Run: .venv/bin/python test_background_yolo.py --image media/hannah_bib_test.jpg
Add --vlm-model HuggingFaceTB/SmolVLM-500M-Instruct --ground-truth 1548
for an optional comparison. VLM weights must be cached or downloadable.
"""
import argparse
import importlib
import json
from pathlib import Path
import shutil
import statistics
import time


def timed(function, *args, **kwargs):
    # Synchronize accelerators so measurements include completed inference.
    import torch

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()

    sync()
    start = time.perf_counter()
    result = function(*args, **kwargs)
    sync()
    return result, time.perf_counter() - start


def run_pipeline(yolo, source):
    start = time.perf_counter()
    frame, load_sec = timed(yolo.load_image, source)
    if frame is None:
        raise ValueError(f"Cannot load image: {source}")
    people, person_sec = timed(yolo.detect_person, frame)
    bib_sec = number_sec = 0.0
    numbers = []
    bibs = 0
    for person in people:
        crop, elapsed = timed(yolo.detect_bib, person["frame_crop"])
        bib_sec += elapsed
        if crop is not None:
            bibs += 1
            number, elapsed = timed(
                yolo.extract_bib_number, crop, scale=1.5, area=50, border=0, psm=11
            )
            number_sec += elapsed
            if number:
                numbers.append(number)
    return {
        "load_image_sec": load_sec,
        "detect_person_sec": person_sec,
        "detect_bib_sec": bib_sec,
        "detect_number_sec": number_sec,
        "total_sec": time.perf_counter() - start,
        "people": len(people), "bibs": bibs, "numbers": numbers,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=str(Path(__file__).resolve().parent / "media/hannah_bib_test.jpg"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("background_yolo_profile.json"))
    parser.add_argument("--vlm-model", help="Optional Hugging Face model ID or local model directory")
    parser.add_argument("--ground-truth", help="Expected bib; exact-match checks use every detected YOLO number")
    parser.add_argument("--cpu", action="store_true", help="Force both pipelines to use CPU")
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0:
        parser.error("--runs must be positive and --warmup nonnegative")
    start = time.perf_counter()
    yolo = importlib.import_module("vision.background_yolo")
    load_sec = time.perf_counter() - start
    missing = [path for path in (yolo.PERSON_MODEL_PATH, yolo.BIB_MODEL_PATH) if not Path(path).is_file()]
    if missing:
        parser.exit(1, f"WARNING: Missing model(s): {', '.join(missing)}\n")
    if yolo.GLOBAL_PERSON_MODEL is None or yolo.GLOBAL_BIB_MODEL is None:
        parser.exit(1, "YOLO model loading failed; see errors above.\n")
    if not shutil.which(yolo.pytesseract.pytesseract.tesseract_cmd):
        parser.exit(1, "Tesseract executable is missing. Install Tesseract OCR before running the number stage.\n")
    if args.cpu:
        yolo.GLOBAL_PERSON_MODEL.to("cpu")
        yolo.GLOBAL_BIB_MODEL.to("cpu")
    for _ in range(args.warmup):
        run_pipeline(yolo, args.image)
    runs = [run_pipeline(yolo, args.image) for _ in range(args.runs)]
    report = {
        "image": args.image, "ground_truth": args.ground_truth,
        "person_model": yolo.PERSON_MODEL_PATH, "bib_model": yolo.BIB_MODEL_PATH,
        "import_and_model_load_sec": load_sec, "warmup_runs": args.warmup,
        "yolo_runs": runs,
        "yolo_mean_sec": {key: statistics.mean(run[key] for run in runs) for key in runs[0] if key.endswith("_sec")},
    }
    if args.ground_truth is not None:
        for run in runs:
            run["exact_match"] = args.ground_truth in run["numbers"]
    if args.vlm_model:
        from src import vlm_vision as vlm
        model_and_processor, elapsed = timed(vlm.get_model, args.vlm_model, device=vlm.get_device(args.cpu))
        model, processor = model_and_processor
        report["vlm_model_load_sec"] = elapsed

        def run_vlm():
            image = vlm.get_image(args.image)
            if image is None:
                raise ValueError(f"Cannot load image: {args.image}")
            return vlm.vision_prompt(model, processor, image, query="What is the bib number of the runner? Output only the digits.")

        for _ in range(args.warmup):
            run_vlm()
        report["vlm_runs"] = []
        for _ in range(args.runs):
            metrics, elapsed = timed(run_vlm)
            metrics["total_sec"] = elapsed
            if args.ground_truth is not None:
                metrics["exact_match"] = metrics["extracted_bib"] == args.ground_truth
            report["vlm_runs"].append(metrics)
        report["vlm_mean_total_sec"] = statistics.mean(run["total_sec"] for run in report["vlm_runs"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved profile: {args.output}")


if __name__ == "__main__":
    main()
