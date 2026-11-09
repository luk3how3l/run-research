import argparse
import os
import csv
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Single-image offline SmolVLM/Pixtral inference and profiling"
    )

    # CLI arguments with fallbacks to manual defaults
    parser.add_argument(
        "--image", "-i",
        type=str,
        default="media/quinton_bib_image.jpg",
        help="Path or URL to the runner image (default: 'media/quinton_bib_image.jpg')"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="HuggingFaceTB/SmolVLM-500M-Instruct",
        help="Local model directory, or cached SmolVLM/Pixtral checkpoint ID"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="Read the race bib number of the central foreground runner. Return only its digits, or UNKNOWN if unreadable.",
        help="Prompt sent to the VLM"
    )
    parser.add_argument(
        "--ground_truth", "-gt",
        type=str,
        default="",
        help="Known bib digits; legacy similarity is SequenceMatcher, not CER"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force execution on CPU instead of GPU"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="benchmark_results.csv",
        help="Output CSV filename to record benchmark runs"
    )

    parser.add_argument("--device", choices=["cpu", "cuda"], help="Explicit device; CUDA refuses CPU fallback")
    parser.add_argument("--revision", help="Pinned 40-hex revision when using a cached Hub ID")
    parser.add_argument("--quantization", choices=["nf4"], help="CUDA-only 4-bit quantization")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--json", dest="json_path", help="Save raw answer and measurements as JSON")
    args = parser.parse_args()
    if args.cpu and args.device == "cuda":
        parser.error("--cpu conflicts with --device cuda")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.revision:
        import re
        if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
            parser.error("--revision must be a full 40-hex hash")
    if args.quantization and (args.cpu or args.device == "cpu"):
        parser.error("--quantization nf4 requires CUDA")

    # Deferred so import and --help never pull in Torch/Transformers.
    import torch
    torch.set_num_threads(args.threads)
    from vlm_vision import run_vlm_benchmark

    print("==================================================")
    print("           VLM RUNNER BENCHMARK RUNNER            ")
    print("==================================================")
    print(f"Image Source : {args.image}")
    print(f"Model ID     : {args.model}")
    print(f"Ground Truth : {args.ground_truth if args.ground_truth else 'Not provided'}")
    print(f"Device Mode  : {'cpu' if args.cpu else (args.device or 'auto (CUDA/MPS preferred)')}")
    print(f"CSV Export   : {args.csv}")
    print("--------------------------------------------------")

    result = run_vlm_benchmark(
        image_source=args.image,
        model_id=args.model,
        prompt_query=args.prompt,
        ground_truth_bib=args.ground_truth,
        force_cpu=args.cpu, device="cpu" if args.cpu else args.device,
        revision=args.revision, local_files_only=True, quantization=args.quantization
    )

    if result is None:
        print("[!] Execution aborted: Could not process image.")
        raise SystemExit(1)

    if args.json_path:
        import json
        from pathlib import Path
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_path).write_text(json.dumps(result, indent=2) + "\n")

    # User-specified display format
    print(f"\n>>> {result['display']} <<<")
    if result["ground_truth"]:
        print(f"Accuracy Check : {result['char_accuracy']}% Match")
        print(f"Error Category : {result['error_type']}")

    print("\n--- Speed & Resource Profiling ---")
    print(f"Model load     : {result['model_load_sec']:.2f} seconds")
    print(f"Coarse Latency : {result['latency_sec']} seconds ({result['fps']} FPS)")
    print(f"Peak VRAM Usage: {result['peak_vram_mb']} MB")
    print(f"Input Tokens   : {result['input_tokens']} (Text) + {result['visual_tokens']} (Vision Patches)")
    print(f"Output Tokens  : {result['output_tokens']}")
    print(f"Total Tokens   : {result['total_tokens']}")

    # Write or append run data to CSV
    csv_file = args.csv
    file_exists = os.path.isfile(csv_file)
    from pathlib import Path
    Path(csv_file).parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_source", "model_id", "ground_truth", "extracted_bib", "raw_output", "model_load_sec", "pipeline_sec",
        "char_accuracy", "error_type", "latency_sec", "fps",
        "input_tokens", "visual_tokens", "output_tokens", "total_tokens",
        "peak_vram_mb"
    ]

    if file_exists:
        with open(csv_file, newline="", encoding="utf-8") as existing:
            if next(csv.reader(existing), None) != fields:
                raise SystemExit("CSV columns differ from this version; choose a new --csv filename")
    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()

        row_data = {k: result.get(k) for k in fields}
        writer.writerow(row_data)

    print(f"\n[+] Run successfully saved to '{csv_file}'.")


if __name__ == "__main__":
    main()
