import argparse
import os
import csv
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))


def main():
    parser = argparse.ArgumentParser(
        description="VLM Vision Benchmark Runner vs Multi-Stage Cascade Models"
    )

    # CLI arguments with fallbacks to manual defaults
    parser.add_argument(
        "--image", "-i",
        type=str,
        default="test_bib.jpg",
        help="Path or URL to the runner image (default: 'test_bib.jpg')"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="HuggingFaceTB/SmolVLM-500M-Instruct",
        help="HuggingFace checkpoint (e.g., 'HuggingFaceTB/SmolVLM-500M-Instruct', 'Qwen/Qwen2.5-VL-3B-Instruct')"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="What is the bib number worn on the runner's chest? State only the digits.",
        help="Prompt sent to the VLM"
    )
    parser.add_argument(
        "--ground_truth", "-gt",
        type=str,
        default="",
        help="Known bib digits (e.g. '1548') to compute exact character match accuracy"
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

    args = parser.parse_args()

    # Deferred so import and --help never pull in Torch/Transformers.
    from vlm_vision import run_vlm_benchmark

    print("==================================================")
    print("           VLM RUNNER BENCHMARK RUNNER            ")
    print("==================================================")
    print(f"Image Source : {args.image}")
    print(f"Model ID     : {args.model}")
    print(f"Ground Truth : {args.ground_truth if args.ground_truth else 'Not provided'}")
    print(f"Device Mode  : {'CPU' if args.cpu else 'Auto (CUDA/MPS preferred)'}")
    print(f"CSV Export   : {args.csv}")
    print("--------------------------------------------------")

    result = run_vlm_benchmark(
        image_source=args.image,
        model_id=args.model,
        prompt_query=args.prompt,
        ground_truth_bib=args.ground_truth,
        force_cpu=args.cpu
    )

    if result is None:
        print("[!] Execution aborted: Could not process image.")
        return

    # User-specified display format
    print(f"\n>>> {result['display']} <<<")
    if result["ground_truth"]:
        print(f"Accuracy Check : {result['char_accuracy']}% Match")
        print(f"Error Category : {result['error_type']}")

    print("\n--- Speed & Resource Profiling ---")
    print(f"Coarse Latency : {result['latency_sec']} seconds ({result['fps']} FPS)")
    print(f"Peak VRAM Usage: {result['peak_vram_mb']} MB")
    print(f"Input Tokens   : {result['input_tokens']} (Text) + {result['visual_tokens']} (Vision Patches)")
    print(f"Output Tokens  : {result['output_tokens']}")
    print(f"Total Tokens   : {result['total_tokens']}")

    # Write or append run data to CSV
    csv_file = args.csv
    file_exists = os.path.isfile(csv_file)
    fields = [
        "image_source", "model_id", "ground_truth", "extracted_bib", 
        "char_accuracy", "error_type", "latency_sec", "fps", 
        "input_tokens", "visual_tokens", "output_tokens", "total_tokens", 
        "peak_vram_mb"
    ]

    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        
        row_data = {k: result.get(k) for k in fields}
        writer.writerow(row_data)

    print(f"\n[+] Run successfully saved to '{csv_file}'.")


if __name__ == "__main__":
    main()