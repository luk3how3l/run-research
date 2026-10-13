#vlm_vision.py

import os
import re
import time
import requests
import numpy as np
from PIL import Image
import torch
import difflib
from transformers import AutoProcessor, AutoModelForVision2Seq

try:
    import cv2
except ImportError:
    cv2 = None


# ==============================================================================
# 1. HELPER FUNCTIONS
# ==============================================================================

def check_url_or_img(target):
    """
    Determines if input is a remote URL, existing local path, or invalid.
    Returns: 'url', 'path', or 'invalid'
    """
    if isinstance(target, str):
        if target.startswith("http://") or target.startswith("https://"):
            return "url"
        if os.path.exists(target):
            return "path"
    return "invalid"


def load_image(url_or_path: str):
    """Loads an image into a cv2 BGR numpy array from URL or local file path."""
    target_type = check_url_or_img(url_or_path)

    if target_type == "url":
        try:
            response = requests.get(url_or_path, timeout=10)
            response.raise_for_status()
            img_array = np.frombuffer(response.content, np.uint8)
            if cv2 is not None:
                return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            else:
                from io import BytesIO
                return Image.open(BytesIO(response.content)).convert("RGB")
        except requests.exceptions.RequestException as e:
            print(f"[Error] Failed to fetch image URL: {e}")
            return None

    elif target_type == "path":
        if cv2 is not None:
            return cv2.imread(url_or_path)
        else:
            return Image.open(url_or_path).convert("RGB")

    print(f"[Error] Target image path or URL does not exist: {url_or_path}")
    return None


def get_image(url_or_path):
    """
    Loads an image using load_image (OpenCV) and converts it to a PIL Image (RGB)
    suitable for HuggingFace vision processors.
    """
    cv_img = load_image(url_or_path)
    if cv_img is None:
        return None
    
    # Convert OpenCV BGR to RGB, then to PIL Image
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)

def get_device(force_cpu: bool = False) -> str:
    """Auto-detects GPU (CUDA / MPS) or falls back to CPU."""
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_model(model_id: str, device: str = "cuda"):
    """
    Dynamically loads both standard HF Vision2Seq models (SmolVLM, PaliGemma) 
    and Qwen2-VL / Qwen2.5-VL architectures without shape/class mismatches.
    """
    print(f"[*] Initializing model: {model_id} on [{device}]...")
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Qwen-specific loader branch
    if "qwen" in model_id.lower():
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device if device != "cpu" else None
            )
        except ImportError:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device if device != "cpu" else None
            )
    else:
        # Standard Vision2Seq (SmolVLM, Idefics3, PaliGemma)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device if device != "cpu" else None
        )

    if device == "cpu":
        model = model.to("cpu")

    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def extract_bib_digits(raw_text: str) -> str:
    """Extracts only consecutive digits representing the bib number from the response."""
    digits = re.findall(r"\b\d+\b", raw_text)
    if digits:
        return digits[0]
    # Fallback to any grouped digits
    fallback = re.findall(r"\d+", raw_text)
    return fallback[0] if fallback else ""

def compute_character_accuracy(predicted: str, ground_truth: str):
    """
    Calculates character-by-character match accuracy (Levenshtein similarity)
    and pinpoints error types (substitution, deletion, insertion).
    """
    pred_str = str(predicted).strip()
    gt_str = str(ground_truth).strip()

    if not gt_str:
        return {"char_accuracy": 1.0 if not pred_str else 0.0, "match_type": "EXACT" if not pred_str else "INSERTION"}

    matcher = difflib.SequenceMatcher(None, gt_str, pred_str)
    ratio = matcher.ratio()  # 1.0 is 100% match

    # Categorize error types
    diffs = list(matcher.get_opcodes())
    miss_types = set()
    for tag, i1, i2, j1, j2 in diffs:
        if tag == "replace":
            miss_types.add("SUBSTITUTION")
        elif tag == "delete":
            miss_types.add("DELETION/DROPPED_DIGIT")
        elif tag == "insert":
            miss_types.add("INSERTION/EXTRA_DIGIT")

    error_category = "EXACT_MATCH" if ratio == 1.0 else ", ".join(miss_types)

    return {
        "char_accuracy": round(ratio * 100, 2),
        "error_type": error_category
    }

def vision_prompt(model, processor, pil_img: Image.Image, query: str = "What is the bib number of the athlete?"):
    """
    Executes inference and collects latency, token counts, and memory consumption.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query}
            ]
        }
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[pil_img], return_tensors="pt").to(model.device)

    # Input token profiling
    input_ids = inputs.get("input_ids")
    input_tokens_count = input_ids.shape[1] if input_ids is not None else 0
    
    # Estimate visual token grid count if available
    pixel_values = inputs.get("pixel_values")
    visual_tokens_count = pixel_values.shape[1] if (pixel_values is not None and len(pixel_values.shape) > 3) else 0

    # Reset GPU memory stats
    if torch.cuda.is_available() and str(model.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Latency profiling
    start_time = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False
        )
    if torch.cuda.is_available() and str(model.device).startswith("cuda"):
        torch.cuda.synchronize()
    latency_sec = time.perf_counter() - start_time

    # Output token profiling
    output_tokens = generated_ids[0][input_tokens_count:]
    generated_tokens_count = len(output_tokens)
    total_tokens = input_tokens_count + generated_tokens_count

    # Decode text
    raw_output = processor.decode(output_tokens, skip_special_tokens=True).strip()
    bib_number = extract_bib_digits(raw_output)

    # Peak VRAM profiling
    peak_vram_mb = 0.0
    if torch.cuda.is_available() and str(model.device).startswith("cuda"):
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "raw_output": raw_output,
        "extracted_bib": bib_number,
        "latency_sec": round(latency_sec, 4),
        "fps": round(1.0 / latency_sec, 2) if latency_sec > 0 else 0,
        "input_tokens": input_tokens_count,
        "visual_tokens": visual_tokens_count,
        "output_tokens": generated_tokens_count,
        "total_tokens": total_tokens,
        "peak_vram_mb": round(peak_vram_mb, 2)
    }

# ==============================================================================
# 2. MAIN BENCHMARK / INFERENCE WRAPPER
# ==============================================================================

def run_vlm_benchmark(
    image_source: str,
    model_id: str = "HuggingFaceTB/SmolVLM-500M-Instruct",
    prompt_query: str = "What is the bib number of the runner? Output only the digits.",
    ground_truth_bib: str = "",
    force_cpu: bool = False
):
    """Orchestrates loading, prompt generation, extraction, and accuracy evaluation."""
    device = get_device(force_cpu=force_cpu)
    pil_img = get_image(image_source)
    if pil_img is None:
        return None

    model, processor = get_model(model_id, device=device)
    metrics = vision_prompt(model, processor, pil_img, query=prompt_query)

    # Format the required output display
    img_name = os.path.basename(image_source) if not image_source.startswith("http") else image_source
    formatted_display = f"bib number in image {img_name} : {metrics['extracted_bib']}"
    metrics["display"] = formatted_display
    metrics["image_source"] = image_source
    metrics["model_id"] = model_id

    # Evaluate accuracy if ground truth is supplied
    if ground_truth_bib:
        acc_info = compute_character_accuracy(metrics["extracted_bib"], ground_truth_bib)
        metrics.update(acc_info)
        metrics["ground_truth"] = ground_truth_bib
    else:
        metrics["char_accuracy"] = None
        metrics["error_type"] = None
        metrics["ground_truth"] = None

    return metrics

# ==============================================================================
# 3. EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Internal baseline dry-run
    test_image = "test_bib.jpg"
    test_model = "HuggingFaceTB/SmolVLM-500M-Instruct"
    print("\n--- Running Dry-Run Pipeline Check ---")
    if os.path.exists(test_image):
        res = run_vlm_benchmark(test_image, model_id=test_model, ground_truth_bib="1548")
        print(f"\n{res['display']}")
        print(f"Accuracy: {res['char_accuracy']}% ({res['error_type']})")
        print(f"Latency: {res['latency_sec']}s ({res['fps']} FPS) | Tokens: {res['total_tokens']} | Peak VRAM: {res['peak_vram_mb']} MB")
    else:
        print(f"Dry-run test image '{test_image}' not found in current directory. Use start.py with custom arguments.")