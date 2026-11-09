#vlm_vision.py

import os
import re
import time
import requests
import numpy as np
from PIL import Image
import torch
import difflib
from transformers import AutoConfig, AutoProcessor

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

    if isinstance(cv_img, Image.Image):
        return cv_img.convert("RGB")

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

def get_model(model_id: str, device: str = "cuda", *, revision=None,
              local_files_only=True, quantization=None):
    """Explicit architecture loaders for the two research baselines; offline by default."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Check nvidia-smi, GPU passthrough, "
                           "CUDA_VISIBLE_DEVICES and the Torch build before loading weights.")
    if quantization not in (None, "nf4") or (quantization and device != "cuda"):
        raise ValueError("NF4 requires CUDA; choose no quantization for CPU")
    if device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA profile requires BF16 support; no automatic dtype fallback")
    options = dict(local_files_only=local_files_only, trust_remote_code=False)
    if revision:
        options["revision"] = revision
    config = AutoConfig.from_pretrained(model_id, **options)
    if config.model_type == "idefics3":
        from transformers import Idefics3ForConditionalGeneration
        loader = Idefics3ForConditionalGeneration
    elif (config.model_type == "llava" and
          getattr(config.vision_config, "model_type", None) == "pixtral"):
        from transformers import LlavaForConditionalGeneration
        loader = LlavaForConditionalGeneration
    else:
        raise ValueError(f"Unsupported architecture {config.model_type!r}; use SmolVLM or Pixtral")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model_options = dict(options, config=config, torch_dtype=dtype, use_safetensors=True)
    if device == "cuda":
        # Logical device 0 is within the mask chosen before process startup.
        model_options["device_map"] = {"": 0}
    if quantization:
        from transformers import BitsAndBytesConfig
        model_options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
    print(f"[*] Loading {model_id} on {device} (offline={local_files_only}, quantization={quantization})")
    processor = AutoProcessor.from_pretrained(model_id, **options)
    model = loader.from_pretrained(model_id, **model_options)
    if device != "cuda":
        model = model.to(device)
    return model.eval(), processor


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
    Calculates legacy SequenceMatcher similarity (not Levenshtein/CER)
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

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=prompt, images=[pil_img], return_tensors="pt").to(model.device)

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

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
    force_cpu: bool = False,
    device=None, revision=None, local_files_only=True, quantization=None
):
    """Orchestrates loading, prompt generation, extraction, and accuracy evaluation."""
    device = device or get_device(force_cpu=force_cpu)
    image_start = time.perf_counter()
    pil_img = get_image(image_source)
    if pil_img is None:
        return None

    image_load_sec = time.perf_counter() - image_start
    load_start = time.perf_counter()
    model, processor = get_model(model_id, device=device, revision=revision,
                                 local_files_only=local_files_only, quantization=quantization)
    load_sec = time.perf_counter() - load_start
    pipeline_start = time.perf_counter()
    metrics = vision_prompt(model, processor, pil_img, query=prompt_query)

    metrics["pipeline_sec"] = time.perf_counter() - pipeline_start + image_load_sec
    metrics["model_load_sec"] = load_sec

    # Format the required output display
    img_name = os.path.basename(image_source) if not image_source.startswith("http") else image_source
    formatted_display = f"bib number in image {img_name} : {metrics['extracted_bib']}"
    metrics["display"] = formatted_display
    metrics["image_source"] = image_source
    metrics["model_id"] = model_id
    metrics["revision"] = revision
    metrics["device"] = device
    metrics["quantization"] = quantization
    metrics["model_architecture"] = type(model).__name__

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
