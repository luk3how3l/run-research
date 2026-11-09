# Race-bib recognition research

Compare one runner's bib number using **SmolVLM**, **Pixtral**, and the original **YOLO person → YOLO bib → Tesseract OCR** cascade. Phase 2 adds detector → VLM hybrids. Primary outcome: exact bib accuracy; also measure speed and memory for eventual live background processing. Target deliverables are a paper and demonstration by the end of November 2026.

**Setting up RENN? Start with [SETUP_RENN.md](SETUP_RENN.md).** It covers SSH, system dependencies, separate pip requirement lists, your CUDA error, manual model downloads, and GPU execution. [Laptop setup and annotation](SETUP_LAPTOP_AND_RENN.md) is separate.

## Current state

- Annotation UI, CSV save/resume/edit, and reviewed-manifest export work.
- A bounded CPU comparison ran on the three known development photos on 2 November. Both systems read 3/3 bibs correctly after digit extraction. Cascade averaged 4.66 s/image; SmolVLM averaged 104.68 s/image. No downloading occurred during inference. [Raw artifacts and protocol](results/smoke-2026-11-02/REPORT.md).
- The earlier 14 October SmolVLM run scored 1/3 normalized matches, using different software. Preserve it as a separate historical result; see [the original handoff](coding_agent_10_19.md).
- `start.py` now uses explicit SmolVLM/Pixtral loaders and supports offline local directories, explicit CUDA, and optional NF4. Updated CUDA/NF4 execution still needs a real RENN smoke test.
- The shared configurable 100-image runner, final comparison reporter, and hybrids are pending. Existing configuration files are groundwork, not a runnable shared evaluator.
- The fixed 100-photo dataset is still being collected and labeled. The three original photos are development data, not an untouched test set. Additional local photos must be labeled before evaluation.

Three examples do not establish general accuracy or a winning architecture. See [goals and sprints](PROJECT_GOALS_AND_SPRINTS.md) for delivery work.

## Programs and modules

Run commands from the repository root with your environment activated.

| Program/module | Purpose | Output or limitation |
|---|---|---|
| `python -m tools.verify_pc` / `python -m tools.verify_renn` | CPU/GPU preflight plus isolated pipeline smoke tests | Pass/fail, logs, predictions, timing; no downloads |
| `python -m start` | One image through SmolVLM or Pixtral | CSV; optional raw JSON. Loads once per invocation, offline weights only |
| `python -m tools.annotate` | Browser-based human labeling | CSV and optional validated JSON manifest; no ML dependencies |
| `python -m tools.smoke_compare` | Known three-photo CPU smoke experiment, cascade or SmolVLM | Unique output directory with predictions and artifacts; fixed CPU settings and cached VLM revision |
| `python test_background_yolo.py` | Legacy cascade stage profiling; optional VLM profiling | JSON; processes all people, so not a controlled designated-runner accuracy test |
| `python -m benchmark.resource_check` | Preliminary RAM/GPU/disk/package inspection | JSON; does not load models or download |
| `benchmark/config.py` and `configs/` | Settings resolution and CPU/GPU presets | Library for the future shared runner; `start.py` does not accept these JSON presets |
| `benchmark/manifest.py` and `manifests/` | Validate image hashes, labels, splits, and target metadata | Smoke manifest and incomplete 100-slot scaffold |
| `benchmark/scoring.py` | Strict/normalized exact match and true Levenshtein CER | Library functions; failed trials retained in planned counts |
| `src/vlm_vision.py` | VLM image loading, explicit architecture loading, inference, legacy metrics | Called by `start.py` and the optional legacy profiler |
| `vision/background_yolo.py` | Existing YOLO/OCR stages | Local weights required; still loads models at import and writes fixed debug images |
| `vision/capture.py` | Screen-capture helper for the original application | Requires a desktop and `requirements/capture.txt` after cascade dependencies; not used by still-image experiments |
| `test_qwen.py` | Historical model experiment | Not a unit test or supported launch command; loads at import with incompatible model/class assumptions |
| `tests/` | Focused tests | Run only this directory; no inference or downloads |

## Verify the installed pipelines

After installing the relevant requirements and downloading weights, use the dedicated smoke-test programs. They do not install packages or download models. Each model runs sequentially in an isolated subprocess, with a 900-second timeout. Defaults use the known Quinton photo, full resolution, zero warmup, one pass, and two Torch threads.

**PC (CPU):**

```bash
python -m tools.verify_pc --check-only
python -m tools.verify_pc
```

Defaults check and execute the cascade and SmolVLM. The model directory is detected at `models/smolvlm-500m`, falling back to the pinned default Hub cache. To specify it explicitly or run all three known photos:

```bash
python -m tools.verify_pc --smolvlm-dir models/smolvlm-500m --all-images
```

**RENN (allocated GPU):** after [GPU setup](SETUP_RENN.md), replace the UUID below with your allocated GPU's real UUID:

```bash
python -m tools.verify_renn --gpu-uuid GPU-YOUR-ACTUAL-ALLOCATED-UUID --check-only
python -m tools.verify_renn --gpu-uuid GPU-YOUR-ACTUAL-ALLOCATED-UUID
```

RENN defaults to cascade, SmolVLM, and Pixtral NF4, all on the same single GPU. Local directories are detected at `models/smolvlm-500m` and `models/pixtral-12b`, or use `--smolvlm-dir` / `--pixtral-dir`. If a scheduler already supplies a single-GPU mask, omit `--gpu-uuid`; multiple assigned UUIDs require explicit selection within that mask. A single numeric scheduler mask is preserved and its actual Torch UUID cross-checked; multiple numeric indices require a single-GPU allocation or UUID-based mask. The program refuses CPU fallback, selection outside an inherited mask, busy GPUs, and insufficient Pixtral VRAM under its stated screening policy. These checks are not a GPU reservation or memory-fit guarantee.

Before downloading Pixtral, verify only the systems already installed:

```bash
python -m tools.verify_renn --systems cascade smolvlm --gpu-uuid GPU-YOUR-ACTUAL-ALLOCATED-UUID
```

Both programs accept `--systems`, `--check-only`, `--all-images`, `--threads`, `--timeout`, and `--output NEW_DIRECTORY`. A cascade-only test needs no VLM weights. The known three-photo manifest is used; newly added unlabeled photos are excluded.

**Local verification, 4 November:** the PC program completed cascade and SmolVLM inference on Quinton: both read 2300 (cascade 2.22 s, VLM 68.58 s). See [the verification report](results/verify-pc-20261104/REPORT.md). The 47 focused tests pass; real RENN GPU inference remains pending.

Each invocation creates a timestamped directory under `results/` with `REPORT.md`, `report.json`, `probe.log`, and per-system worker logs and raw results. It checks image hashes, dependency metadata, native Torch/torchvision operations, weight/shard structure, and Tesseract presence when applicable. Safetensors header/file bounds checks do not replace cryptographic verification. For RENN it also exercises a CUDA tensor, BF16 support, and bitsandbytes import when Pixtral is selected. Actual inference tests the full model path.

- **`preflight_passed`**: files/dependencies and tiny runtime checks passed; models were not loaded and inference is unverified.
- **`passed`**: every requested pipeline executed on the requested sample(s). Incorrect, empty, or missing-detection predictions lower accuracy but are separate from execution failure.
- **`failed`**: missing inputs, runtime/load errors, timeouts, or trial execution failures. Exit status is nonzero; inspect logs.

The cascade applies the same development target heuristic as the bounded comparison, not the legacy all-person accuracy rule. Passing a smoke test verifies only the selected setup and examples, not the final 100-photo experiment or real-time performance.

## Run a VLM on your PC

Linux/WSL, Python 3.12 CPU setup (annotation alone does not require a virtual environment):

```bash
python3 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/torch-cpu.txt
python -m pip install -r requirements/common.txt
```

Download SmolVLM **once**, explicitly:

```bash
hf download HuggingFaceTB/SmolVLM-500M-Instruct \
  --revision a7da5b986cb59b408707209984f360a5f4ad7e47 \
  --include '*.json' '*.safetensors' '*.model' '*.jinja' \
  --local-dir models/smolvlm-500m
```

Then run offline:

```bash
python -m start \
  --image media/quinton_bib_image.jpg \
  --model models/smolvlm-500m --cpu --ground_truth 2300 \
  --csv results/pc-smolvlm.csv --json results/pc-smolvlm.json
```

Repeat for another image with its correct label. Each command starts a new process and reloads the model; it does not redownload weights. Choose unique JSON names because `--json` overwrites that file. CSV appends with header validation; use a new filename for CSVs written by older versions.

`start` supports `--device cpu|cuda`, `--cpu`, `--model`, `--revision`, `--quantization nf4`, `--threads`, `--image`, `--prompt`, `--ground_truth`, `--csv`, and `--json`. Use `python -m start --help`. `--device cuda` refuses CPU fallback. Omitting the device preserves legacy automatic selection; explicit devices are recommended for recorded experiments. Hub IDs require already cached files; manual download directories must be passed as `--model`.

The VLM families are selected from checkpoint configuration: Idefics3 for this pinned SmolVLM, and Llava with Pixtral vision configuration for Pixtral. Arbitrary model IDs, including Qwen, are not supported by this two-model loader. Remote model code is disabled and weights are safetensors only.

## Run Pixtral on RENN's GPU

First complete [RENN setup](SETUP_RENN.md), including the CUDA tensor test and manual Pixtral download. On the assigned GPU node with its allocation/mask already established:

```bash
cd ~/run-research
source .venv-renn-gpu/bin/activate
python -m start \
  --image media/quinton_bib_image.jpg \
  --model models/pixtral-12b \
  --device cuda --quantization nf4 --ground_truth 2300 \
  --csv results/renn-pixtral.csv --json results/renn-pixtral.json
```

Pixtral uses approximately 25.4 GB of original downloaded weights. NF4 reduces runtime weight precision; it does not shrink the original transfer. GPU fit is not guaranteed. Run SmolVLM on the same allocated GPU with `--model models/smolvlm-500m --device cuda` for a hardware-matched diagnostic comparison. Quantized and unquantized variants must be labeled separately.

## Label photos

```bash
python -m tools.annotate --images media --csv annotations/bibs.csv
```

Open http://127.0.0.1:8765. Enter the designated runner's bib digits; keep leading zeros. Set the target description and split, and add a source/permission note for evaluation photos. Save and next persists a record. Previous lets you edit; navigation without saving discards edits. Restart with the same arguments to resume. The CSV is written atomically with a unique backup of the preceding version at each save.

Supported files: JPEG, PNG, WebP, including subdirectories. Skip/unreadable/ambiguous statuses must have blank bibs. Model predictions are not shown during labeling. Multiple-person photos need a reviewed target description; box drawing is not implemented. Images and CSV must stay paired; missing/changed images or duplicate content are refused.

Export after all scanned photos have reviewed digit labels:

```bash
python -m tools.annotate --images media --csv annotations/bibs.csv \
  --export annotations/reviewed100.json --expected-count 100
```

Use the actual scanned count for smaller collections. `media/` may now contain more than the original three photos, so a three-record export only works with a three-photo image directory. Unresolved photos block export rather than silently disappearing. CSV columns are `image_path,bib,sha256,status,target_identity,split,provenance`; import bibs as text in spreadsheet software. Unreadable-example scoring must be agreed before final evaluation. SSH tunneling is covered in the RENN guide.

## Reproduce the bounded CPU comparison

Install `requirements/cascade.txt` as well as common/CPU requirements, install native Tesseract, and supply local `models/yolov8s.pt` and `models/number632resol.pt`. The VLM smoke harness currently expects the pinned revision in the **default Hub cache**, unlike the directory-based `start` examples:

```bash
hf download HuggingFaceTB/SmolVLM-500M-Instruct \
  --revision a7da5b986cb59b408707209984f360a5f4ad7e47 \
  --include '*.json' '*.safetensors' '*.model' '*.jinja'
python -m tools.smoke_compare --system cascade --output results/cascade-new-run
python -m tools.smoke_compare --system vlm --output results/vlm-new-run
```

Run sequentially, with new output directories. This harness reads `benchmark/manifests/smoke3.json`, not the entire photo folder or annotation CSV. It forces CPU, two threads, no warmup, and one pass. The cascade selects the largest person crossing the horizontal midpoint, otherwise the closest horizontal person center. That development heuristic differs from the legacy all-person loop. Outputs include raw OCR, target boxes, prediction scores, timings, weight hashes, package versions, and process high-water RSS. Review target selections before interpreting results.

For stage profiling on one image:

```bash
python test_background_yolo.py --image media/quinton_bib_image.jpg \
  --ground-truth 2300 --cpu --warmup 0 --runs 1 \
  --output results/cascade-profile.json
```

`--vlm-model models/smolvlm-500m` adds optional VLM profiling through the offline loader; this legacy profiler has no NF4 option. Its accuracy check searches all detected bibs and is not the final target-level metric. Dropping `--cpu` allows YOLO default device selection; the profiler does not enforce a GPU explicitly.

## Metrics, checks, and limits

For `start.py`, JSON/CSV preserve raw VLM answers and extracted digits. `char_accuracy` is legacy SequenceMatcher similarity, **not CER**. `latency_sec` is synchronized generation time; `pipeline_sec` adds image loading, preprocessing, and decoding; `model_load_sec` is separate. `fps` is reciprocal generation latency, not measured live-video throughput. `visual_tokens` is a legacy tensor-shape estimate, not an authoritative visual-token count. Peak VRAM is the Torch allocator's generation-phase peak; the legacy CPU field is zero and is not a CPU memory measurement. Use the smoke report's separately named process RSS field for its CPU results.

```bash
python -m benchmark.resource_check
python -m unittest discover -s tests -p 'test_*.py'
```

Resource recommendations are preliminary. Cache completeness is unknown without a pinned file inventory. `--probe` opts into importing ML packages and tiny operations; avoid its UUID override when an inherited allocation mask must be preserved. Never use broad root test discovery with `test_qwen.py` present.

Final paper evaluation still needs a reviewed dataset, frozen target policy, shared multi-image runner, phase memory measurement, controlled repeated timing, and hybrid adapters. Use the [sprint plan](PROJECT_GOALS_AND_SPRINTS.md) as the current plan; the [October handoff](coding_agent_10_19.md) retains historical details and protocol requirements.
