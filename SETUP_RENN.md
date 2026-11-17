# RENN GPU setup, from a fresh SSH session

This guide assumes Ubuntu/Debian Linux, Python 3.12, and an assigned NVIDIA GPU VM or compute node. You already have SSH access. Every command below runs **on RENN**, except the explicitly marked file-transfer commands. Run sections in order. No scheduler is assumed.

**Local validation:** 77 repository tests pass in the local project virtual environment, including 16 annotation tests and transaction recovery checks. Browser-script save/error/empty-queue smoke checks pass. Earlier loader validation showed that the updated loader imports with Transformers 5.17.0 and the pinned cached SmolVLM loads offline on CPU. This new dependency profile and Pixtral CUDA/NF4 have not been exercised on RENN.

**Already have RENN set up?** Keep your existing working environment. For the new photo workflow, go to [section 11](#11-annotation-and-lightweight-verification), then [the multi-image comparison](#running-a-multi-image-three-model-comparison). For a complete copy/paste walkthrough, use [Annotate photos and prepare a benchmark batch](ANNOTATION_WORKFLOW.md). Annotation can be completed before the GPU setup; no GPU or model downloads are needed to label photos.

The previous `AutoModelForVision2Seq` error is fixed in the updated source by using explicit model classes. The separate `torch.cuda.is_available() == False` error needs the hardware checks below; installing Python packages alone cannot attach a GPU to a VM.

## 1. Get the updated source and inspect the machine

```bash
cd ~/run-research
hostname
python3 --version
nvidia-smi
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}"
free -h
df -h .
```

**Continue only when `nvidia-smi` shows the GPU assigned to you.** If it is missing, cannot communicate with the driver, or shows no devices, verify that you connected to the GPU machine and ask the RENN administrator to check VM GPU passthrough and the host driver. Don't try to fix host GPU drivers with pip. A login-only VM may have no GPU even though you can SSH into it.

Get the updated repository through your normal Git workflow. Changes made on the laptop must be committed/pushed before `git pull` can retrieve them. Alternatively, from the **laptop's repository directory**, copy the working source without replacing the VM environment:

```bash
rsync -av --exclude=.git --exclude='.venv*' --exclude=__pycache__ --exclude=models --exclude=media --exclude=annotations --exclude=results ./ luke@student-luke:~/run-research/
```

Replace `luke@student-luke` with your SSH destination if different. This source-only command excludes media, annotations, models, and results so it does not replace an existing dataset. Transfer selected new photos separately using [the annotation guide](ANNOTATION_WORKFLOW.md#3-add-a-small-batch-of-new-images). This command has no deletion option. If rsync is unavailable, use SCP or your normal file-transfer application. Then return to the RENN terminal.

For a fresh source-only rsync installation, also copy the three original smoke-test photos separately from the **laptop repository**, if they are still present at their original paths:

```bash
rsync -av --ignore-existing media/quinton_bib_image.jpg media/hannah_bib_test.jpg media/ryan_rockie_bib.jpg luke@student-luke:~/run-research/media/
```

Create `~/run-research/media/` on RENN first if needed (`ssh luke@student-luke 'mkdir -p ~/run-research/media'`). The smoke-test commands below use these original paths. A custom reviewed batch uses its exported manifest instead.

## 2. Install operating-system prerequisites

On an Ubuntu/Debian VM where you have sudo:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git rsync ca-certificates libgl1 libglib2.0-0 tesseract-ocr
```

Tesseract is needed for the original cascade, not for VLM-only inference. If you lack sudo, ask the administrator to install missing system packages. Do not install a display server or CUDA toolkit just for annotation or the pip-based inference commands here.

## 3. Create an isolated Python environment

Keep the failing `.venv-renn` intact while testing this new environment:

```bash
cd ~/run-research
python3 -m venv .venv-renn-gpu
source .venv-renn-gpu/bin/activate
python -m pip install --upgrade pip
python -c 'import sys; print(sys.executable)'
```

The last line must point inside `~/run-research/.venv-renn-gpu/`. Use `python -m pip` throughout so installation and execution use the same interpreter. Reconnect later with:

```bash
cd ~/run-research
source .venv-renn-gpu/bin/activate
```

## 4. Install the research dependencies

For the NVIDIA L4 machine with a driver supporting CUDA 12.6, use this pinned pair from [PyTorch's official version instructions](https://pytorch.org/get-started/previous-versions/):

```bash
python -m pip install -r requirements/torch-renn-cu126.txt
python -m pip install -r requirements/common.txt
python -m pip install -r requirements/cascade.txt
python -m pip install -r requirements/gpu-nf4.txt
python -m pip check
```

Choose only the files you need:

| File | Purpose |
|---|---|
| `requirements/torch-renn-cu126.txt` | Torch 2.10.0 + torchvision 0.25.0, CUDA 12.6 wheels |
| `requirements/torch-cpu.txt` | Same pair with CPU wheels, for a CPU laptop instead |
| `requirements/common.txt` | Pinned Transformers 4.57.6, Hub CLI, processor/image/runtime dependencies |
| `requirements/cascade.txt` | YOLO, OpenCV, OCR bindings, plotting dependencies |
| `requirements/gpu-nf4.txt` | Optional bitsandbytes for Pixtral 4-bit loading |
| `requirements/capture.txt` | Optional laptop desktop capture helper; skip on headless RENN |

The 12.6 profile is a concrete installation candidate for the earlier L4/driver snapshot, not proof of this VM's current compatibility. If the current GPU or driver differs, use the administrator-supported matching Torch/torchvision pair from the official page and record the change. Do not reinstall the reported 13.2 build blindly. These pinned requirements are not a fully resolved cross-platform lockfile and have not been installed/tested on RENN by this assistant.

Annotation and its focused tests need none of these pip packages. The full suite includes image validation tests that need Pillow (installed by `common.txt`). The root `requirements.txt` now includes only `common.txt`; unrelated historical app packages have been removed.

## 5. Make the assigned GPU visible and verify CUDA

Physical GPU UUIDs identify devices independently of their numeric indices in `nvidia-smi`. `CUDA_VISIBLE_DEVICES` restricts which devices CUDA can access and determines their logical ordering. PyTorch's `cuda:0` is the first device visible under that mask, not necessarily physical GPU index 0. On university infrastructure, use only your allocated GPU and never bypass a scheduler-provided mask.

If a scheduler/container already assigns `CUDA_VISIBLE_DEVICES`, keep that assignment. If you control a dedicated GPU VM and no mask is assigned, list GPU UUIDs:

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv
```

Select your allocated UUID, not an arbitrary idle GPU. For example, replace the placeholder below with the **actual UUID printed above**:

```bash
export CUDA_VISIBLE_DEVICES=GPU-c17b88c2-24ec-4249-f527-ae8b09e8942a
```

An empty value hides all GPUs. A literal placeholder such as `GPU-REPLACE-WITH-YOUR-ALLOCATED-UUID` also prevents selection. If you previously copied a placeholder and this is your dedicated VM, replace it with the real UUID. Do not clear a scheduler's mask to expose other GPUs.

Now test:

```bash
python - <<'PY'
import os
import torch
print('Mask:', os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>'))
print('Torch:', torch.__version__)
print('Torch CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('Visible GPU count:', torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit('STOP: fix GPU/driver/mask access before loading Pixtral')
print('GPU:', torch.cuda.get_device_name(0))
print('BF16 supported:', torch.cuda.is_bf16_supported())
print('Test tensor:', (torch.ones(2, device='cuda') + 1).cpu())
PY
```

Expected: CUDA available `True`, at least one visible GPU, and a tensor containing two `2` values. The supplied CUDA inference profile also requires BF16. Logical `cuda:0` is the first GPU inside the assigned mask, not necessarily physical GPU 0.

If this still reports `False`:

- `nvidia-smi` fails: wrong node, missing passthrough, driver, or permissions; ask the administrator.
- `nvidia-smi` works but the mask is empty/wrong: correct your own VM mask or ask about the assigned scheduler mask.
- Torch prints a CPU build or `torch.version.cuda` is `None`: verify the active environment and reinstall its CUDA Torch profile.
- CUDA build and mask look correct: run `python -c 'import torch; torch.cuda.init()'` to expose the driver initialization error and send that error plus `nvidia-smi` to the administrator.

[PyTorch's availability API](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available) reports runtime availability; printing a CUDA build version does not establish it.

## 6. Manually download SmolVLM once

This is an explicit download, about 1 GB of model files. Check available space and quota first. Choose a model directory you own; commands below use `models/`, which Git ignores.

```bash
mkdir -p models/smolvlm-500m
hf download HuggingFaceTB/SmolVLM-500M-Instruct \
  --revision a7da5b986cb59b408707209984f360a5f4ad7e47 \
  --include '*.json' '*.safetensors' '*.model' '*.jinja' \
  --local-dir models/smolvlm-500m
```

The Hub CLI is installed by `common.txt`. If authentication is required, run `hf auth login` and follow the prompt; do not put tokens into scripts. Repeating the same download command reuses completed local files and can resume missing data. See the [official Hub CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

Run the first GPU smoke test **offline**, explicitly specifying a real image:

```bash
python -m start \
  --image media/quinton_bib_image.jpg \
  --model models/smolvlm-500m \
  --device cuda --ground_truth 2300 \
  --csv results/renn-smolvlm.csv --json results/renn-smolvlm.json
```

If you choose a CPU test instead, replace `--device cuda` with `--cpu`. Bare `python -m start` uses the default photo but expects SmolVLM in the default Hub cache; it does not discover the manual directory automatically. Use the explicit `--model` commands here.

## 7. Manually download Pixtral once

Use the **Transformers-converted** checkpoint and immutable revision below. It contains about **25.4 GB of original weights in six shards** ([pinned file listing](https://huggingface.co/mistral-experimental/pixtral-12b/tree/c2756cbbb9422eba9f6c5c439a214b0392dfc998)). NF4 quantization happens during loading and does not make this original download smaller.

Allow at least 40 GiB of free working space on the selected filesystem, check your personal quota, and confirm the GPU test above works before spending time downloading. This is planning headroom, not a measured runtime requirement or guarantee. The manual CLI command has no enforced research download cap; running it explicitly initiates the large transfer.

```bash
df -h .
mkdir -p models/pixtral-12b
hf download mistral-experimental/pixtral-12b \
  --revision c2756cbbb9422eba9f6c5c439a214b0392dfc998 \
  --include '*.json' '*.safetensors' '*.model' '*.jinja' \
  --local-dir models/pixtral-12b
```

To download from a browser instead, open the pinned file listing, download all six `model-0000N-of-00006.safetensors` shards plus every `.json` file, and place them together in `models/pixtral-12b/`. Include the model index, tokenizer, chat template, processor, and preprocessing files. Do not download Git-LFS pointer text in place of actual weights. The CLI route is less error prone.

Check every indexed weight shard against its size in the pinned Hub metadata (this fetches metadata only, not weights):

```bash
python - <<'PYVERIFY'
import json
from pathlib import Path
from huggingface_hub import HfApi
root = Path('models/pixtral-12b')
index = json.loads((root / 'model.safetensors.index.json').read_text())
names = sorted(set(index['weight_map'].values()))
info = HfApi().model_info('mistral-experimental/pixtral-12b',
    revision='c2756cbbb9422eba9f6c5c439a214b0392dfc998', files_metadata=True)
sizes = {f.rfilename: f.size for f in info.siblings}
assert len(names) == 6
for name in names:
    actual = (root / name).stat().st_size
    assert sizes[name] is not None and actual == sizes[name], f'Missing/truncated shard: {name}'
    print(name, actual)
print('All six shard sizes match the pinned metadata; this is not a cryptographic integrity check.')
PYVERIFY
```

## 8. Run Pixtral offline on the assigned GPU

First check current memory and utilization with `nvidia-smi`. The profile proposes at least 18 GiB free GPU memory and 12 GiB available RAM; these are conservative screening thresholds, not a fit guarantee. Full-image processing can still run out of memory. Never change image size or quantization silently between comparisons.

```bash
python -m start \
  --image media/quinton_bib_image.jpg \
  --model models/pixtral-12b \
  --device cuda --quantization nf4 --ground_truth 2300 \
  --csv results/renn-pixtral.csv --json results/renn-pixtral.json
```

The loader uses `LlavaForConditionalGeneration` with Pixtral vision configuration and bitsandbytes NF4, consistent with the [Transformers Pixtral example](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/pixtral). The entire model is assigned to logical GPU 0; there is no automatic second-GPU use or CPU/disk spill. If it fails, keep the error and adjust a separately named experiment rather than silently changing the test.

The CLI loads weights once per invocation and performs one image prediction. The JSON retains raw text; the CSV records the extracted bib and timings. `latency_sec` measures generation, `pipeline_sec` includes image loading/preprocessing/generation/decoding, and `model_load_sec` is separate. These are diagnostic single-image runs, not yet the frozen 100-image benchmark. CUDA/NF4 execution of this updated loader still needs a real RENN smoke test.

## 9. Set up the original cascade

Copy the weights from the **laptop** (they are not tracked by Git):

```bash
ssh luke@student-luke 'mkdir -p ~/run-research/models'
scp models/yolov8s.pt models/number632resol.pt luke@student-luke:~/run-research/models/
```

Then on **RENN**:

```bash
tesseract --version
ls -lh models/yolov8s.pt models/number632resol.pt
python test_background_yolo.py \
  --image media/quinton_bib_image.jpg --ground-truth 2300 \
  --warmup 0 --runs 1 --output results/renn-cascade.json
```

This legacy profiler uses YOLO's default device selection and handles **all detected people**. It does not expose an explicit GPU argument or enforce the designated-runner policy. Respect the inherited mask and inspect the device reported by YOLO; do not call it a controlled one-runner GPU comparison. The CPU three-photo comparison in the README uses a different bounded harness with explicit target selection.

## 10. Verify the GPU pipelines end to end

For GPU diagnosis, inspect the physical devices, inherited allocation mask, and runtime verification together:

```bash
cd ~/run-research
source .venv-renn-gpu/bin/activate

nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader

printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}"

python -m tools.verify_renn \
  --gpu-uuid GPU-c17b88c2-24ec-4249-f527-ae8b09e8942a \
  --check-only
```

Replace the placeholder with your authorized UUID. `nvidia-smi` may report `GPU-c17b88c2-24ec-4249-f527-ae8b09e8942a` while PyTorch reports `c17b88c2-24ec-4249-f527-ae8b09e8942a`. The previous raw-string comparison falsely rejected this case even when CUDA could access the selected GPU. Verification now validates both complete UUIDs, removes only an optional leading `GPU-`, and compares their content without regard to letter case. A prefix-only difference is not a GPU mismatch. A genuine UUID mismatch, unavailable or malformed identity, or allocation-mask violation still blocks execution; GPU names and partial UUIDs cannot establish identity.

After downloading the selected weights and passing the CUDA tensor test, run the RENN verification program. Set your actual allocated UUID, or omit `--gpu-uuid` when inheriting a single-GPU scheduler mask:

```bash
python -m tools.verify_renn \
  --gpu-uuid GPU-YOUR-ACTUAL-ALLOCATED-UUID --check-only
python -m tools.verify_renn \
  --gpu-uuid GPU-YOUR-ACTUAL-ALLOCATED-UUID
```

Default systems: cascade, SmolVLM, Pixtral NF4. Default image: Quinton, label 2300. Use `--all-images` for all three known labeled photos, or `--systems cascade smolvlm` to test only those two before Pixtral is downloaded. Directory overrides:

```bash
python -m tools.verify_renn \
  --gpu-uuid GPU-YOUR-ACTUAL-ALLOCATED-UUID \
  --smolvlm-dir models/smolvlm-500m --pixtral-dir models/pixtral-12b \
  --all-images --timeout 1200
```

All requested systems must have their files/dependencies ready before any models are loaded. This program never downloads weights, never falls back to CPU, and checks GPU allocation masks. Single numeric scheduler masks are preserved and the actual Torch GPU UUID is cross-checked; multiple numeric indices require a single-GPU allocation or UUID-based mask. It refuses devices above 10% utilization or 1 GiB used; Pixtral additionally requires 18 GiB free by screening policy. Availability may change after the check; stop if your allocation changes. It is not a scheduler or a guarantee that the model fits.

Each system runs in a separate subprocess, sequentially. Results go into a new timestamped `results/verify-renn-*` folder. Read `REPORT.md` first, then `report.json` and each system's `worker.log` / `result.json`. Check-only success is named `preflight_passed`, not proof of inference. A full `passed` means the selected pipelines executed; exact bib accuracy is reported separately and may be low. Failures/timeouts return a nonzero exit code and retain available logs/results.

Use the exact directory printed by the verifier to inspect the runtime identity and overall result:

```bash
# Replace the timestamp with the directory printed by your run.
cat results/verify-renn-TIMESTAMP/probe.json
cat results/verify-renn-TIMESTAMP/report.json
cat results/verify-renn-TIMESTAMP/REPORT.md
```

`probe.json` contains the PyTorch GPU UUID and effective CUDA mask when the runtime probe succeeds. `report.json` contains the selected physical GPU, runtime probe data, status, and errors. `REPORT.md` provides the readable status and errors. If preflight stops before the probe, `probe.json` may be absent; inspect `report.json`, `REPORT.md`, and `probe.log` if present. Successful check-only output is:

```text
Status: preflight_passed
```

This verifies preflight checks and the tiny runtime probe only. It does not load model weights or establish successful model inference; run the full verification command separately for that.

Run the corresponding laptop test with `python -m tools.verify_pc`; it forces CPU and defaults to cascade plus SmolVLM. See [the README](README.md#verify-the-installed-pipelines) for all options. GPU success cannot be established by running this on a CPU laptop; execute the RENN command on the actual allocated machine.

## 11. Annotation and lightweight verification

Use [the full step-by-step annotation guide](ANNOTATION_WORKFLOW.md) for photo selection, copying from the imported dataset, label corrections, recovery, export, and laptop-to-RENN transfer. For annotation directly on RENN:

1. Prepare the queue from the repository root:

   ```bash
   cd ~/run-research
   mkdir -p media/unannotated media/annotated media/needs_review annotations benchmark/manifests
   ```

2. Copy selected new photos into `media/unannotated/`, keeping imported originals in `media/drive-test-trainbib/`. Use real filenames and start with a small batch. Do not queue an image again if it is already in the CSV. Leave the three original smoke-test photos at their existing paths.

3. Start the server and leave this RENN terminal running:

   ```bash
   python3 -m tools.annotate --csv annotations/bibs.csv
   ```

4. In a second terminal on the **laptop**, open the tunnel (substitute your SSH destination):

   ```bash
   ssh -N -L 8765:127.0.0.1:8765 luke@student-luke
   ```

5. Open **http://127.0.0.1:8765** on the laptop. Enter bib digits, target runner, development/evaluation split, and provenance. Keep leading zeros. Choose **reviewed** only for a confident reading; for **skipped**, **unreadable**, or **ambiguous**, clear the bib. Evaluation records require a source/permission note.

6. Click **Save and next**. Reviewed photos move to `media/annotated/`; the other three saved statuses move to `media/needs_review/`. The authoritative `annotations/bibs.csv` records the new path and metadata. Counts update and the next pending photo appears. A failed save shows an error and does not advance. Pending images stay in the queue.

7. Restart with the same command to resume. Select **All images / edit labels** to correct saved records or revisit needs-review images. The empty-queue message means no pending rows remain, not that all photos were reviewed. Stop with Ctrl+C before exporting a frozen batch, then continue below.

The default scan does not ingest the entire imported dataset. Older CSV rows remain available even if their paths are outside the managed directories. Avoid `--images media`: it would recursively scan imported originals and may introduce duplicates or unintended pending records. Keep the CSV, its backups, images, and any `.transaction` journal together. Interrupted moves recover on restart; see [recovery instructions](ANNOTATION_WORKFLOW.md#6-resume-correct-labels-or-recover-from-an-error).

For focused annotation verification without extra packages:

```bash
python3 -m unittest discover -s tests -p 'test_annotation.py'
```

For the full suite, use the installed research environment (Pillow is needed by some image validation tests):

```bash
source .venv-renn-gpu/bin/activate
python -m unittest discover -s tests -p 'test_*.py'
python -m start --help
```

Avoid root-level broad test discovery: `test_qwen.py` is a legacy inference script that loads a model on import. After successful installation and smoke checks, record the environment:

```bash
mkdir -p results
python -m pip freeze > results/renn-environment.txt
nvidia-smi > results/renn-nvidia-smi.txt
```

Keep model files and personal photos outside commits. Remaining project work is listed in [the sprint plan](PROJECT_GOALS_AND_SPRINTS.md).

## 12. Run the comparison and find results

After the setup and smoke checks above, run from the repository root on your
allocated GPU machine:

```bash
cd ~/run-research
source .venv-renn-gpu/bin/activate
python -m benchmark.runner --device cuda \
  --gpu-uuid GPU-YOUR-ACTUAL-ALLOCATED-UUID \
  --smolvlm-dir models/smolvlm-500m \
  --pixtral-dir models/pixtral-12b
```

Replace the GPU placeholder with your allocated UUID, respecting the scheduler's
existing mask. This runs the legacy cascade, SmolVLM, and Pixtral NF4 sequentially,
prints progress, and saves a new `results/comparison-TIMESTAMP/` folder.
It does not download weights. Use `--timeout 1800` to set the per-worker timeout
(default 1800 seconds), or `--systems cascade smolvlm` to select fewer systems.

For a browser **Run comparison** button, start this on RENN instead:

```bash
cd ~/run-research
source .venv-renn-gpu/bin/activate
python -m tools.compare_button
```

Leave that terminal open. In a second terminal on the laptop, forward the button's
port using the same SSH host you use for RENN:

```bash
ssh -N -L 8766:127.0.0.1:8766 luke@student-luke
```

Open http://127.0.0.1:8766 on the laptop. Select **cuda**, enter your allocated GPU
UUID, enter the model directories above, and click **Run comparison**. Relative
model paths are relative to the repository on RENN. Progress and the final table
appear in the RENN terminal that launched the button. CPU is the initial selection;
CPU runs mark Pixtral blocked. The page reports completion, while detailed results
remain in the terminal and files.

### Result files

The runner now writes per-image `comparison.csv` and per-split `summary.csv`.
The browser button still uses the three development photos by default; use the
CLI below to select an annotated batch. Earlier experiment folders retain their
original schemas and should not be overwritten.

## Running a multi-image three-model comparison

Use the existing `.venv-renn-gpu` and local model directories. No dependency
reinstallation is needed for this refactor. Confirm your **current** GPU allocation
with the administrator/scheduler before running; idle readings do not authorize a
device. Keep inherited `CUDA_VISIBLE_DEVICES`. Replace the UUID placeholder below
with the complete UUID of your currently allocated device.

### A. Confirm annotation status

Complete [section 11](#11-annotation-and-lightweight-verification) or the [annotation walkthrough](ANNOTATION_WORKFLOW.md) first. On RENN, inspect the authoritative CSV rather than counting imported source files:

```bash
cd ~/run-research
source .venv-renn-gpu/bin/activate
python - <<'PYSTATUS'
import csv
from collections import Counter
from pathlib import Path
path = Path('annotations/bibs.csv')
if not path.exists():
    raise SystemExit('Save annotations in the browser first; no CSV exists yet.')
with path.open(newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
reviewed = [r for r in rows if r['status'] == 'reviewed']
print('Status:', dict(Counter(r['status'] for r in rows)))
print('Reviewed splits:', dict(Counter(r['split'] for r in reviewed)))
print('Reviewed export count:', len(reviewed))
PYSTATUS
```

Preserve development status for photos used in tuning. Mark genuinely held-out photos evaluation, with provenance/permission notes. Resolve label uncertainty through human review. Keep exclusions visible and document them before evaluating predictions.

### B. Export the reviewed batch

If the command above reports **18 reviewed** photos, export as follows; otherwise replace `18` with the actual reviewed count:

```bash
python -m tools.annotate --csv annotations/bibs.csv \
  --export benchmark/manifests/renn-batch-01.json --expected-count 18
```

Export includes all reviewed rows from this CSV, with both splits and their current paths. Pending, unreadable, ambiguous, and skipped records remain in the CSV and are excluded from the export. No reviewed records or a wrong expected count causes failure. Custom benchmark manifests still require every included record to be reviewed and valid. The legacy smoke manifest alone retains its `user_supplied` exception.

Use a new filename for each frozen batch. Finish corrections and image moves before exporting; later CSV edits do not change existing manifest snapshots. If annotations were created on your laptop, [transfer the CSV, manifest, and referenced images](ANNOTATION_WORKFLOW.md#9-if-annotated-locally-transfer-the-dataset-to-renn) before running on RENN. The manifest alone does not contain the image data.

### C. Check readiness, then perform inference

Run the same frozen ordered records through all three systems:

```bash
python -m benchmark.runner \
  --device cuda \
  --gpu-uuid GPU-YOUR-ACTUAL-ALLOCATED-UUID \
  --manifest benchmark/manifests/renn-batch-01.json \
  --systems cascade smolvlm pixtral \
  --smolvlm-dir models/smolvlm-500m \
  --pixtral-dir models/pixtral-12b \
  --gpu-wait-seconds 30 \
  --gpu-poll-seconds 5 \
  --timeout 1200 \
  --check-only
```

After `preflight_passed`, repeat the command without `--check-only` for inference. Keep your actual allocated UUID and inherited mask. Each invocation writes a new results directory. The runner without `--manifest` and the browser comparison button use the original three photos, not your annotation queue.

`tools.verify_renn` and `tools.verify_pc` also accept `--manifest`. Explicit manifests
use every validated record, regardless of `--all-images`. With no manifest, the
runner retains the three-image smoke comparison, and direct verifiers retain their
single-image default (`--all-images` selects all three). All paths in a manifest
are relative to the repository root. Missing files, changed SHA256 hashes,
duplicate paths/content/IDs, invalid labels or missing review data stop validation
before models load. The runner freezes exact manifest bytes and digest in its new
output folder, then workers recheck image hashes before each inference.

### D. Inspect the completed run

Use the exact directory printed by the runner. Each comparison uses a new `results/comparison-<unique-run-id>/` directory:

| Artifact | Contents |
|---|---|
| `manifest.json` | Frozen source manifest with ordered records, targets, splits and provenance |
| `comparison.csv` | One row per planned image per system; 18 images × 3 systems = 54 rows |
| `summary.csv` | One row per system per dataset split |
| `metadata.json` | Git SHA, manifest digest, timestamps, versions, GPU/runtime observations, model configurations and timing definitions |
| `REPORT.md` | Status, errors and scientific limitations |
| `gpu-preflight.json` | Every GPU screening attempt with timestamp, UUID/index, memory, utilization, mask, process observations and policy decision |
| `SYSTEM/report.json` | Verifier report, dependencies, runtime identity and incremental results |
| `SYSTEM/SYSTEM/result.json` | Worker predictions, scores, resolutions, timing and memory measurements |
| `SYSTEM/SYSTEM/worker.log` | Offline load/inference logs |

The nested worker directories preserve the existing verifier layout. Early failures
may leave no worker directory. A blocked CPU Pixtral still has per-image rows.

Inspect CSVs with the standard library, preserving bib strings:

```bash
python - <<'PYCSV'
import csv
from pathlib import Path
run = sorted(Path('results').glob('comparison-*'))[-1]  # or choose an exact run
for name in ('comparison.csv', 'summary.csv'):
    with (run / name).open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    print(run / name, 'rows:', len(rows))
    for row in rows[:3]:
        print(row)
PYCSV
```

`comparison.csv` includes image identifiers/hash/split/target/ground truth, system
and model identity, raw and extracted predictions, strict/normalized exact and CER,
execution status/error, pipeline/generation seconds and CUDA allocated/reserved
peaks. Failed and unattempted rows remain visible; absent numeric measurements are
blank. Import bib columns as **text** in spreadsheet software, which may otherwise
discard leading zeros. Cascade raw OCR is captured before its existing digit
filter; scoring retains the legacy cascade output and established scoring rules.

`summary.csv` counts planned, attempted, successful, failed and unattempted images;
normalized accuracy divides correct predictions by **all planned images in that
split**. Timing statistics use successful executions only; P95 uses nearest rank.
Load time is system-wide and repeated across its split rows. Generation timing is
VLM `model.generate` only; pipeline timing includes decode/preprocessing/inference/
postprocessing and excludes model load. GPU peaks include resident weights and are
measured during the recorded phase. No live-video throughput is inferred.
Check-only rows are unattempted and do not establish measured accuracy.

Scientific comparability: every system gets the same labeled records, but the
unchanged cascade selects the largest person crossing the horizontal midpoint,
otherwise the nearest horizontal person center. VLMs receive the unchanged central
foreground runner prompt and full RGB image using their own local processors.
Descriptions do not steer selection. Every cascade row is flagged for target review;
VLM rows with differing descriptions are also flagged. Inspect selected bounding
boxes/debug crops against human descriptions before interpreting accuracy as target
recognition. Target agreement is **not automatically guaranteed**. Metadata records
processor configuration/resolution differences, BF16/FP32 settings and Pixtral NF4.
Pinned model revisions are recorded as expected revisions; offline structural checks
cannot cryptographically authenticate arbitrary supplied local weights. Cascade
weight hashes are in worker results. Preserve download provenance for local weights.

For GPU-busy failures, inspect the actual attempt history and contemporaneous state:

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}"
```

Use the exact run's `gpu-preflight.json`, `SYSTEM/report.json` and `probe.log`.
The default waiting window is zero; `--gpu-wait-seconds 30 --gpu-poll-seconds 5`
permits bounded retries without changing thresholds: utilization ≤10%, used memory
≤1024 MiB, plus ≥18432 MiB free for Pixtral. Each retry revalidates identity and
allocation. Unavailable measurements fail closed; `null` process information means
unavailable, while `[]` means measured no compute processes. An empty process list
alone never passes screening. Query durations add overhead to the polling window.
No CPU fallback or alternate GPU is selected. An idle later reading cannot explain
an earlier busy sample or prove a CUDA leak. Each worker exits and is reaped before
the next starts; timeout cleanup kills only that worker's own process group.
The parent never uses `torch.cuda.empty_cache()` to simulate resource release.

Preserve original experimental folders and annotation backups. Use a new output
path for every rerun (`--output` rejects existing directories); retain manifest,
metadata, logs and CSVs together. Never commit OAuth tokens, model weights, private
photos or annotation datasets. Review manifests for private paths/provenance before
committing them.

Optional cloud upload is a separate post-inference step. If you already configured
an rclone remote named `gdrive`, choose the exact completed run directory:

```bash
rclone listremotes
rclone copy results/comparison-YOUR-RUN-ID gdrive:RENN/comparison-YOUR-RUN-ID \
  --include comparison.csv --include summary.csv --include metadata.json \
  --include REPORT.md --include gpu-preflight.json --include manifest.json \
  --dry-run
# After inspecting the dry run, repeat without --dry-run to upload.
```

This copies artifacts without deleting earlier remote results. Upload neither
private images nor rclone configuration/OAuth credentials. Inference performs no
cloud upload and requires no Drive authorization.
