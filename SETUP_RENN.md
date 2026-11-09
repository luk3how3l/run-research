# RENN GPU setup, from a fresh SSH session

This guide assumes Ubuntu/Debian Linux, Python 3.12, and an assigned NVIDIA GPU VM or compute node. You already have SSH access. Every command below runs **on RENN**, except the explicitly marked file-transfer commands. Run sections in order. No scheduler is assumed.

**Local validation:** 35 focused tests pass, the updated loader imports with Transformers 5.17.0, and the pinned cached SmolVLM loads offline on CPU. This new dependency profile and Pixtral CUDA/NF4 have not been exercised on RENN.

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
rsync -av --exclude=.git --exclude='.venv*' --exclude=__pycache__ --exclude=models --exclude=annotations --exclude=results ./ luke@student-luke:~/run-research/
```

Replace `luke@student-luke` with your SSH destination if different. This command has no deletion option. If rsync is unavailable, use SCP or your normal file-transfer application. Then return to the RENN terminal.

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

Annotation and the standard-library tests need none of these pip packages. The root `requirements.txt` now includes only `common.txt`; unrelated historical app packages have been removed.

## 5. Make the assigned GPU visible and verify CUDA

If a scheduler/container already assigns `CUDA_VISIBLE_DEVICES`, keep that assignment. If you control a dedicated GPU VM and no mask is assigned, list GPU UUIDs:

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv
```

Select your allocated UUID, not an arbitrary idle GPU. For example, replace the placeholder below with the **actual UUID printed above**:

```bash
export CUDA_VISIBLE_DEVICES=GPU-YOUR-ACTUAL-ALLOCATED-UUID
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

Run the corresponding laptop test with `python -m tools.verify_pc`; it forces CPU and defaults to cascade plus SmolVLM. See [the README](README.md#verify-the-installed-pipelines) for all options. GPU success cannot be established by running this on a CPU laptop; execute the RENN command on the actual allocated machine.

## 11. Annotation and lightweight verification

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m start --help
python -m tools.annotate --images media --csv annotations/bibs.csv
```

For the last command, leave the RENN terminal running. From a second terminal on the **laptop**:

```bash
ssh -N -L 8765:127.0.0.1:8765 luke@student-luke
```

Open http://127.0.0.1:8765 in the laptop browser. No GUI or GPU is needed for annotation. CSV labels are saved on RENN. The [README](README.md) explains labeling and export. Avoid root-level broad test discovery: `test_qwen.py` is a legacy inference script that loads a model on import.

After installation and successful smoke tests, record your environment:

```bash
mkdir -p results
python -m pip freeze > results/renn-environment.txt
nvidia-smi > results/renn-nvidia-smi.txt
```

Keep model files and personal photos outside commits. Remaining project work is listed in [the sprint plan](PROJECT_GOALS_AND_SPRINTS.md).
