# Laptop and RENN setup

Updated 2 November 2026. Sprint 0 (annotation) is implemented. The shared model runner and SmolVLM/Pixtral benchmark adapters are still pending. GPU setup below prepares the environment; it is not a completed model experiment.

## 1. Start annotation on your laptop

Use Python 3.10 or newer and a browser. No pip packages, GPU, desktop GUI toolkit, or model downloads are needed. Open a terminal in the repository:

```bash
cd /path/to/run-research
python3 -m tools.annotate --images media --csv annotations/bibs.csv
```

Open **http://127.0.0.1:8765** in your browser. On Windows, use `py -3` instead of `python3` if needed.

1. Read the bib of the central foreground runner. When multiple people make that target ambiguous, mark the image ambiguous rather than guessing.
2. Enter the bib as digits, preserving leading zeros. Confirm or edit the target description.
3. Keep the existing three photos in `development`. Use `evaluation` for new final-test photos and enter their source/permission note. Reserve separate development examples if tuning more systems.
4. Press **Save and next**. Previous and Next without saving navigate without persisting edits; save before moving away from an edited record.
5. For unreadable, ambiguous, or skipped photos, clear the bib and select the status before saving.
6. Stop the terminal process with Ctrl+C. Run the same command to resume. It opens the first image not marked reviewed; Previous lets you revise earlier labels.

CSV columns: `image_path,bib,sha256,status,target_identity,split,provenance`. Each save writes atomically and preserves the preceding CSV as a uniquely named `.backup-*` file. Back up the annotation directory elsewhere regularly. If opening CSV in spreadsheet software, import the bib column as text so `007` stays `007`.

Put additional JPEG, PNG, or WebP photos in the selected image directory; subdirectories are scanned too. Images must remain inside the repository root. Restart to discover new images. Duplicate content, missing previously annotated images, and images changed since annotation cause an error instead of silently reusing labels. Keep the CSV paired with its full collection directory.

This sprint uses a reviewed textual target description, not bounding-box drawing. The cascade target-matching implementation remains a later sprint.

## 2. Export labels for experiments

For the current three photos, after reviewing all three:

```bash
python3 -m tools.annotate --images media --csv annotations/bibs.csv --export annotations/smoke-reviewed.json --expected-count 3
```

For the complete collection, use the same command with a new output name and `--expected-count 100`. Export includes every scanned photo; unresolved entries block export. There is no silent exclusion of difficult photos. The treatment of unreadable bibs in final scoring must be decided before the full evaluation.

The exported JSON is compatible with `benchmark.manifest.validate`. Paths are relative to the repository root, so preserve that layout when transferring to RENN. Export validates hashes, reviewed labels, count, and split metadata. It does not validate that the human entered the correct number; review your labels independently.

## 3. Transfer to RENN

Use your university's assigned SSH hostname and username; the examples use placeholders. Copy this updated checkout, because uncommitted tools and docs will not appear in a fresh Git clone. From your laptop, for example:

```bash
rsync -av --exclude=.git --exclude=.venv --exclude=.venv-renn --exclude=__pycache__ --exclude=models --exclude=annotations run-research/ USER@RENN_HOST:~/run-research/
rsync -av run-research/annotations/ USER@RENN_HOST:~/run-research/annotations/
```

Run these from the directory containing `run-research`. Use SCP or your normal file-transfer application if rsync is unavailable. Copy private evaluation photos separately with the same relative paths. Models are excluded deliberately; transfer the required local detector files separately when preparing the cascade:

```bash
rsync -av run-research/models/yolov8s.pt run-research/models/number632resol.pt USER@RENN_HOST:~/run-research/models/
```

The model files are ignored by Git. Copying only the repository through Git does not copy them.

## 4. Annotate remotely without a VM desktop

On RENN:

```bash
cd ~/run-research
python3 -m tools.annotate --images media --csv annotations/bibs.csv
```

On your laptop, in another terminal:

```bash
ssh -N -L 8765:127.0.0.1:8765 USER@RENN_HOST
```

Open http://127.0.0.1:8765 on the laptop. CSV files are saved on RENN in this case. Stop any local annotation server using the same port first, or select another port with `--port`. The tool binds to loopback; no VM firewall opening is needed. If the compute node is behind a login gateway, use your university's prescribed jump-host route; the tunnel destination must be the machine running the annotation server.

## 5. Prepare the RENN GPU environment

Run GPU operations on your allocated GPU VM/compute node, respecting university allocation rules. Annotation does not require a GPU. These commands inspect the current machine:

```bash
hostname
python3 --version
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv
free -h
df -h .
```

Confirm storage quota separately. The earlier two-L4 snapshot is historical, not a current allocation.

Create an isolated environment if there is not already an approved working environment:

```bash
cd ~/run-research
python3 -m venv .venv-renn
source .venv-renn/bin/activate
python -m pip --version
```

If `venv` is unavailable, use the university-provided Python/module environment or ask the administrator for its supported setup. Do not replace system Python.

For later inference sprints, choose matching Torch and torchvision CUDA wheels using the [official PyTorch installer](https://pytorch.org/get-started/locally/). Select the build supported by this VM's driver and GPU. Record the exact chosen versions and installation command; do not infer the Torch runtime from the CUDA label printed by nvidia-smi. This sprint does not install or validate an inference dependency stack. The repository-wide `requirements.txt` includes unrelated application packages and is not a reproducible benchmark environment lock.

On a dedicated VM with permission to select a GPU, bind only the assigned GPU before Python imports Torch:

```bash
export CUDA_VISIBLE_DEVICES=GPU-REPLACE-WITH-YOUR-ALLOCATED-UUID
python -c 'import torch; print("Torch:", torch.__version__); print("Runtime:", torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print((torch.ones(2, device="cuda") + 1).cpu())'
```

If a scheduler/container already supplies `CUDA_VISIBLE_DEVICES`, retain its mask rather than replacing it. GPU numbering inside Python is relative to that mask. The command above tests a tiny allocation; it does not load a model. PyTorch documents CUDA availability checking [here](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available).

The existing read-only checker needs no ML packages unless you opt into its runtime probe:

```bash
python -m benchmark.resource_check
python -m benchmark.resource_check --gpu-uuid GPU-REPLACE-WITH-YOUR-ALLOCATED-UUID
```

`--probe` performs tiny runtime operations. Use it only after establishing the allocation/mask; the present probe can overwrite a mask when supplied a UUID. Recommendations are preliminary, and cache completeness is unknown without a verified pinned file list.

## 6. What is ready versus pending

Ready: browser annotation, CSV save/resume/edit, reviewed manifest export, focused tests, and preliminary resource inspection.

Pending: shared runner, clean cascade adapter, pinned VLM adapters, explicit capped downloads, measured GPU runs, and comparison reporting. The existing `benchmark/configs/cpu-small.json` and `gpu-large.json` are presets for that work, not runnable model-launch commands. Do not launch `test_qwen.py` as a setup test; it loads a model on import.

The historical Pixtral checkpoint has about 25.4 GB of original weights; planned 4-bit execution does not remove that original download requirement. GPU fit must be measured after checking the runtime and allocation. No weights are downloaded by this sprint.

## 7. Verify this sprint

From the repository root:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m tools.annotate --help
```

The focused test directory avoids the legacy root-level inference script. Tests cover label persistence, leading zeros, edits/backups, manifest export, incomplete data, changed images, duplicate/missing images, and invalid labels.
