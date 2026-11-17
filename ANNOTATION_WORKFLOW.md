# Annotate photos and prepare a benchmark batch

Use this guide for the current status-based workflow. Commands run from `~/run-research`. Annotation uses Python 3.10+ and a browser; it needs no GPU, model weights, or pip packages. Run the benchmark only after completing the appropriate [RENN setup](SETUP_RENN.md) or [laptop setup](README.md#run-a-vlm-on-your-pc).

## 1. Choose where to annotate

For the simplest RENN workflow, keep the images and authoritative CSV on RENN and open its annotation page through an SSH tunnel. Follow steps 2–8 on RENN. If you annotate on the laptop instead, run those steps locally and follow step 9 before benchmarking on RENN. Use one active annotation server and one authoritative CSV for this collection.

## 2. Prepare the directories

```bash
cd ~/run-research
mkdir -p media/unannotated media/annotated media/needs_review annotations benchmark/manifests
```

The layout is:

```text
media/unannotated/         new photos waiting for annotation
media/annotated/           reviewed photos ready for benchmark export
media/needs_review/        skipped, unreadable, or ambiguous photos
media/drive-test-trainbib/  imported source dataset; keep the originals here
annotations/bibs.csv       authoritative annotation record
benchmark/manifests/       exported benchmark snapshots
```

Existing photos directly under `media/` and imported directories are not automatically queued or reorganized. The default scan covers the three managed directories and all paths already recorded in the CSV. Existing CSV records, including older paths and pending records outside the queue directory, are retained. Older reviewed records move to the managed directory when you save them again.

## 3. Add a small batch of new images

Copy selected source photos into `media/unannotated/`. Keep imported originals where they are. For example, if this existing photo is available:

```bash
cp -n media/IMG_1852.png media/unannotated/
```

For a downloaded image, substitute its real filename:

```bash
cp -n 'media/drive-test-trainbib/YOUR-SELECTED-PHOTO.jpg' media/unannotated/
```

`-n` prevents overwriting a queue file. Choose a unique destination filename if different photos share a name. Supported extensions are `.jpg`, `.jpeg`, `.png`, and `.webp`, including uppercase variants. BMP and TIFF need conversion before adding them; do not merely rename their extension.

Add each image once. Do not copy it into the queue again after it moves to `annotated/` or `needs_review/`. The tool rejects duplicate content across the managed collection and recorded CSV paths, even under different filenames. If the image is already in the CSV, use **All images / edit labels** instead of copying it again. In particular, keep the original three smoke-test photos in place: existing smoke manifests reference their paths.

If your images are only on the laptop and you want to annotate on RENN, run this from the **laptop repository**, using the real SSH host and filename:

```bash
ssh luke@student-luke 'mkdir -p ~/run-research/media/unannotated'
rsync -av --ignore-existing media/IMG_1852.png luke@student-luke:~/run-research/media/unannotated/
```

## 4. Start the annotation server

On the machine holding the dataset:

```bash
cd ~/run-research
python3 -m tools.annotate --csv annotations/bibs.csv
```

Leave that terminal running. For **RENN**, open a second terminal on your **laptop**:

```bash
ssh -N -L 8765:127.0.0.1:8765 luke@student-luke
```

Replace the SSH destination if different. Leave the tunnel running, then open **http://127.0.0.1:8765** in your laptop browser. For local laptop annotation, open that address directly without a tunnel.

If port 8765 is already occupied, start the server with `--port 8767`, forward `8767:127.0.0.1:8767`, and open port 8767 in the browser.

## 5. Label one designated runner

For each photo:

1. Check the filename and identify the intended runner. Describe that runner in **Target runner**. The benchmark's fixed central-foreground policy must identify the same person; a description does not change model selection.
2. Choose **reviewed** only when you can read the bib confidently. Enter digits only, such as `007`; leading zeros are preserved.
3. Choose **development** for examples used to tune or debug the systems. Choose **evaluation** for genuinely held-out examples, and enter a source/permission note. Record provenance for development photos too when known.
4. For an unreadable, uncertain, or intentionally skipped photo, choose **unreadable**, **ambiguous**, or **skipped** and clear the bib input. Preserve the target description, split, and source note.
5. Click **Save and next**. Confirm the success message and updated counts.

Saving verifies the image hash and annotation fields before moving the file. Reviewed images move to `media/annotated/`; the other three saved statuses move to `media/needs_review/`. The CSV path is updated and saved atomically, with a uniquely named backup of the previous CSV. The next pending image appears only after a successful save. If saving fails, the page stays on the current record and shows an error.

**Previous** and **Next without saving** only navigate; they discard unsaved input. When the queue is empty, the page says **All available images have been annotated.** This means no pending rows remain; it does not mean every image was readable or benchmark eligible.

## 6. Resume, correct labels, or recover from an error

Stop the server with Ctrl+C and restart it with the same command and CSV to resume. Keep the CSV and images together. Use **All images / edit labels** to revisit reviewed or needs-review rows. Correct a bib or change a needs-review status to reviewed, then save; the file moves to the appropriate managed directory.

The CSV columns remain `image_path,bib,sha256,status,target_identity,split,provenance`. Image SHA-256 is the stable identity across moves. Import bibs as **text** when opening the CSV in spreadsheet software. Prefer editing through the tool.

| Error or situation | What to do |
|---|---|
| Destination filename collision | Keep both files. Give the pending source a unique filename while the server is stopped, restart, and retry. The tool reconnects existing records by content hash within its scan scope. |
| Duplicate image content | Check the paths. Remove only an accidental, unrecorded queue copy after confirming the recorded original is intact; keep the CSV record. |
| Image changed/hash mismatch | Restore the exact original image from your backup. Do not replace its recorded hash to bypass the check. A genuinely different image needs its own annotation. |
| Missing CSV image | Restore that image at its recorded path, or into a managed directory where its hash can be found. Records are not silently dropped. |
| Interrupted save | Restart the tool with the same CSV. Its adjacent `.transaction` journal reconciles the move with the persisted CSV. |
| Recovery fails, disk full, or permissions error | Preserve the CSV, backups, journal, and both image locations. Fix the reported storage/permission issue, then restart. Do not delete the journal to suppress the error. |

Keep all three managed directories on the same filesystem: safe moves use hard-link creation followed by unlinking the source. Destination files are never overwritten. Symbolic links in collection paths are rejected. Keep backups of the CSV **and images**; CSV backups alone cannot recover a missing photo.

## 7. Count reviewed images before exporting

Stop annotation before freezing a batch. Run:

```bash
python3 - <<'PY'
import csv
from collections import Counter
from pathlib import Path
path = Path('annotations/bibs.csv')
if not path.exists():
    raise SystemExit('No saved annotations yet. Save an image in the browser first.')
with path.open(newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
print('Statuses:', dict(Counter(r['status'] for r in rows)))
print('Reviewed:', sum(r['status'] == 'reviewed' for r in rows))
print('Reviewed splits:', dict(Counter(r['split'] for r in rows if r['status'] == 'reviewed')))
PY
```

The CSV is the source of truth. Counts of files in imported directories are not counts of reviewed annotations.

## 8. Export a validated manifest

For example, if step 7 reports **18 reviewed** images:

```bash
python3 -m tools.annotate --csv annotations/bibs.csv \
  --export benchmark/manifests/renn-batch-01.json --expected-count 18
```

Replace `18` with the actual reviewed count. Export includes **every reviewed row** in this CSV, across both splits. It excludes pending, skipped, unreadable, and ambiguous rows while retaining them in the CSV. This is not a per-folder or evaluation-only filter. No reviewed rows, a wrong count, invalid labels, duplicate content, missing files, or hash changes cause export to fail.

Review and report exclusions when defining the evaluation dataset; do not claim excluded difficult photos as successful trials. Choose and freeze inclusion rules before looking at benchmark predictions. This tool does not establish target agreement or detect near-duplicate/event leakage automatically.

Use a new manifest filename for a new frozen batch. Finish label corrections and file moves before export: the benchmark validates the exact paths, hashes, and labels in that snapshot. Editing the CSV does not update an existing manifest. Keep snapshots and their exact images available to reproduce runs. Export itself does not run any model.

## 9. If annotated locally, transfer the dataset to RENN

Skip this step if you annotated on RENN. Stop the local server and use a RENN destination that does not already contain a different active collection. From the laptop repository:

```bash
ssh luke@student-luke 'mkdir -p ~/run-research/media ~/run-research/annotations ~/run-research/benchmark/manifests'
rsync -av --ignore-existing media/unannotated media/annotated media/needs_review \
  luke@student-luke:~/run-research/media/
rsync -av --ignore-existing annotations/ luke@student-luke:~/run-research/annotations/
rsync -av --ignore-existing benchmark/manifests/renn-batch-01.json \
  luke@student-luke:~/run-research/benchmark/manifests/
```

These commands retain directory names and do not delete remote data. `--ignore-existing` leaves existing remote files untouched; it does **not** merge divergent CSVs or refresh an older copy. If the destination already has `annotations/bibs.csv`, compare and back it up, then deliberately choose the authoritative version before proceeding. Also transfer any images still referenced outside the managed directories at the same repository-relative paths. Validation in step 10 detects missing or mismatched files.

## 10. Run the exported batch on RENN

After the GPU/dependency/model setup in [SETUP_RENN.md](SETUP_RENN.md), on RENN:

```bash
cd ~/run-research
source .venv-renn-gpu/bin/activate
python -m benchmark.runner \
  --device cuda --gpu-uuid GPU-YOUR-ACTUAL-ALLOCATED-UUID \
  --manifest benchmark/manifests/renn-batch-01.json \
  --systems cascade smolvlm pixtral \
  --smolvlm-dir models/smolvlm-500m --pixtral-dir models/pixtral-12b \
  --check-only
```

Use your actual allocated UUID and retain the inherited allocation mask. After `preflight_passed`, repeat the command **without `--check-only`** to perform inference. Add `--gpu-wait-seconds 30 --gpu-poll-seconds 5` for bounded busy-GPU retries. Each invocation creates a new results directory. Check-only does not establish accuracy or successful model inference.

Always pass `--manifest` for this batch. The browser comparison button and the runner without `--manifest` use the original three development photos. Merely populating `media/annotated/` does not change the benchmark input.

## 11. Read and retain the results

Open the exact `results/comparison-.../` directory printed by the command:

1. Read `REPORT.md` for status and errors.
2. Open `summary.csv` for metrics per system and dataset split.
3. Open `comparison.csv` for each planned image/system prediction, failure, timing, and score. Preserve bib columns as text.
4. Keep `manifest.json`, `metadata.json`, `gpu-preflight.json`, and per-system reports and logs with the CSVs. Inspect target selection before interpreting accuracy.

See [RENN's result-file reference](SETUP_RENN.md#running-a-multi-image-three-model-comparison) for metric definitions and GPU troubleshooting. Keep personal photos, labels, credentials, and weights out of commits.
