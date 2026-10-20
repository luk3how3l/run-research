# Coding-agent handoff: reproducible bib-number comparison

Date: 19 October 2026

## 1. Task and current boundary

Build a portable benchmark comparing SmolVLM-500M, Pixtral-12B, and eventually the repository's YOLO-person → YOLO-bib → Tesseract cascade on the same labeled race photos. Organize delivery around the four milestones in [README.md](README.md).

This commit is documentation only. The user requested this handoff while Destopper was offline so another coding agent can implement and help test the code. No new resource checker, benchmark implementation, dependency installation, model download, GPU run or 100-image experiment is being claimed here.

Inspected source baseline: `05ee66e85096ede530592b4ead437a5042c1111d`. Before editing, inspect the current branch, local changes and any later commits. Preserve unrelated work. These instructions do not authorize blindly overwriting concurrent changes or publishing future code changes without confirming that scope with the user.

## 2. Existing draft you can reuse

A small earlier deliverable was supplied to the user as `configurable-vlm-test-20261016.zip`:

- Size: 27,897 bytes
- SHA256: `0dc7f1c57b2aca10e4d380b640414e179a960ca6ce4fee2146850aa975bdde19`
- Archive root: `vlm-benchmark/`
- Contains `vlm_benchmark.py`, `vlm_adapters.py`, JSON presets, `bib_manifest.json`, `run_vlm.sh`, common/optional requirements, unit tests and documentation.
- Its 41 standard-library tests, syntax checks and three launcher check modes passed. Its adapter APIs were reviewed against Transformers 4.57.6.
- It was not installed, downloaded, loaded or inference-tested as a new CPU/GPU implementation. It was not committed to this repository.

If the archive is not available in your environment, ask the user to attach it. Do not assume a path from another machine exists. Verify its hash before using it. You can implement the design below without it, but do not describe the absent draft as already present in Git.

Preserve its useful behavior: config/CLI precedence, immutable revisions, explicit capped download mode, offline inference, image hashes, strict/normalized scoring, new output directories, explicit adapters and single-GPU UUID isolation. Review and extend rather than blindly importing it.

Known improvements needed: separate resource selection, manifest validation for 100 photos, comparison reports, explicit missing/invalid predictions, phase-specific memory measurements, honest cascade readiness, and robust failure records.

## 3. Confirmed historical result, not a new run

On 14 October 2026 a bounded three-photo experiment completed on Destopper, Ubuntu 24.04 WSL, Intel Core i5-1235U, 12 visible logical CPUs, CPU float32, two Torch threads. Roughly 5.2 GiB RAM was available during setup. Torch reported CUDA unavailable. Do not substitute a different laptop identity or call this a university-GPU result.

- Model: `HuggingFaceTB/SmolVLM-500M-Instruct`
- Revision: `a7da5b986cb59b408707209984f360a5f4ad7e47`

The historical harness called the existing repository's model/image/generation functions, with explicit local weights and offline mode. It loaded once, used no warmup, and ran one pass in Ryan/Hannah/Quinton order. It is not identical to a future rewritten harness; reproduce before comparing implementation effects.

| Image | Ground truth | Raw response | Normalized response | Generation s | Image pipeline s |
|---|---|---|---|---|---|
| media/ryan_rockie_bib.jpg | 3187 | `318.` | 318 | 42.8642 | 44.1620 |
| media/hannah_bib_test.jpg | 2629 | `209.` | 209 | 39.6320 | 40.2108 |
| media/quinton_bib_image.jpg | 2300 | `2300.` | 2300 | 40.9939 | 41.4240 |

All three executed successfully. Normalized exact accuracy was 1/3; strict raw exact accuracy and digits-only compliance were 0/3. Model load took 5.3446 s. External sampled peak RSS was 3,838,005,248 bytes; process high-water RSS was 3,804,372 KiB. Preserve the distinction between methods. True Levenshtein CER was not computed in the old run; repository SequenceMatcher similarity must not be relabeled as CER.

The earlier model download totaled 1,019,883,656 bytes for selected files and was covered by a specific under-2-GB exception. That does not authorize a fresh large download elsewhere.

Historical package versions included Torch 2.10.0, torchvision 0.25.0, Transformers 4.57.6, huggingface-hub 0.36.2, safetensors 0.7.0 and Pillow 12.1.1. These are provenance, not a universal installation prescription. Original raw result files remain separate historical evidence; do not fabricate a new run directory from this table.

## 4. Shared data and scoring contract

### Freeze the three-photo smoke set

Use these exact image hashes and labels:

| Image | Label | SHA256 |
|---|---|---|
| media/ryan_rockie_bib.jpg | 3187 | `59e3e492a3d018e4721e6fd845141223a287f4469207b27ca92a744b851de137` |
| media/hannah_bib_test.jpg | 2629 | `16d391e3a244236333ea44ca1f91991f56e421c0513bf1df10a629480e9ca066` |
| media/quinton_bib_image.jpg | 2300 | `97f3d0544a5846ef8ca6ff47d5a0819961de67fd9d9257cf75afdd88509a6009` |

Prompt, verbatim:

> Read the race bib number of the central foreground runner. Return only its digits, or UNKNOWN if unreadable.

Keep labels as strings, preserving leading zeros. The central target matters: Hannah's photo contains another visible bib near the left edge. Do not choose a target after reading a model's answer.

### Build the 100-image scaffold, not fictional data

Create a versioned manifest with the three known development rows and 97 empty, explicitly unlabelled slots, or an equivalent collection checklist. Empty slots must not pass runnable-manifest validation. No fabricated filenames, labels, target boxes, provenance or results.

For actual records require image path, hash, bib string, split, label-review status and target identity. Add an absolute-pixel `target_bbox: [x1, y1, x2, y2]` if using annotated matching for the cascade. Validate box bounds against the actual image. Do not invent boxes for the three current photos.

Validate exact expected count, file existence, path containment, hash match, unique paths and unique content hashes, digit-string labels and completed review. Flag near-duplicates/same-event leakage for human review. Record permission/provenance for new photos; obtain the real photos and human-reviewed labels from the user or an authorized source.

Freeze inclusion/exclusion rules and parameters before evaluating. The three existing images are development data. If the collection contains 100 total, report the three development photos separately from the 97 new evaluation photos. If 100 genuinely held-out examples are required, collect 100 additional reviewed images instead. Do not call already-tuned examples held out.

### Measure the same task

All systems receive the original full-frame image. VLMs use their model-specific processors; record effective resolution/token settings. Do not silently crop a difficult image for one model. A crop experiment is a separate named protocol applied consistently.

Choose and freeze one target-selection policy:

- **Annotated matching:** run cascade detection/OCR on the full frame, then match its emitted person detections to the reviewed target box using a declared IoU/ambiguity rule. Never feed the ground-truth box as an oracle crop to the detector. No matching detection counts as a miss.
- **Or** a documented central-foreground heuristic, fixed before test predictions and reviewed to identify the same person the VLM prompt names. Record ambiguous-target exclusions before scoring.

Do not select whichever cascade bib happens to equal the ground truth. Label-assisted best-answer selection is invalid. Annotated matching is for evaluating target identity, not recovering the correct digits.

### Metrics

Save raw responses before normalization. Strict exact match strips surrounding whitespace only. Preserve the existing first standalone digit-group rule, falling back to the first digit group anywhere, for normalized exact comparison. Flag multiple digit groups. Record digits-only compliance, abstention, empty/missing output and invalid-format output separately; document overlapping categories.

Report correct counts over all planned trials. Failed, missing and unattempted samples cannot disappear from the denominator. Keep failed execution separate from a successful but incorrect answer. Use true Levenshtein CER if implemented, with an explicit denominator and treatment of execution failures. Do not rename SequenceMatcher similarity.

## 5. Milestone implementation plan

Suggested file layout is illustrative, not a claim that these files exist:

- `benchmark/runner.py`: shared explicit check/download/run entry point
- `benchmark/adapters/`: SmolVLM, Pixtral, optional other supported families, and cascade
- `benchmark/resource_check.py`: separate read-only resource/runtime recommendation
- `benchmark/manifest.py`: smoke/frozen-set validation and scaffold progress
- `benchmark/compare.py`: consistent two-/three-system reports
- `benchmark/configs/`: CPU-small and single-GPU-large JSON presets
- `benchmark/manifests/`: three-photo manifest and non-runnable 100-slot scaffold
- `scripts/run_benchmark.sh`: explicit profile/mode selection; no automatic download
- Focused tests that do not import or download models at collection time

Reuse the earlier `vlm-benchmark/` layout if simpler; avoid duplicate competing entry points. Document the chosen runnable commands only after implementing and testing them.

### Milestone 1: Reproducible local CPU baseline

1. Fix or bypass the `start.py` import problem (`vlm_vision` is under `src/`) with a tested clean entry point; preserve unrelated existing behavior.
2. Make module import and `--help` free of model loading/network calls.
3. Use SmolVLM on CPU with float32, no CUDA quantization and explicit thread settings.
4. Load once, preserve the shared protocol, save raw predictions and clear failure status.
5. Confirm environment/storage authority before installs or downloads, then reproduce the three-photo baseline on an available approved machine.

**Acceptance:** unit/CLI tests pass; invalid inputs fail before expensive work; an actual authorized CPU run has exit status, configuration, hashes, raw answers and measurements saved. Do not mark inference complete on the strength of mocked tests. If the intended laptop differs from Destopper, mark laptop reproduction pending until actually run there.

### Milestone 2: Renn GPU SmolVLM versus Pixtral

Use the Transformers-compatible checkpoint:

- Model ID: `mistral-experimental/pixtral-12b`
- Revision: `c2756cbbb9422eba9f6c5c439a214b0392dfc998`
- Original BF16 weight payload: 25,365,479,424 bytes
- Loader: `LlavaForConditionalGeneration` with Pixtral vision configuration
- Proposed runtime: bitsandbytes NF4, nested quantization if explicitly configured, BF16 compute on supported hardware; explicit FP16 alternative only after validation

The older `mistral-community/pixtral-12b` address in Transformers documentation redirects to this converted checkpoint. Do not substitute the original Mistral distribution without checking its format/API.

SmolVLM at the pinned revision uses `Idefics3ForConditionalGeneration`, not the separate SmolVLM architecture class. Explicit adapters must validate checkpoint architecture. Keep `trust_remote_code=False` and safetensors-only loading unless a separate reviewed requirement changes that policy. Do not silently fall back to an unrelated causal-LM class.

Renn snapshot supplied by the user: 2× NVIDIA L4, 23,034 MiB each; GPU 0 had 3 MiB used / 0% utilization; GPU 1 had 12,991 MiB used / 99% utilization. About 125 GiB RAM and 680 GiB filesystem space were available; personal quota was not established. Python 3.12.3 and driver 580.178.04 were reported. These are dated observations, not current allocations.

Recheck selected GPU and node policy. Use only one explicitly authorized physical GPU, bind it by UUID before Torch import and respect inherited scheduler/container masks. Avoid unrestricted `device_map="auto"`; no automatic second-GPU use or CPU/disk spill. A mismatch or busy GPU is a stop condition, not permission to kill another job.

Suggested conservative Pixtral preflight: >=18 GiB free GPU memory, <=10% GPU utilization, <=1 GiB already used and >=12 GiB available host RAM, plus storage headroom. These are policy thresholds, not measured model requirements or fit guarantees. Log them and any user-reviewed override. Check BF16 support and CUDA bitsandbytes compatibility before loading. Do not interpret the driver's "CUDA Version" label as the installed Torch runtime.

Run SmolVLM and Pixtral on the same GPU, sequentially when practical, in addition to the historical CPU comparison. Otherwise label results as a deployment comparison involving both model and hardware changes.

**Acceptance:** authorized capped download/cache verification; actual offline GPU runs on the same complete smoke manifest; pinned revisions, raw/normalized scores, load/warmup/runtime/memory recorded; no silent fallback or unclaimed use of another GPU. Three samples establish smoke-test behavior only.

### Milestone 3: Separate portable resource selection

Provide a read-only checker that works without ML dependencies where possible. It must not import the legacy cascade or load any model.

Inspect:

- Available host RAM and detectable cgroup/container/job limits; state unknown limits
- Explicitly selected GPU's identity, VRAM free/used/total, utilization and driver
- Disk free space on the cache filesystem, existing pinned cache files, missing/truncated files, and quota as unknown unless actually checked
- Python and installed package versions; Torch/torchvision compatibility, CUDA availability, BF16 and bitsandbytes where applicable

Metadata-only package presence is not runtime proof. Make an optional runtime probe explicit; it may import installed libraries and exercise tiny CPU/CUDA operations on the permitted device, but must never load/download a model. Capture probe errors, timeouts and missing native libraries as refusal reasons.

Recommend a small CPU candidate, a large single-GPU candidate, or refuse, with actionable reasons and separate hardware/runtime/cache readiness. If weights are missing, report an explicit capped download step still required; do not say an offline run is ready. Never infer runtime RAM/VRAM from file size alone. The historical CPU run fit about 5.2 GiB available RAM with ~3.9 GB process peak; a 6 GiB conservative small-profile threshold can refuse that machine, so expose and explain policy thresholds rather than treating them as physics.

Configuration and CLI should cover model ID, immutable revision, adapter, CPU/CUDA device, explicit GPU identity, dtype, quantization, cache, manifest, prompt, generation controls, repetitions/warmup, image-processing limits and output location. Use CLI > config > defaults, reject unknown keys and invalid combinations, and allow clearing nullable settings. Keep approval flags on CLI rather than buried in a config.

Requirements must distinguish common dependencies, platform-appropriate matching Torch/torchvision builds, optional CUDA NF4 dependencies and optional cascade dependencies/native Tesseract. Do not force bitsandbytes on CPU-only users. The previous draft used Transformers 4.57.6; pin a reviewed compatible version, not an untested latest API.

**Acceptance:** fixture tests cover CPU-only, suitable selected GPU, busy GPU, insufficient RAM/VRAM/disk, missing cache, dependency mismatch, inherited-mask mismatch and probe failure. Hardware detection performs no install, download or inference. Actual runtime success remains separately recorded.

### Milestone 4: Frozen 100-photo three-way evaluation

Implement strict manifest validation and a clearly incomplete scaffold. Collect and review real data, freeze splits/target rules, then execute each ready system over the same complete ordered set. Comparison must reject mismatched images, hashes, labels, target definition, prompts/decoding settings for VLMs, warmup or repetitions rather than quietly compare an intersection.

A third system's status can be "blocked"; do not turn that into fake prediction records or completed three-way accuracy. When a real cascade run exists, compare target-level outputs and full-pipeline timing under its documented settings. Model-specific preprocessing and cascade stage parameters may differ, but must be disclosed. VLM decoding settings are not applicable to Tesseract; share the task/data/warmup/repeat contract and retain system-specific controls separately.

**Acceptance:** 100 unique reviewed records pass validation; development/evaluation splits are explicit; every system has raw records or a genuine failure for every planned trial; comparison reports accuracy counts, failure/missing/invalid counts, per-image timing and memory with provenance. No fabricated labels or performance claims.

## 6. Cascade work: preserve provenance, expose blockers

Inspect `vision/background_yolo.py` without importing it first. Current blockers:

- Imports `core.database.AthleteDatabase` and `core.image_cache`, which are absent from this repository
- Creates database/cache and YOLO objects at module import
- Hardcodes Windows paths for `yolov8s.pt`, custom `number632resol.pt`, and Tesseract
- Custom bib-detector weights are not in the repository
- Writes fixed-name OCR debug images and has no benchmark-safe artifact policy
- Returns results for all detected people, without the required single target policy

The legacy `process_entire_frame` uses person confidence 0.5, bib confidence 0.28, no padding, and OCR scale 1.5 / area 50 / border 0 / PSM 11 with Otsu preprocessing. `process_clicked_athlete` has different OCR settings; do not accidentally call the clicked-athlete path and label it the same baseline.

Prefer an optional clean adapter isolating the detector/crop/OCR logic from database/cache/UI dependencies. Require explicit local person/bib weight paths, verified hashes/provenance, an installed Tesseract executable and explicit device selection. Do not let Ultralytics fetch a named weight automatically. Preserve raw Tesseract text and separately log the legacy joined-digit output; if extraction behavior changes, name it a new variant, not exact legacy parity. Handle no detections, no bibs, ambiguous target matches, empty OCR and execution errors distinctly.

Until weights, native dependencies and target policy are available, provide a structured blocked readiness report. A blocked legacy adapter is better than a guessed replacement or fabricated success. Do not launch a real inference attempt just to discover that these inputs are missing.

## 7. Runtime, memory and result contract

Use separate check, explicit download, and offline run modes. Never download on import, `--help`, resource detection or a cache miss during inference. Pin one exact revision for both processor and weights. Include required tokenizer/config files and every referenced safetensors shard; avoid duplicate formats/ONNX alternatives. Enforce a finite positive decimal-GB cap against all selected files before transfer, account for existing cache and temporary space, and verify downloaded sizes/hashes as available.

Preserve conservative storage authority: stop and ask before this work would exceed 1 GB of new local storage unless the user explicitly approves that operation and environment. The prior under-2-GB SmolVLM exception and intended university large-model run do not authorize new cloud weight downloads. A 30-GB download example is not permission to execute it. Never store weights, caches, credentials or private dataset images in Git by accident.

Save metadata, raw JSONL predictions, a compact CSV and summary. Use unique output directories. Record model/processor IDs and revisions, input/config/code hashes, software versions, hardware, target policy and original model weight size separately from measured memory.

For each trial save image/hash/label, raw text, normalized prediction, status, errors/stage, token counts if applicable, preprocessing time and full image-pipeline time. For VLMs save synchronized generate time; for the cascade save detection/crop/OCR stage times. Compare end-to-end image-pipeline time across systems, not VLM generation-only versus full cascade time.

Measure model load separately from warmup and measured trials. Warmup count defaults to zero to reproduce the old run; allow separately named warmed experiments. Repeated trials are timing observations, not more independent images.

Reset CUDA allocator peaks per phase and record allocated/reserved peaks. For CPU phase peaks, use a documented RSS sampler or equivalent; keep process-lifetime high-water RSS as a different field. Sampling can miss short peaks, and Torch allocator memory is not whole-board memory. Do not label missing/unsupported measurements as zero. Capture partial failures where possible; hard kills/disk-full failures may prevent final records.

Do not silently lower resolution, change quantization, retry a failed sample with a different model, or drop failed/unattempted trials. Record requested versus effective generation/processing settings. Greedy decoding is not a promise of bitwise reproducibility across hardware.

## 8. Test and delivery checklist

Before any expensive runtime validation:

- Standard-library tests for config precedence, nullable overrides, invalid revisions and unsupported adapters
- Score cases for punctuation, leading zeros, abstentions, no digits, multiple numbers and wrong digits
- Manifest tests for incomplete scaffold, wrong count, duplicate hashes, missing/mislabeled files, path escape and target-box validation
- CPU/CUDA/quantization and resource refusal fixtures; download budget and offline-cache guards
- Bash syntax and argument forwarding; import/`--help` tests proving no downloads or model construction
- Comparison tests rejecting protocol/data mismatches and preserving failed/unattempted denominators
- Cascade mocked tests for missing weights/Tesseract, detection failure and target ambiguity without real weights

Avoid broad test discovery until `test_qwen.py` is made safe: it currently loads a model at import and has mismatched model/class assumptions. No tests should install dependencies or fetch weights unexpectedly.

Then request/confirm the environment and resource/download authority needed for real runs. Run focused smoke tests first, then repeated timing and the frozen evaluation set. Report exactly what passed, failed and was never run. Update README milestone status only from saved evidence.

## 9. Inputs still needed

1. An available approved execution environment and, for Renn, confirmed node/GPU allocation and personal storage quota.
2. Any required permission for package installation and model downloads in that specific environment.
3. The optional earlier draft ZIP if reusing it, and historical raw run artifacts if deeper reproduction is needed.
4. The real custom bib-detector weights, person weights, their provenance/hashes, and Tesseract setup.
5. Real additional photos with usage permission, reviewed digit labels, target annotations/policy and frozen dataset splits.
6. Agreement on whether "100 images" means 100 total including the three development photos or 100 new held-out photos.

Proceed with small code/config/tests that do not require missing inputs; keep dependent runtime stages explicitly blocked.

---

## Progress since this handoff (added 19 October 2026, uncommitted work)

Everything below is unit-tested only. Nothing here has been installed, downloaded, or run against a model.

- **Done:** `start.py` import fix (deferred `vlm_vision` import, `src/` on path); `benchmark/scoring.py`, `manifest.py`, `config.py`, `resource_check.py`; `benchmark/configs/` presets; `benchmark/manifests/smoke3.json` and `scaffold100.json`; `tests/test_benchmark_core.py` (run with `python3 -m unittest tests.test_benchmark_core`).
- **Not done:** `runner.py`, adapters (SmolVLM, Pixtral, cascade), download cap enforcement, memory sampling, `compare.py`, `scripts/run_benchmark.sh`, cascade blocked-readiness report, and every actual run.
- **Still open:** whether "100 images" means 100 total or 100 new held-out; all items in section 9.
