# Research goals and coding sprints

Updated: 2 November 2026. Target: finish a paper and demonstration by 30 November 2026. The schedule below is proposed; remaining experiment decisions will refine scope.

## Confirmed deliverables and execution environments

- **Paper:** explain the task, dataset, reproducible method, baseline comparison, selected hybrid experiments, results, error analysis, and limitations.
- **Demonstration:** run selected systems on the same example photos and show their bib predictions and measured runtime. The demonstration format remains to be decided.
- **Laptop:** develop and test small components, validation, scoring, result handling, and bounded smoke experiments that fit its resources.
- **RENN network:** execute larger model runs and the main evaluation. Confirm the actual execution node, GPU allocation, dependencies, and storage before running; network access alone does not establish those resources.
- **Configuration portability:** select the model adapter, pinned model revision, weight paths, device, dtype, quantization, and cache through configuration and explicit overrides. Switching from laptop SmolVLM to RENN Pixtral must not require editing source code. Switching profiles must not automatically download weights or run inference.
- Use the same code, versioned data, configuration conventions, and result schema across environments. Record hardware differences and use the same RENN hardware for direct runtime comparisons where practical.

## Proposed November delivery schedule

| Dates | Coding and experiment focus | Reviewable deliverable |
|---|---|---|
| Nov 1–7 | Sprints 0–2: annotation tool, shared framework and legacy cascade recovery; start photo collection and label review | Tested runner and cascade adapter; actual smoke run when weights are available |
| Nov 8–14 | Sprint 3 and RENN preparation: small VLM baseline, comparison reports, setup verification | Saved cascade/VLM smoke comparison; documented laptop and RENN commands |
| Nov 15–21 | Sprint 4 and selected Sprint 6 work: larger VLM and initially one hybrid; finish Sprint 5 dataset preparation | Development results, frozen evaluation protocol and reviewed dataset |
| Nov 22–25 | Sprint 5 evaluation: execute frozen systems on RENN and analyze errors | Final result artifacts, tables, plots, and limitations |
| Nov 26–30 | Finish paper, demonstration, and reproducibility checks | Paper and working demonstration by Nov 30 |

Draft the paper's methods and maintain experiment notes throughout the month. Reserve the final days for writing and demonstration fixes. If resources or data delay progress, reduce additional model/hybrid variants first and document the resulting scope. Do not replace missing evaluation evidence with smoke-test claims.

## Research goal

Compare the existing YOLO person detector → YOLO bib detector → OCR cascade with vision-language models on the same race photos and target runners. Establish a reproducible comparison first, then use the same experiment framework to evaluate alternative cascades and hybrid systems.

Primary metric: exact bib-number accuracy for one designated runner per image. Compare complete systems: the legacy pretrained/custom detector and OCR cascade versus SmolVLM and Pixtral. This evaluates these specific systems, rather than establishing that VLMs outperform pretrained machine learning in general.

Speed is a second key outcome because the intended application processes images live in the background. Record end-to-end per-image latency, latency distribution, throughput under a declared execution method, and memory. A live deployment latency target remains to be defined; still-image benchmarks alone do not establish live-video performance.

**Phase 1:** prepare and execute pure full-image VLM versus legacy cascade comparisons.

**Phase 2:** add YOLO person → VLM and YOLO person → YOLO bib → VLM hybrids after the baseline comparison works.

## Starting point from the October review

These observations describe the prior review, not verification of subsequent code changes.

- Three development photos and a documented historical SmolVLM CPU pilot: one of three normalized bib predictions correct. Original raw run artifacts are not tracked here.
- Configuration, scoring, manifest validation, presets, resource checks, and 23 passing focused tests exist as of the repository review on 22 October.
- The shared runner, model adapters, controlled download implementation, phase memory sampling, and comparison reports remain to be built.
- The legacy cascade depends on absent database/cache modules, machine-specific paths, and missing detector weights. It processes multiple people rather than enforcing the VLM's central-runner task.
- The 100-photo manifest contains three populated development records and 97 empty slots. The current labels are user-supplied rather than marked reviewed.

## Current local model inventory (2 November)

The local `models/` directory contains `yolov8s.pt`, `number632resol.pt`, `number832resol.pt`, and `best.pt`. The current cascade points to `yolov8s.pt` and `number632resol.pt`; use those as the proposed legacy baseline, recording hashes and verifying their identities before experiments. The other files are not assumed to be interchangeable or better models.

Recent local changes already make database/cache imports optional and point the cascade at local models and available Tesseract. It still constructs YOLO models on import. Preserve these changes and inspect `test_background_yolo.py` before implementing the benchmark adapter. Local file presence is not evidence of a successful cascade run.

## Sprint 0 — Photo annotation tool and dataset preparation

**Implementation status (2 November):** `python3 -m tools.annotate` now provides browser annotation, atomic CSV saves with backups, resume/edit, and strict reviewed-manifest export. Six annotation tests and the existing 23 tests pass; a localhost HTTP smoke test verified page/image/rows/save endpoints. Browser visual review, human annotation, and RENN execution remain pending. See [setup instructions](SETUP_LAPTOP_AND_RENN.md).

**Outcome:** a separate laptop program displays one image at a time and lets the researcher enter and review its bib number.

- Show the image, filename, progress, and designated target; support save/next, back/edit, skip, and resume.
- Export CSV with at least `image_path,bib`; preserve bibs as strings, including leading zeros. Add image hash, target identity, split, and review status for reproducibility.
- Keep unreadable/ambiguous/skipped entries explicitly marked and separate from completed digit labels. Decide their scoring policy before evaluation.
- For photos with multiple people, provide target annotation or require explicit review of the fixed central-runner policy. Do not choose the target from predictions.
- Import validated CSV labels into the shared manifest; detect missing files, duplicate content, and changed image hashes.
- Save progress safely and do not overwrite existing annotations without preserving prior work. Do not display model predictions during ground-truth annotation.

**Done when:** the three existing photos can be labeled, saved, reopened, corrected, and imported into a valid manifest without losing leading zeros. This tool helps build the fixed 100-photo collection while benchmark coding proceeds.

## Experiment principles

1. Evaluate the same photos and target runner for every system; freeze target selection before examining test predictions.
2. Preserve raw outputs, leading zeros, model and weight identities, configuration, and input hashes.
3. Count failures, missing predictions, and abstentions explicitly. Keep all planned trials in the accuracy denominator.
4. Compare complete image-pipeline runtime; report loading, warmup, preprocessing, generation or OCR stages separately.
5. Record hardware and memory measurement methods. Distinguish model effects from hardware changes.
6. Use development photos to tune systems. Freeze new evaluation data and settings before final comparisons.
7. Preserve the legacy cascade's settings and extraction behavior where possible. Name and disclose behavior changes.
8. Model loading and network downloads must be explicit. Imports, help, tests, and resource inspection must not fetch weights.

## Sprint 1 — Runnable shared experiment framework

**Outcome:** a consistent interface for checking inputs, running a system, and saving results.

- Implement the runner and a system-adapter interface with explicit check and offline-run modes.
- Connect existing configuration, manifest validation, and scoring modules.
- Save unique run directories containing configuration, provenance, raw per-trial JSONL, CSV, and a summary.
- Record failures and unattempted trials without dropping them from planned counts.
- Add meaningful tests for validation before model loading, failure recording, and safe imports/help.

**Done when:** a fake adapter exercises the complete result lifecycle, invalid inputs fail before model construction, and focused tests pass. This does not count as an inference experiment.

## Sprint 2 — Reproduce the old cascade

**Outcome:** the existing detector/OCR approach can run independently of its old application.

- Isolate person detection, bib detection, and OCR from database/cache/UI dependencies.
- Accept explicit local weight paths and Tesseract location; record weight hashes and stage settings.
- Retain raw OCR text and legacy joined-digit output as separate fields.
- Implement the agreed target-selection policy with explicit miss and ambiguity outcomes.
- Replace fixed debug filenames with optional per-run artifacts.
- Report missing inputs as blocked readiness; test missing weights, no detections, ambiguity, and OCR errors.

**Done when:** an actual three-photo cascade run saves target-level outputs and full-pipeline measurements, with preserved settings or documented differences from the legacy implementation.

**External inputs:** detector weights and provenance, Tesseract, approved execution environment, and agreed target policy.

## Sprint 3 — VLM baseline and first comparison

**Outcome:** the cascade and a VLM can be evaluated through the same framework.

- Implement a pinned SmolVLM adapter and verified offline-cache checks.
- Implement a separate explicit download mode with a finite cap and cache verification.
- Measure model loading, warmup, per-image execution, and CPU/GPU memory with named methods.
- Run the shared three-photo development set through SmolVLM and the cascade.
- Implement comparison reports that reject incompatible datasets or protocols.
- Update setup instructions and project status from saved evidence.

**Done when:** actual runs produce a reviewable side-by-side report with strict and normalized accuracy, failures, runtime, memory, and provenance. Three photos establish smoke-test behavior, not general performance.

## Sprint 4 — GPU and additional VLM baseline

**Outcome:** evaluate the larger VLM under an explicit GPU configuration.

- Implement the pinned Pixtral adapter and configured quantization without silent fallback.
- Complete runtime/cache/resource checks and selected-GPU isolation.
- Run SmolVLM and Pixtral sequentially on the same allocated GPU where practical.
- Compare them with the cascade, stating hardware differences explicitly.

**Done when:** authorized offline GPU runs and compatible reports are saved. Keep this sprint conditional on model choices, allocation, and download/storage approval.

## Sprint 5 — Dataset and reliability evaluation

**Outcome:** move from development examples to a credible evaluation set.

- Collect real permitted photos and review bib labels and target identities.
- Build the fixed 100-photo collection using the annotation tool. Working assumption: 100 total, including the three development photos; report these separately from the remaining 97. Confirm only if 100 additional held-out photos are intended.
- Annotate targets if the selected matching protocol requires boxes.
- Review duplicates, event overlap, difficult cases, and unreadable-bib policy.
- Freeze dataset version, splits, inclusion rules, metrics, and system settings.
- Run all ready baselines and report error categories and uncertainty alongside counts.

**Done when:** the agreed number of unique reviewed records passes validation and every planned trial has a recorded outcome or explicit unattempted status.

**Parallel preparation:** collecting and reviewing data can begin during earlier coding sprints.

## Sprint 6 — Hybrid systems and alternative cascades

**Outcome:** compare controlled architectural changes using the established framework.

Selected Phase 2 experiments:

- YOLO person/bib detection → VLM recognition of the detected crop.
- YOLO person detection → VLM bib localization/recognition within the person crop.
OCR-first fallback, other detectors, and alternative OCR engines are deferred until the two selected hybrids and baselines are complete.

Give each variant a versioned configuration. Keep the full-image baseline separate from crop-based protocols and disclose the detector's contribution. Count localization misses and fallback costs in the complete system result. Tune routing rules on development data only; define detector/OCR confidence signals before using them to route samples.

**Done when:** selected hybrids have saved runs, declared routing and target policies, failure breakdowns, and comparisons against frozen baselines. Choose variants based on development results without repeatedly tuning against the final evaluation set.

## Sprint workflow

For each sprint: agree on the outcome, implement a reviewable increment, run focused checks, execute real inference only when inputs and resources are available, and record evidence and remaining blockers. Code completion and experiment completion are separate statuses. Update this document as decisions and results become concrete.

## Recorded project decisions

1. **Deliverables:** paper and demonstration by the end of November 2026.
2. **Hardware/configuration:** laptop component tests and small runs; larger RENN runs; configurable model and hardware profiles with no source edits to switch systems.
3. **Cascade weights:** YOLOv8s person detector and local custom bib weights. Current legacy path selects `number632resol.pt`; verify weight identity and hash before freezing the baseline.
4. **Task:** one runner per image/trial. Target matching still needs a concrete policy for multi-person photos.
5. **VLMs:** SmolVLM and Pixtral only for Phase 1.
6. **Dataset:** fixed 100 photos; three available now. Working assumption is 100 total, with the three development examples reported separately.
7. **Annotation:** researcher enters bib numbers in a separate image-viewing program; CSV labels feed benchmark verification. Unreadable/ambiguous-label policy remains open.
8. **Priority:** exact bib accuracy, with raw and normalized outputs retained.
9. **Practical use:** live background processing; profile speed alongside accuracy and memory. A specific latency/FPS requirement remains open.
10. **Phases:** pure VLM versus old cascade first; person-detector → VLM and person/bib-detectors → VLM second.

## Immediate coding order

1. Build the laptop annotation tool and CSV/manifest round trip (Sprint 0).
2. Implement the shared runner and result contract (Sprint 1).
3. Integrate the existing local cascade work without import-time model loading (Sprint 2).
4. Add SmolVLM, Pixtral, portable launch profiles, and comparison reports (Sprints 3–4).
5. Freeze development-tuned settings and the reviewed dataset, then run the final comparison on RENN (Sprint 5).
6. Develop selected hybrids against development data after Phase 1 works; freeze their settings before final evaluation (Sprint 6).

Dataset annotation and collection proceed throughout the early sprints. Hybrid development must happen before the frozen final evaluation; sprint numbering describes work packages rather than a requirement to inspect final test results before building hybrids.

## Supporting references in this repository

- [README](README.md): overview and historical pilot result; implementation status needs updating.
- [Detailed coding handoff](coding_agent_10_19.md): protocol, model revisions, legacy settings, resource constraints, and acceptance requirements.
