# Run Research

Can a vision-language model read race bib numbers more accurately than our person-detector → bib-detector → OCR pipeline, and what does each approach cost in time and memory?

We want to compare three systems on the **same photos and the same runner in each photo**:

1. **SmolVLM-500M-Instruct**: the small model for a local CPU baseline.
2. **Pixtral-12B**: the larger model for the Renn university GPU machine.
3. **YOLO + OCR cascade**: the existing detection and Tesseract approach, once its dependencies and target-selection behavior are made testable.

**Current status, 19 October 2026:** one three-photo SmolVLM CPU experiment has completed. The Pixtral run, portable benchmark implementation, working cascade comparison and 100-image evaluation are still pending. This update adds documentation only; it does not install or implement the planned tools.

If you are continuing the coding work, start with [coding_agent_10_19.md](coding_agent_10_19.md). It contains the implementation plan, known blockers, test requirements and stopping rules.

## What we have actually measured

The completed experiment ran on **Destopper's Ubuntu 24.04 WSL environment**, using an Intel Core i5-1235U and CPU float32. It was not a new run on the Renn machine. If “laptop” means a different machine, that reproduction still needs to happen.

It used the original full photos, one model load, no warmup, one pass per image, greedy decoding and a 32-token output limit.

| Photo | Correct bib | Raw answer | Extracted bib | Correct after extraction? | Generation time |
|---|---|---|---|---|---|
| Ryan | 3187 | `318.` | 318 | No | 42.8642 s |
| Hannah | 2629 | `209.` | 209 | No | 39.6320 s |
| Quinton | 2300 | `2300.` | 2300 | Yes | 40.9939 s |

**Result: 1/3 normalized exact matches and 0/3 strict raw exact matches.** All three images executed successfully. The model loaded in 5.3446 seconds. Two different memory-observation methods recorded approximately 3.84 GB sampled peak RSS and 3.90 GB process high-water RSS; they are not interchangeable measurements.

These are three observed development examples, not proof of general accuracy or evidence that one system wins. We have no Pixtral or cascade results to compare yet.

## Milestone 1: Local / laptop baseline

**Partly complete.** The historical Destopper CPU result above is preserved. Next, make the benchmark reproducible through a clean entry point and rerun the same three photos with saved configuration, raw answers, timing and memory measurements. Reproduce on a separate laptop only if that is a different intended target.

Done means another person can follow the documented setup and get a recorded run, including an honest error report if dependencies, storage or memory are insufficient. It does not mean the model must answer every bib correctly.

## Milestone 2: Renn GPU comparison

**Pending.** Run Pixtral-12B on the same photos, labels, prompt and scoring protocol as SmolVLM. Also run SmolVLM on the same GPU when practical, so hardware changes are not mistaken for a model-only speed improvement.

Pixtral's original BF16 weights contain **25,365,479,424 bytes, about 25.4 GB**. That is its weight payload, not a measured RAM or VRAM requirement. The proposed single-GPU setup uses **4-bit NF4 quantization** to reduce runtime GPU memory. It still downloads the original weights, and quantization may affect accuracy.

The supplied Renn snapshot showed two NVIDIA L4 GPUs, each with 23,034 MiB. GPU 0 was idle; GPU 1 was busy at 99% utilization. This is a historical snapshot, not a reservation. Recheck availability, confirm university permission and use only the explicitly allocated GPU. Do not assume an empty process list means a GPU is free.

No Slurm setup is assumed. Do not run inference on a login node or disturb another person's job.

## Milestone 3: Portability and resource checks

**Planned, not yet implemented in this repository.** Add a separate checker that reports:

- Available RAM, including container or job limits where detectable
- GPU identity, free/used VRAM and utilization
- Disk space, selected cache contents and unknown personal quota
- Installed package versions and runtime compatibility
- Whether the small CPU or larger single-GPU profile is a reasonable candidate, or why it must refuse

Keep the checker separate from model execution. Detecting capable hardware must never trigger a download or inference run. A recommendation is not proof that a model fits.

The evaluator should use configuration files and command-line overrides rather than hardcoding a model. Provide clear CPU and GPU launch scripts, matching platform-specific Torch instructions, common dependencies and an optional CUDA quantization dependency file. Downloads must be explicit, revision-pinned and capped; inference should work offline from a verified cache.

## Milestone 4: Grow from 3 photos to 100

**Pending real photos and reviewed labels.** Create one shared manifest containing image paths, SHA256 hashes, bib labels as strings and the target runner. Check all 100 records before running any model.

Do not generate fake labels or pretend empty template rows are data. A scaffold may contain the three known records and 97 clearly unfilled slots. Keep duplicate images out. Decide the target, labels, inclusion rules and parameters before viewing test predictions.

The existing three photos have already been used for development. If they are included in a 100-photo collection, report their results separately from the 97 new evaluation photos. Do not describe all 100 as an untouched held-out test set.

The final three-way comparison needs:

- The same complete image list, labels and target runner for every system
- Raw and normalized accuracy, plus missing, invalid and failed predictions
- Per-image end-to-end runtime, with load and warmup reported separately
- Clearly labeled CPU and GPU peak-memory measurements
- Saved model revisions, parameters, software versions and hardware details

## What is blocking the cascade?

The source is in [vision/background_yolo.py](vision/background_yolo.py). It currently imports missing database/cache modules, loads models during import, and contains machine-specific Windows model and Tesseract paths. The custom bib detector weights are not included in this repository.

It also processes **all detected people**, while the VLM prompt asks for the **central foreground runner**. We need one explicit target-matching rule before comparing scores. A future clean adapter should take local weight and Tesseract paths, avoid database/cache side effects, and clearly distinguish any changed behavior from the legacy pipeline. Until those inputs and rules exist, report the cascade as blocked rather than inventing a result.

## Continue the work

The detailed coding handoff is [coding_agent_10_19.md](coding_agent_10_19.md). It describes the existing draft bundle, suggested file layout, milestone acceptance tests and the inputs still needed.

The original `start.py` has an import-path issue, and `test_qwen.py` loads a model during import. Do not use broad test discovery as a harmless setup check. The handoff calls for a separate, dependency-light test suite and explicit execution commands.

For now, pulling `main` retrieves these instructions and the existing research code. It does **not** mean the proposed resource checker, adapters or new benchmark have been implemented.

### Model references

- [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)
- [Transformers Pixtral guide, including NF4](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/pixtral)
- [Pinned Transformers-compatible Pixtral checkpoint](https://huggingface.co/mistral-experimental/pixtral-12b/tree/c2756cbbb9422eba9f6c5c439a214b0392dfc998)
