# Laptop development comparison — 2 November 2026

Actual run on Destopper WSL2, CPU float32 for SmolVLM, two Torch threads, full original photos, no warmup, one pass per image. No model downloads. Only the three known user-labeled development photos were evaluated.

| Photo | Label | Cascade output | Cascade pipeline seconds | SmolVLM raw answer | SmolVLM pipeline seconds |
|---|---|---|---:|---|---:|
| ryan_rockie_bib | 3187 | 3187 | 9.86 | ` 3187.` | 106.40 |
| hannah_bib_test | 2629 | 2629 | 2.38 | ` 2629.` | 123.61 |
| quinton_bib_image | 2300 | 2300 | 1.75 | ` 2300.` | 84.03 |

Both systems: 3/3 normalized exact bib matches; no failed executions. SmolVLM: 0/3 strict raw matches and 0/3 digits-only outputs, because all responses end in a period. Cascade: 3/3 exact matches for its legacy joined-digit output; original OCR text is also preserved in JSON.

Mean image pipeline time: cascade 4.66 s; SmolVLM 104.68 s.

Import/model setup: cascade 3.39 s; SmolVLM 36.56 s. Setup includes different imports and is not a model-only load comparison.

Process lifetime peak RSS: cascade 1.15 GiB; SmolVLM 3.41 GiB. Linux ru_maxrss, not phase peaks or sampled RSS. The laptop used swap during the VLM run.

## Protocol and limits

- Cascade: local yolov8s.pt → number632resol.pt → Tesseract; person confidence 0.5, bib confidence 0.28, zero padding; OCR scale 1.5, contour area 50, border 0, PSM 11, Otsu. Weight hashes are saved in cascade.json.
- Target selection is an explicit development variant: largest person box crossing the horizontal midpoint, otherwise nearest horizontal box center. Only the selected person proceeds to bib detection/OCR. The old full-frame loop processes every detected person, so this is not an exact reproduction of its full-frame runtime. No label is used for selection.
- Saved target images were visually inspected and select the intended runner on these three photos. They do not establish that the heuristic generalizes.
- VLM sees the full image and uses the shared central-foreground-runner prompt, pinned cached revision, greedy decoding, and 32 new tokens. Effective processor defaults are from installed Transformers 5.17.0.
- Cascade timing is from a final separate process after VLM completion. Earlier debugging/setup-overlap runs were excluded. VLM setup initially overlapped cascade setup; no cascade process remained during its measured predictions. Other laptop activity is uncontrolled.
- No warmup means first-image timings include lazy initialization. Single-pass times are smoke measurements, not reliable latency distributions or live-video throughput.
- Transformers emitted an invalid pad_token_id configuration warning during VLM loading. The run completed without overriding checkpoint settings; validate the inference stack before treating these as final paper results.
- This environment differs from the historical pilot (including Torch 2.14.0 and Transformers 5.17.0). These fresh results do not replace or explain the old run.
- Three development photos cannot establish general accuracy or show that either architecture wins. Pixtral was not run.

## Re-run

Use a new output directory each time, and run the commands sequentially from the repository root:

```bash
.venv/bin/python -m tools.smoke_compare --system cascade --output /tmp/cascade-new-run
.venv/bin/python -m tools.smoke_compare --system vlm --output /tmp/vlm-new-run
```

The script uses the known smoke manifest, local cascade weights, and the pinned SmolVLM cache under the current home directory. It is a bounded development harness, not the configurable shared runner planned for later sprints.
