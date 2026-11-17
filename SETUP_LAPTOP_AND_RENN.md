# Laptop and RENN setup entry points

Choose the steps for the work you want to do:

1. **Label photos only:** follow [Annotate photos and prepare a benchmark batch](ANNOTATION_WORKFLOW.md). Python 3.10+ and a browser are sufficient; no GPU, pip dependencies, or model downloads are needed.
2. **Run models on a CPU laptop:** follow [the README's PC instructions](README.md#run-a-vlm-on-your-pc).
3. **Set up the GPU VM:** follow [SETUP_RENN.md](SETUP_RENN.md) for the environment, GPU checks, model downloads, and smoke tests. If that environment already works, go directly to [annotation on RENN](SETUP_RENN.md#11-annotation-and-lightweight-verification).
4. **Run your reviewed images:** export the manifest using the annotation guide, then follow [RENN's multi-image comparison steps](SETUP_RENN.md#running-a-multi-image-three-model-comparison). Pass `--manifest`; the comparison button uses the three development photos.

## Start annotation on your laptop

```bash
cd ~/run-research
mkdir -p media/unannotated media/annotated media/needs_review annotations benchmark/manifests
# Copy selected new JPG/JPEG/PNG/WebP photos into media/unannotated/ first.
python3 -m tools.annotate --csv annotations/bibs.csv
```

Open http://127.0.0.1:8765. Enter bib digits, target runner, split, and provenance, then click **Save and next**. Reviewed photos move to `media/annotated/`; skipped, unreadable, and ambiguous photos move to `media/needs_review/` and require blank bibs. The CSV retains metadata and the new paths. Restart the same command to resume; choose **All images / edit labels** to correct saved work.

Keep imported originals in `media/drive-test-trainbib/` and add selected images only once. Do not use `--images media` to ingest the entire dataset unintentionally. For SSH forwarding, counts, export commands, error recovery, and transfer to RENN, follow [the full walkthrough](ANNOTATION_WORKFLOW.md).
