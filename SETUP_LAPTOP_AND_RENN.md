# Setup entry points

For the GPU VM you can already SSH into, use **[SETUP_RENN.md](SETUP_RENN.md)**. It contains the complete ordered installation, GPU diagnosis, manual SmolVLM/Pixtral downloads, and run commands.

For CPU development on your PC, use **[the README's PC instructions](README.md#run-a-vlm-on-your-pc)**. For labeling only, Python 3.10+ and a browser are sufficient:

```bash
cd /path/to/run-research
python3 -m tools.annotate --images media --csv annotations/bibs.csv
```

Open http://127.0.0.1:8765. Follow [the annotation instructions](README.md#label-photos) to save, resume, edit, and export labels. No GPU or pip dependencies are needed for this tool.

The previous combined setup guide has been replaced so RENN installation can be followed in one place.
