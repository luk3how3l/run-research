#!/usr/bin/env bash
set -euo pipefail

# Preview with: bash pull_bib_images.sh --dry-run
# Download with: bash pull_bib_images.sh
remote="${DRIVE_REMOTE:-utahtech}"
dest="${DEST:-$HOME/run-research/media/bib_training_source}"
source="${remote}:BibProject/Test-TrainBib"

exec rclone copy "$source" "$dest" \
  --progress --ignore-case \
  --include '*.jpg' --include '*.jpeg' --include '*.png' \
  --include '*.webp' --include '*.bmp' \
  --include '*.tif' --include '*.tiff' \
  "$@"
