"""Manifest validation. Empty scaffold slots never pass a runnable check."""
import hashlib
import json
import os
import re

RECORD_KEYS = {"id", "path", "sha256", "bib", "split", "review_status",
               "target_identity", "target_bbox", "provenance", "event_id"}
SPLITS = {"development", "evaluation"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_slot(rec):
    return rec.get("status") == "unlabelled"


def progress(manifest):
    recs = manifest.get("records", [])
    return {"total": len(recs), "empty_slots": sum(is_slot(r) for r in recs),
            "filled": sum(not is_slot(r) for r in recs)}


def _image_size(path):
    try:
        from PIL import Image
    except ImportError:
        return None
    with Image.open(path) as im:
        return im.size


def validate(manifest, root, expected_count, accepted_review=("reviewed",), require_bbox=False):
    """Return (errors, warnings). Runnable only if errors is empty."""
    errors, warnings = [], []
    recs = manifest.get("records")
    if not isinstance(recs, list):
        return ["manifest has no 'records' list"], warnings
    if len(recs) != expected_count:
        errors.append(f"expected {expected_count} records, found {len(recs)}")
    root = os.path.realpath(root)
    paths, hashes, events = {}, {}, {}
    for i, r in enumerate(recs):
        tag = r.get("id", f"#{i}")
        if is_slot(r):
            errors.append(f"{tag}: unlabelled slot")
            continue
        unknown = set(r) - RECORD_KEYS
        if unknown:
            errors.append(f"{tag}: unknown keys {sorted(unknown)}")
        for k in ("id", "path", "sha256", "bib", "split", "review_status", "target_identity"):
            if not r.get(k) or not isinstance(r.get(k), str):
                errors.append(f"{tag}: missing or non-string '{k}'")
        if not r.get("provenance") and r.get("split") == "evaluation":
            errors.append(f"{tag}: evaluation record needs provenance/permission note")
        if isinstance(r.get("bib"), str) and not re.fullmatch(r"\d+", r["bib"]):
            errors.append(f"{tag}: bib must be a digit string")
        if r.get("split") not in SPLITS:
            errors.append(f"{tag}: split must be one of {sorted(SPLITS)}")
        if r.get("review_status") not in accepted_review:
            errors.append(f"{tag}: review_status {r.get('review_status')!r} not in {list(accepted_review)}")
        p = r.get("path")
        if isinstance(p, str) and p:
            full = os.path.realpath(os.path.join(root, p))
            if os.path.isabs(p) or os.path.commonpath([root, full]) != root:
                errors.append(f"{tag}: path escapes manifest root")
                continue
            if p in paths:
                errors.append(f"{tag}: duplicate path (also {paths[p]})")
            paths[p] = tag
            if not os.path.isfile(full):
                errors.append(f"{tag}: file not found: {p}")
                continue
            digest = sha256_file(full)
            if digest != r.get("sha256"):
                errors.append(f"{tag}: sha256 mismatch")
            if digest in hashes and hashes[digest] != tag:
                errors.append(f"{tag}: duplicate image content (same as {hashes[digest]})")
            hashes[digest] = tag
            bbox = r.get("target_bbox")
            if bbox is None:
                if require_bbox:
                    errors.append(f"{tag}: target_bbox required")
            else:
                size = _image_size(full)
                ok = (isinstance(bbox, list) and len(bbox) == 4
                      and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in bbox))
                if not ok:
                    errors.append(f"{tag}: target_bbox must be [x1,y1,x2,y2] numbers")
                elif size is None:
                    errors.append(f"{tag}: cannot validate target_bbox bounds (Pillow missing)")
                else:
                    x1, y1, x2, y2 = bbox
                    if not (0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]):
                        errors.append(f"{tag}: target_bbox outside {size[0]}x{size[1]} image")
        if r.get("event_id"):
            events.setdefault(r["event_id"], set()).add(r.get("split"))
    for ev, splits in events.items():
        if len(splits) > 1:
            warnings.append(f"event {ev!r} spans splits {sorted(splits)}: review for leakage")
    warnings.append("near-duplicate / same-event leakage is not detected automatically; review by hand")
    return errors, warnings
