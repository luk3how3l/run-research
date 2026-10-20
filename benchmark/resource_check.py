"""Read-only resource/runtime recommendation. Never imports ML libraries, installs, downloads or loads a model.

Thresholds are policy, not physics: they are exposed so a machine that really fits can be allowed explicitly.
"""
import importlib.metadata as md
import json
import os
import shutil
import subprocess
import sys

GIB = 1024 ** 3
POLICY = {
    "small_cpu": {"min_ram_gib": 6.0},
    "large_gpu": {"min_gpu_free_gib": 18.0, "max_gpu_util_pct": 10, "max_gpu_used_gib": 1.0,
                  "min_ram_gib": 12.0, "min_disk_gib": 40.0},
    "small_disk_gib": 4.0,
}
PACKAGES = ["torch", "torchvision", "transformers", "huggingface_hub", "safetensors",
            "pillow", "bitsandbytes", "accelerate"]


def _meminfo_available():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _cgroup_limit():
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            if v != "max" and int(v) < 1 << 60:
                return int(v)
        except (OSError, ValueError):
            continue
    return None


def _gpus():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        i, uuid, name, tot, used, util, drv = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(i), "uuid": uuid, "name": name, "total_mib": int(tot),
                     "used_mib": int(used), "util_pct": int(util), "driver": drv})
    return gpus


def _versions():
    out = {}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def cache_status(cache_dir, expected_files=None):
    """expected_files: {relative_path: size_bytes}. Without a list, completeness is unknown."""
    if not cache_dir or not os.path.isdir(cache_dir):
        return {"state": "missing", "detail": "cache directory absent"}
    if not expected_files:
        return {"state": "unknown", "detail": "no pinned file list supplied; cannot verify completeness"}
    missing = [f for f in expected_files if not os.path.isfile(os.path.join(cache_dir, f))]
    trunc = [f for f, n in expected_files.items() if f not in missing
             and os.path.getsize(os.path.join(cache_dir, f)) != n]
    state = "complete" if not (missing or trunc) else "incomplete"
    return {"state": state, "missing": missing, "wrong_size": trunc}


def collect_facts(cache_dir=None, expected_files=None):
    path = cache_dir if cache_dir and os.path.isdir(cache_dir) else os.getcwd()
    return {
        "ram_available_bytes": _meminfo_available(),
        "cgroup_memory_limit_bytes": _cgroup_limit(),
        "gpus": _gpus(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "disk_free_bytes": shutil.disk_usage(path).free,
        "quota": "unknown (not checked)",
        "python": sys.version.split()[0],
        "packages": _versions(),
        "cache": cache_status(cache_dir, expected_files),
        "runtime_probe": None,
    }


PROBE_CODE = """
import json, torch
r = {"torch": torch.__version__, "cuda": torch.cuda.is_available()}
torch.ones(2) + torch.ones(2)
if r["cuda"]:
    r["bf16"] = torch.cuda.is_bf16_supported()
    (torch.ones(2, device="cuda") + 1).cpu()
try:
    import bitsandbytes
    r["bitsandbytes"] = bitsandbytes.__version__
except Exception as e:
    r["bitsandbytes_error"] = repr(e)
print(json.dumps(r))
"""


def runtime_probe(gpu_uuid=None, timeout=60):
    """Explicit opt-in: imports Torch in a subprocess and runs tiny ops. Never loads a model."""
    env = dict(os.environ)
    if gpu_uuid:
        env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    try:
        r = subprocess.run([sys.executable, "-c", PROBE_CODE], capture_output=True, text=True,
                           timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"probe timed out after {timeout}s"}
    if r.returncode != 0:
        return {"ok": False, "error": (r.stderr or "probe failed").strip()[-500:]}
    return {"ok": True, **json.loads(r.stdout.strip().splitlines()[-1])}


def recommend(facts, gpu_uuid=None, policy=POLICY):
    """Return {recommendation, small_cpu, large_gpu}; each has ready flag and reasons. Pure function."""
    pk = facts["packages"]
    ram = facts["ram_available_bytes"]
    lim = facts.get("cgroup_memory_limit_bytes")
    eff_ram = min(x for x in (ram, lim) if x is not None) if (ram or lim) else None
    probe = facts.get("runtime_probe")
    cache = facts["cache"]["state"]

    def common(reasons):
        if cache != "complete":
            reasons.append(f"weights cache {cache}: an explicit capped download step is still required"
                           if cache in ("missing", "incomplete") else "weights cache completeness unknown")
        if probe is not None and not probe.get("ok"):
            reasons.append(f"runtime probe failed: {probe.get('error')}")

    small = {"ready": False, "reasons": [], "hardware_ok": True}
    sp = policy["small_cpu"]
    if eff_ram is None:
        small["reasons"].append("available RAM unknown")
        small["hardware_ok"] = False
    elif eff_ram < sp["min_ram_gib"] * GIB:
        small["reasons"].append(f"available RAM {eff_ram / GIB:.1f} GiB < policy {sp['min_ram_gib']} GiB "
                                "(policy threshold, not a measured limit)")
        small["hardware_ok"] = False
    if facts["disk_free_bytes"] < policy["small_disk_gib"] * GIB:
        small["reasons"].append("insufficient free disk for cache headroom")
        small["hardware_ok"] = False
    for need in ("torch", "torchvision", "transformers", "pillow"):
        if not pk.get(need):
            small["reasons"].append(f"package missing: {need}")
            small["hardware_ok"] = False
    common(small["reasons"])
    small["ready"] = small["hardware_ok"] and not small["reasons"]

    large = {"ready": False, "reasons": [], "hardware_ok": True}
    lp = policy["large_gpu"]
    gpu = None
    if not gpu_uuid:
        large["reasons"].append("no explicit GPU UUID selected")
    elif facts.get("gpus") is None:
        large["reasons"].append("nvidia-smi unavailable; GPU state unknown")
    else:
        gpu = next((g for g in facts["gpus"] if g["uuid"] == gpu_uuid), None)
        if gpu is None:
            large["reasons"].append(f"selected GPU {gpu_uuid} not found")
    mask = facts.get("cuda_visible_devices")
    if gpu and mask not in (None, ""):
        toks = [t.strip() for t in mask.split(",")]
        if gpu["uuid"] not in toks and str(gpu["index"]) not in toks:
            large["reasons"].append(f"inherited CUDA_VISIBLE_DEVICES={mask!r} excludes selected GPU")
            gpu = None
    if gpu:
        free = (gpu["total_mib"] - gpu["used_mib"]) / 1024
        if free < lp["min_gpu_free_gib"]:
            large["reasons"].append(f"GPU free {free:.1f} GiB < policy {lp['min_gpu_free_gib']} GiB")
        if gpu["used_mib"] / 1024 > lp["max_gpu_used_gib"]:
            large["reasons"].append(f"GPU already uses {gpu['used_mib']} MiB (busy)")
        if gpu["util_pct"] > lp["max_gpu_util_pct"]:
            large["reasons"].append(f"GPU utilization {gpu['util_pct']}% > {lp['max_gpu_util_pct']}%")
    if eff_ram is None or eff_ram < lp["min_ram_gib"] * GIB:
        large["reasons"].append(f"available RAM below policy {lp['min_ram_gib']} GiB or unknown")
    if facts["disk_free_bytes"] < lp["min_disk_gib"] * GIB:
        large["reasons"].append("insufficient free disk for large-model cache")
    for need in ("torch", "torchvision", "transformers", "bitsandbytes", "accelerate"):
        if not pk.get(need):
            large["reasons"].append(f"package missing: {need}")
    if probe is not None and probe.get("ok"):
        if not probe.get("cuda"):
            large["reasons"].append("Torch reports CUDA unavailable")
        elif not probe.get("bf16"):
            large["reasons"].append("BF16 not supported on selected GPU")
        if "bitsandbytes_error" in probe:
            large["reasons"].append(f"bitsandbytes import failed: {probe['bitsandbytes_error']}")
    elif probe is None:
        large["reasons"].append("runtime probe not run; CUDA/BF16/bitsandbytes compatibility unverified")
    large["hardware_ok"] = not large["reasons"]  # recorded before cache/probe reasons are appended
    common(large["reasons"])
    large["ready"] = not large["reasons"]

    if large["hardware_ok"] and gpu_uuid:
        rec = "large_gpu"
    elif small["hardware_ok"]:
        rec = "small_cpu"
    else:
        rec = "refuse"
    return {"recommendation": rec, "small_cpu": small, "large_gpu": large, "policy": policy}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Read-only resource check; no installs, downloads or model loads.")
    ap.add_argument("--gpu-uuid")
    ap.add_argument("--cache-dir")
    ap.add_argument("--probe", action="store_true", help="explicit runtime probe (imports Torch, tiny ops)")
    a = ap.parse_args()
    f = collect_facts(a.cache_dir)
    if a.probe:
        f["runtime_probe"] = runtime_probe(a.gpu_uuid)
    print(json.dumps({"facts": f, **recommend(f, a.gpu_uuid)}, indent=2))
