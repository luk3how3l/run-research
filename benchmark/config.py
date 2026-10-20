"""Settings resolution: CLI > config file > defaults. Unknown keys and bad combinations are rejected."""
import argparse
import json
import re

from .scoring import PROMPT

DEFAULTS = {
    "model_id": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "revision": "a7da5b986cb59b408707209984f360a5f4ad7e47",
    "adapter": "smolvlm",
    "device": "cpu",
    "gpu_uuid": None,
    "dtype": "float32",
    "quantization": None,
    "cache_dir": None,
    "manifest": "benchmark/manifests/smoke3.json",
    "prompt": PROMPT,
    "max_new_tokens": 32,
    "warmup": 0,
    "repetitions": 1,
    "threads": 2,
    "max_image_side": None,
    "output_dir": None,
}
NULLABLE = {"gpu_uuid", "quantization", "cache_dir", "max_image_side", "output_dir"}
ADAPTERS = {"smolvlm", "pixtral"}
DTYPES = {"float32", "bfloat16", "float16"}


class ConfigError(ValueError):
    pass


def _check_keys(d, where):
    unknown = set(d) - set(DEFAULTS)
    if unknown:
        raise ConfigError(f"unknown keys in {where}: {sorted(unknown)}")


def resolve(config=None, cli=None):
    """cli/config contain only explicitly given keys. A value of None clears a nullable key."""
    config, cli = config or {}, cli or {}
    _check_keys(config, "config")
    _check_keys(cli, "cli")
    out = dict(DEFAULTS)
    for src in (config, cli):
        for k, v in src.items():
            if v is None and k not in NULLABLE:
                raise ConfigError(f"{k} cannot be cleared")
            out[k] = v
    validate(out)
    return out


def validate(c):
    if c["adapter"] not in ADAPTERS:
        raise ConfigError(f"unsupported adapter {c['adapter']!r}; choose from {sorted(ADAPTERS)}")
    if not isinstance(c["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", c["revision"]):
        raise ConfigError("revision must be a full 40-hex commit hash (no branches or tags)")
    if c["device"] not in ("cpu", "cuda"):
        raise ConfigError("device must be 'cpu' or 'cuda'")
    if c["dtype"] not in DTYPES:
        raise ConfigError(f"dtype must be one of {sorted(DTYPES)}")
    if c["quantization"] not in (None, "nf4"):
        raise ConfigError("quantization must be null or 'nf4'")
    if c["device"] == "cpu":
        if c["quantization"]:
            raise ConfigError("quantization requires device=cuda")
        if c["dtype"] != "float32":
            raise ConfigError("CPU runs must use float32")
        if c["gpu_uuid"]:
            raise ConfigError("gpu_uuid is only valid with device=cuda")
    else:
        if not c["gpu_uuid"] or not str(c["gpu_uuid"]).startswith("GPU-"):
            raise ConfigError("device=cuda requires an explicit gpu_uuid (GPU-...); no auto device selection")
    for k, lo in (("max_new_tokens", 1), ("repetitions", 1), ("threads", 1), ("warmup", 0)):
        if not isinstance(c[k], int) or isinstance(c[k], bool) or c[k] < lo:
            raise ConfigError(f"{k} must be an integer >= {lo}")
    if c["max_image_side"] is not None and (not isinstance(c["max_image_side"], int) or c["max_image_side"] < 1):
        raise ConfigError("max_image_side must be a positive integer or null")


def _none_or(typ):
    def conv(s):
        return None if s.lower() == "none" else typ(s)
    return conv


def build_parser():
    """Every option defaults to SUPPRESS so only explicit CLI values override the config. 'none' clears nullables."""
    p = argparse.ArgumentParser(description="Bib-number VLM benchmark (importing/--help never loads models).")
    p.add_argument("--config", help="JSON preset")
    p.add_argument("--mode", choices=["check", "download", "run"], default="check")
    for k, v in DEFAULTS.items():
        typ = int if isinstance(v, int) and not isinstance(v, bool) else str
        if k in ("max_image_side",):
            typ = int
        p.add_argument("--" + k.replace("_", "-"), dest=k, default=argparse.SUPPRESS,
                       type=_none_or(typ) if k in NULLABLE else typ)
    # Approval flags live on the CLI only, never in a config file.
    p.add_argument("--download-cap-gb", type=float, default=None,
                   help="finite positive decimal-GB cap; required for --mode download")
    return p


def load_settings(argv):
    ns = build_parser().parse_args(argv)
    cli = {k: v for k, v in vars(ns).items() if k in DEFAULTS}
    config = {}
    if ns.config:
        with open(ns.config, encoding="utf-8") as f:
            config = json.load(f)
    if ns.mode == "download" and not (ns.download_cap_gb and 0 < ns.download_cap_gb < float("inf")):
        raise ConfigError("--mode download requires a finite positive --download-cap-gb")
    return resolve(config, cli), ns
