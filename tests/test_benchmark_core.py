import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from benchmark import config, manifest, resource_check as rc, scoring  # noqa: E402

GIB = 1024 ** 3
UUID = "GPU-aaaa"


class Scoring(unittest.TestCase):
    def test_cases(self):
        s = scoring.score_response("318.", "3187")
        self.assertFalse(s["strict_exact"]); self.assertFalse(s["normalized_exact"]); self.assertFalse(s["digits_only"])
        self.assertTrue(scoring.score_response(" 2300\n", "2300")["strict_exact"])
        self.assertTrue(scoring.score_response("2300.", "2300")["normalized_exact"])
        self.assertTrue(scoring.score_response("007", "007")["strict_exact"])
        self.assertFalse(scoring.score_response("7", "007")["normalized_exact"])
        u = scoring.score_response("UNKNOWN", "123")
        self.assertTrue(u["abstained"] and u["no_digits"])
        self.assertTrue(scoring.score_response("", "1")["empty"])
        m = scoring.score_response("12 34", "12")
        self.assertTrue(m["multiple_digit_groups"] and m["normalized_exact"])
        self.assertEqual(scoring.extract_bib_digits("bib x12y 7"), "7")
        self.assertEqual(scoring.extract_bib_digits("x12y"), "12")

    def test_levenshtein_and_cer(self):
        self.assertEqual(scoring.levenshtein("kitten", "sitting"), 3)
        self.assertAlmostEqual(scoring.score_response("318", "3187")["cer"], 0.25)

    def test_denominator_keeps_failures(self):
        ok = {"status": "ok", "score": scoring.score_response("1", "1")}
        r = scoring.summarize([ok, {"status": "failed", "score": None}], planned=5)
        self.assertEqual((r["planned"], r["failed"], r["unattempted"], r["normalized_exact"]), (5, 1, 3, 1))


class Config(unittest.TestCase):
    def test_precedence_and_clear(self):
        c = config.resolve({"threads": 4, "max_image_side": 512}, {"threads": 8})
        self.assertEqual((c["threads"], c["max_image_side"]), (8, 512))
        self.assertIsNone(config.resolve({"max_image_side": 512}, {"max_image_side": None})["max_image_side"])

    def test_rejections(self):
        bad = [({"bogus": 1}, "unknown"), ({"revision": "main"}, "40-hex"), ({"adapter": "qwen"}, "unsupported"),
               ({"device": "cuda"}, "gpu_uuid"), ({"quantization": "nf4"}, "requires device=cuda"),
               ({"dtype": "bfloat16"}, "float32"), ({"warmup": -1}, "warmup"), ({"adapter": None}, "cleared")]
        for cfg, msg in bad:
            with self.assertRaisesRegex(config.ConfigError, msg):
                config.resolve(cfg)

    def test_gpu_preset_shape(self):
        p = json.load(open(os.path.join(ROOT, "benchmark/configs/gpu-large.json")))
        self.assertEqual(config.resolve(p)["quantization"], "nf4")
        config.resolve(json.load(open(os.path.join(ROOT, "benchmark/configs/cpu-small.json"))))

    def test_cli_and_download_cap(self):
        s, _ = config.load_settings(["--threads", "3", "--max-image-side", "none"])
        self.assertEqual(s["threads"], 3)
        with self.assertRaises(config.ConfigError):
            config.load_settings(["--mode", "download"])
        with self.assertRaises(config.ConfigError):
            config.load_settings(["--mode", "download", "--download-cap-gb", "0"])


class Manifest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.root = self.d.name
        self.recs = []
        for i, data in enumerate([b"aaa", b"bbb"]):
            with open(os.path.join(self.root, f"{i}.jpg"), "wb") as f:
                f.write(data)
            self.recs.append({"id": f"r{i}", "path": f"{i}.jpg", "sha256": hashlib.sha256(data).hexdigest(),
                              "bib": "0123", "split": "development", "review_status": "reviewed",
                              "target_identity": "central"})

    def tearDown(self):
        self.d.cleanup()

    def check(self, recs, n=2, **kw):
        return manifest.validate({"records": recs}, self.root, n, **kw)[0]

    def test_ok(self):
        self.assertEqual(self.check(self.recs), [])

    def test_failures(self):
        r = self.recs
        self.assertTrue(any("expected 3" in e for e in self.check(r, 3)))
        self.assertTrue(any("slot" in e for e in self.check([r[0], {"id": "s", "status": "unlabelled"}])))
        self.assertTrue(any("sha256" in e for e in self.check([dict(r[0], sha256="0" * 64), r[1]])))
        self.assertTrue(any("duplicate path" in e for e in self.check([r[0], dict(r[0], id="x")])))
        self.assertTrue(any("escapes" in e for e in self.check([dict(r[0], path="../x.jpg"), r[1]])))
        self.assertTrue(any("not found" in e for e in self.check([dict(r[0], path="nope.jpg"), r[1]])))
        self.assertTrue(any("digit string" in e for e in self.check([dict(r[0], bib="12a"), r[1]])))
        self.assertTrue(any("review_status" in e for e in self.check([dict(r[0], review_status="draft"), r[1]])))
        self.assertTrue(any("bbox" in e for e in self.check([dict(r[0], target_bbox=[1, 2]), r[1]])))
        self.assertTrue(any("target_bbox required" in e for e in self.check(r, require_bbox=True)))

    def test_duplicate_content(self):
        with open(os.path.join(self.root, "c.jpg"), "wb") as f:
            f.write(b"aaa")
        dup = dict(self.recs[0], id="dup", path="c.jpg")
        self.assertTrue(any("duplicate image content" in e for e in self.check([self.recs[0], dup])))

    def test_repo_manifests(self):
        sm = manifest.load_manifest(os.path.join(ROOT, "benchmark/manifests/smoke3.json"))
        self.assertEqual(manifest.validate(sm, ROOT, 3, accepted_review=("user_supplied", "reviewed"))[0], [])
        sc = manifest.load_manifest(os.path.join(ROOT, "benchmark/manifests/scaffold100.json"))
        self.assertEqual(manifest.progress(sc), {"total": 100, "empty_slots": 97, "filled": 3})
        self.assertNotEqual(manifest.validate(sc, ROOT, 100)[0], [])


def facts(**kw):
    f = {"ram_available_bytes": 32 * GIB, "cgroup_memory_limit_bytes": None,
         "gpus": [{"index": 0, "uuid": UUID, "name": "L4", "total_mib": 23034, "used_mib": 3,
                   "util_pct": 0, "driver": "580"}],
         "cuda_visible_devices": None, "disk_free_bytes": 100 * GIB, "quota": "unknown",
         "packages": {p: "1" for p in rc.PACKAGES}, "cache": {"state": "complete"},
         "runtime_probe": {"ok": True, "cuda": True, "bf16": True}}
    f.update(kw)
    return f


class Resources(unittest.TestCase):
    def rec(self, f, uuid=UUID):
        return rc.recommend(f, uuid)

    def test_suitable_gpu(self):
        r = self.rec(facts())
        self.assertEqual(r["recommendation"], "large_gpu"); self.assertTrue(r["large_gpu"]["ready"])

    def test_cpu_only(self):
        r = self.rec(facts(gpus=None, runtime_probe=None), uuid=None)
        self.assertEqual(r["recommendation"], "small_cpu")

    def test_busy_gpu(self):
        g = [dict(facts()["gpus"][0], used_mib=12991, util_pct=99)]
        r = self.rec(facts(gpus=g))
        self.assertFalse(r["large_gpu"]["ready"]); self.assertNotEqual(r["recommendation"], "large_gpu")
        self.assertTrue(any("busy" in x for x in r["large_gpu"]["reasons"]))

    def test_insufficient(self):
        self.assertEqual(self.rec(facts(ram_available_bytes=2 * GIB, gpus=None), None)["recommendation"], "refuse")
        self.assertFalse(self.rec(facts(disk_free_bytes=GIB))["large_gpu"]["hardware_ok"])
        g = [dict(facts()["gpus"][0], total_mib=8000)]
        self.assertFalse(self.rec(facts(gpus=g))["large_gpu"]["ready"])

    def test_cgroup_limit_wins(self):
        r = self.rec(facts(cgroup_memory_limit_bytes=4 * GIB, gpus=None), None)
        self.assertEqual(r["recommendation"], "refuse")

    def test_missing_cache_not_ready(self):
        r = self.rec(facts(cache={"state": "missing"}))
        self.assertEqual(r["recommendation"], "large_gpu"); self.assertFalse(r["large_gpu"]["ready"])
        self.assertTrue(any("download" in x for x in r["large_gpu"]["reasons"]))

    def test_dependency_mismatch(self):
        p = dict(facts()["packages"], bitsandbytes=None)
        self.assertFalse(self.rec(facts(packages=p))["large_gpu"]["ready"])

    def test_mask_mismatch(self):
        r = self.rec(facts(cuda_visible_devices="1"))
        self.assertTrue(any("CUDA_VISIBLE_DEVICES" in x for x in r["large_gpu"]["reasons"]))
        self.assertTrue(self.rec(facts(cuda_visible_devices=UUID))["large_gpu"]["ready"])

    def test_probe_failure(self):
        r = self.rec(facts(runtime_probe={"ok": False, "error": "libcudart missing"}))
        self.assertTrue(any("libcudart" in x for x in r["large_gpu"]["reasons"]))
        self.assertFalse(r["large_gpu"]["ready"])

    def test_no_gpu_selected(self):
        self.assertFalse(self.rec(facts(), uuid=None)["large_gpu"]["ready"])

    def test_cache_status(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a"), "wb") as f:
                f.write(b"12")
            self.assertEqual(rc.cache_status(d)["state"], "unknown")
            self.assertEqual(rc.cache_status(d, {"a": 2})["state"], "complete")
            self.assertEqual(rc.cache_status(d, {"a": 3})["wrong_size"], ["a"])
            self.assertEqual(rc.cache_status(d, {"b": 1})["missing"], ["b"])
        self.assertEqual(rc.cache_status("/nonexistent")["state"], "missing")


class NoHeavyImports(unittest.TestCase):
    def test_start_help_and_modules_do_not_import_ml(self):
        code = ("import sys; sys.argv=['start.py','--help']; sys.path.insert(0,%r);"
                "import benchmark.config, benchmark.manifest, benchmark.resource_check, benchmark.scoring;"
                "assert not {'torch','transformers'} & set(sys.modules)" % ROOT)
        subprocess.run([sys.executable, "-c", code], check=True)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "start.py"), "--help"],
                           capture_output=True, text=True, cwd=tempfile.gettempdir())
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
