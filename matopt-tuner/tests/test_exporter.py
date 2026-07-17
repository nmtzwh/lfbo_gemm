import hashlib
import json
import tempfile
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from matopt.exporter import (
    ExportError,
    generated_header,
    kernel_id,
    selected_record,
    validate_bundle,
    export_package,
)


def result(record):
    return {"schema_version": 1, "selected": {"one_shot": record}}


def record(*, state="benchmarked", correct=True, stable=True):
    finalized = {"plan": {"M_blk": 8}, "scratchpad_bytes": 64}
    return {
        "state": state,
        "plan": {"M_blk": 8, "pack_b": "direct"},
        "response": {
            "finalized": finalized,
            "measurement": {"correct": correct, "stable": stable},
        },
    }


class ExporterTests(unittest.TestCase):
    def test_selected_accepts_baseline_shape(self):
        self.assertEqual(selected_record(result(record()), "one_shot")["state"], "benchmarked")

    def test_selected_rejects_unstable_or_incorrect(self):
        for kwargs in ({"stable": False}, {"correct": False}, {"state": "rejected"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ExportError):
                selected_record(result(record(**kwargs)), "one_shot")

    def test_kernel_id_is_canonical_and_sensitive(self):
        self.assertEqual(kernel_id({"a": 1, "b": 2}), kernel_id({"b": 2, "a": 1}))
        self.assertNotEqual(kernel_id({"a": 1}), kernel_id({"a": 2}))

    def test_bundle_hash_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "0.bin"
            image.write_bytes(b"machine code")
            entry = {
                "group": "matmul",
                "ordinal": 0,
                "name": "kernel",
                "file": image.name,
                "size": image.stat().st_size,
                "alignment": 64,
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "architecture": "x86_64",
            }
            self.assertEqual(len(validate_bundle({"aot_bundle": {"schema": "aot_bundle_v1", "images": [entry]}}, root)), 1)
            entry["sha256"] = "0" * 64
            with self.assertRaisesRegex(ExportError, "integrity"):
                validate_bundle({"aot_bundle": {"schema": "aot_bundle_v1", "images": [entry]}}, root)

    def test_generated_header_contains_constraints_and_unique_symbols(self):
        manifest = {
            "workload": {"m": 2, "n": 3, "k": 4, "threads": 1},
            "tuning_identity": {"description": "cpu"},
            "aot_bundle": {"architecture": "x86_64", "isa": "AVX2", "vector_bits": 256},
            "selected_measurement": {"packed_bytes": 0},
            "requested_plan": {"M_blk": 2},
            "finalized_plan": {"scratchpad_bytes": 64},
            "objective": "one_shot",
            "packing_lifecycle": "direct",
            "dynamic_dependencies": ["libgomp.so.1"],
        }
        header = generated_header("abc123", manifest)
        self.assertIn("namespace matopt::k_abc123", header)
        self.assertIn("matopt_abc123_create", header)
        self.assertIn("Construction performs no JIT", header)
        self.assertIn("MatMul(const MatMul&) = delete", header)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "consumer.cpp"
            source.write_text(
                header + "\nstatic_assert(matopt::k_abc123::constraints.k == 4);\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                ["g++", "-std=c++17", "-x", "c++", "-c", str(source), "-o", str(Path(directory) / "consumer.o")],
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_atomic_export_with_native_contract(self):
        finalized = {"plan": {"M_blk": 8}, "scratchpad_bytes": 64}
        selected = {
            "state": "benchmarked",
            "plan": {"M_blk": 8, "pack_b": "direct"},
            "response": {
                "finalized": finalized,
                "measurement": {
                    "correct": True,
                    "stable": True,
                    "packed_bytes": 0,
                },
            },
        }
        tuning = {
            "schema_version": 1,
            "runner_fingerprint": "fp",
            "workload": {"m": 2, "n": 3, "k": 4, "threads": 1, "cpus": "0"},
            "selected": {"one_shot": selected},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "plans.json"
            result_path.write_text(json.dumps(tuning), encoding="utf-8")
            runner_path = root / "runner"
            runner_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            runner_path.chmod(0o755)
            (root / "onednn").mkdir()
            output = root / "package"

            class Runner:
                def __init__(self, unused): pass
                def capabilities(self, workload):
                    return {"status": "capabilities", "fingerprint": "fp", "identity": "cpu"}
                def inspect(self, workload, plan, fingerprint):
                    return {"status": "accepted", "finalized": finalized}
                def capture_aot(self, workload, plan, fingerprint, capture_root):
                    capture_root = Path(capture_root)
                    image = capture_root / "image.bin"
                    image.write_bytes(b"code")
                    library = capture_root / "kernel.so"
                    library.write_bytes(b"ELF test double")
                    return {
                        "status": "captured",
                        "finalized": finalized,
                        "library": library.name,
                        "aot_bundle": {
                            "schema": "aot_bundle_v1",
                            "architecture": "x86_64",
                            "isa": "AVX2",
                            "vector_bits": 256,
                            "images": [{
                                "group": "matmul", "ordinal": 0, "name": "brgemm",
                                "file": image.name, "size": 4, "alignment": 64,
                                "sha256": hashlib.sha256(b"code").hexdigest(),
                                "architecture": "x86_64",
                            }],
                        },
                    }
                def validate_aot(self, workload, fingerprint, package):
                    self.package_existed_during_validation = Path(package).is_dir()
                    return {"status": "validated", "jit_events": 0}

            with mock.patch("matopt.exporter.MatOptRunner", Runner):
                exported = export_package(
                    result_path=result_path, objective="one_shot",
                    runner_path=runner_path, onednn_build=root / "onednn",
                    build_dir=root / "build", output=output,
                )
            self.assertTrue(output.is_dir())
            self.assertEqual(exported["kernel_id"], json.loads(
                (output / "share/matopt/manifest.json").read_text(encoding="utf-8")
            )["kernel_id"])
            self.assertEqual(len(list(output.parent.glob(".package.*"))), 0)

    def test_existing_output_is_not_overwritten(self):
        # Publication must never merge a new package into an old one.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "package"
            output.mkdir()
            marker = output / "owned"
            marker.write_text("keep", encoding="utf-8")
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
