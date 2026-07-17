import json
import tempfile
import unittest
from pathlib import Path

from matopt.protocol import Workload
from matopt.space import PlanSpace
from matopt.space_config import SpaceConfig


BASELINE = {
    "schema_version": 1,
    "M_blk": 128,
    "N_blk": 64,
    "K_blk": 256,
    "M_chunk_size": 1,
    "N_chunk_size": 1,
    "K_chunk_size": 1,
    "brgemm_batch_size": 1,
    "nthr_k": 1,
    "pack_a": "direct",
    "pack_b": "direct",
    "bd_block": 0,
    "ld_block2": 0,
    "loop_order": "default",
}

CAPABILITIES = {
    "constraints": {
        "allow_split_k": True,
        "scratchpad_per_thread_bytes": 64 * 1024 * 1024,
    },
    "domains": {
        "pack_a": ["direct", "per_call_padded"],
        "pack_b": [
            "direct",
            "per_call_n32",
            "per_call_n64",
            "persistent_n32",
            "persistent_n64",
        ],
        "bd_block": [0, 4, 5, 6, 7],
        "ld_block2": [0, 2, 3, 4],
        "loop_order": ["default", "one-load"],
    },
}


class SpaceConfigTests(unittest.TestCase):
    def test_explicit_domain_is_strict_by_default(self):
        config = SpaceConfig.from_dict({"domains": {"loop_order": ["one-load"]}})
        space = PlanSpace(
            Workload(256, 256, 1024, 4, "0-3"),
            BASELINE,
            CAPABILITIES,
            config,
        )
        self.assertFalse(config.inherit_baseline)
        self.assertEqual(space.domains["loop_order"], ["one-load"])
        self.assertFalse(space.is_allowed(BASELINE))

    def test_json_and_yaml_round_trip(self):
        value = {
            "space_schema_version": 1,
            "inherit_baseline": False,
            "domains": {
                "M_blk": {"values": [64, 96]},
                "nthr_k": {
                    "values": [1, 2],
                    "require_capability": "allow_split_k",
                },
            },
            "limits": {"minimum_parallel_work_per_thread": 0.5},
        }
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "space.json"
            yaml_path = Path(directory) / "space.yaml"
            json_path.write_text(json.dumps(value), encoding="utf-8")
            yaml_path.write_text(
                "space_schema_version: 1\n"
                "inherit_baseline: false\n"
                "domains:\n"
                "  M_blk: {values: [64, 96]}\n"
                "  nthr_k:\n"
                "    values: [1, 2]\n"
                "    require_capability: allow_split_k\n"
                "limits:\n"
                "  minimum_parallel_work_per_thread: 0.5\n",
                encoding="utf-8",
            )
            json_config = SpaceConfig.load(json_path)
            yaml_config = SpaceConfig.load(yaml_path)
            self.assertEqual(json_config.to_dict(), yaml_config.to_dict())
            self.assertEqual(json_config.hash(), yaml_config.hash())

    def test_custom_domains_baseline_inheritance_and_conditions(self):
        config = SpaceConfig.from_dict(
            {
                "inherit_baseline": False,
                "domains": {
                    "M_blk": {"values": [64, 96, 512]},
                    "M_chunk_size": {"values": [1, 2]},
                    "N_blk": {"values": [32, 64]},
                    "pack_b": {"values": ["direct", "persistent_n32"]},
                },
                "conditions": [
                    {
                        "if": {"pack_b": ["persistent_n32"]},
                        "force": {"N_blk": 32},
                    }
                ],
            }
        )
        space = PlanSpace(
            Workload(256, 256, 1024, 4, "0-3"),
            BASELINE,
            CAPABILITIES,
            config,
        )
        self.assertEqual(space.domains["M_blk"], [64, 96])
        self.assertEqual(space.domains["M_chunk_size"], [1, 2])
        plan = space.canonicalize(
            {"M_blk": 64, "pack_b": "persistent_n32", "N_blk": 64}
        )
        self.assertEqual(plan["N_blk"], 32)
        self.assertTrue(space.is_allowed(plan))

        inherited = PlanSpace(
            Workload(256, 256, 1024, 4, "0-3"),
            BASELINE,
            CAPABILITIES,
            SpaceConfig.from_dict(
                {"inherit_baseline": True, "domains": {"M_blk": [64, 96]}}
            ),
        )
        self.assertEqual(inherited.domains["M_blk"], [64, 96, 128])

    def test_rejects_unsupported_capability_and_nonconforming_limits(self):
        with self.assertRaisesRegex(ValueError, "runner-unsupported"):
            PlanSpace(
                Workload(256, 256, 256, 4, "0-3"),
                BASELINE,
                CAPABILITIES,
                SpaceConfig.from_dict({"domains": {"bd_block": [0, 8]}}),
            )
        caps = {
            **CAPABILITIES,
            "constraints": {
                **CAPABILITIES["constraints"],
                "allow_split_k": False,
            },
        }
        with self.assertRaisesRegex(ValueError, "split-K"):
            PlanSpace(
                Workload(256, 256, 256, 4, "0-3"),
                BASELINE,
                caps,
                SpaceConfig.from_dict({"domains": {"nthr_k": [1, 2]}}),
            )
        with self.assertRaisesRegex(ValueError, "exceeds the runner"):
            PlanSpace(
                Workload(256, 256, 256, 4, "0-3"),
                BASELINE,
                CAPABILITIES,
                SpaceConfig.from_dict(
                    {
                        "limits": {
                            "scratchpad_per_thread_bytes": 128 * 1024 * 1024
                        }
                    }
                ),
            )

    def test_policy_limits_prune_before_native_evaluation(self):
        scratch_limited = PlanSpace(
            Workload(256, 256, 1024, 4, "0-3"),
            BASELINE,
            CAPABILITIES,
            SpaceConfig.from_dict(
                {"limits": {"scratchpad_per_thread_bytes": 1024}}
            ),
        )
        padded = scratch_limited.canonicalize(
            {"M_blk": 64, "K_blk": 256, "pack_a": "per_call_padded"}
        )
        self.assertFalse(scratch_limited.is_allowed(padded))

        parallel_limited = PlanSpace(
            Workload(256, 256, 1024, 8, "0-7"),
            BASELINE,
            CAPABILITIES,
            SpaceConfig.from_dict(
                {"limits": {"minimum_parallel_work_per_thread": 2.0}}
            ),
        )
        coarse = parallel_limited.canonicalize(
            {"M_blk": 256, "N_blk": 64, "N_chunk_size": 8, "nthr_k": 1}
        )
        self.assertFalse(parallel_limited.is_allowed(coarse))


if __name__ == "__main__":
    unittest.main()
