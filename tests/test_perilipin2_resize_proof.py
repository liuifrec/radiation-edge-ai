"""Offline safety tests for the host-only KL720 Resize compatibility probe."""
from __future__ import annotations

import importlib.util
import types
import unittest
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "scripts/windows/perilipin2_resize_proof.py"
SPEC = importlib.util.spec_from_file_location("perilipin2_resize_proof", str(SOURCE))
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class M3CResizeProofTest(unittest.TestCase):
    def node(self, name, interpolation="nearest"):
        attrs = [
            types.SimpleNamespace(name="mode", type=3, s=interpolation.encode()),
            types.SimpleNamespace(name="coordinate_transformation_mode", type=3, s=b"asymmetric"),
            types.SimpleNamespace(name="nearest_mode", type=3, s=b"floor"),
        ]
        return types.SimpleNamespace(
            name=name, op_type="Resize",
            input=["tensor", "roi", "scales", "sizes"], attribute=attrs,
        )

    def test_four_nodes_with_required_nearest_attributes(self):
        model = types.SimpleNamespace(
            graph=types.SimpleNamespace(node=[
                self.node("/Resize_{}".format(i)) for i in range(4)
            ])
        )
        self.assertEqual(len(mod.resize_nodes(model)), 4)
        model.graph.node[1].attribute[0].s = b"linear"
        with self.assertRaisesRegex(ValueError, "interpolation"):
            mod.resize_nodes(model)
        model.graph.node.pop()
        with self.assertRaisesRegex(ValueError, "exactly four"):
            mod.resize_nodes(model)

    def test_fixed_size_input_required(self):
        nodes = [self.node(str(i)) for i in range(4)]
        nodes[2].input[3] = ""
        model = types.SimpleNamespace(graph=types.SimpleNamespace(node=nodes))
        with self.assertRaisesRegex(ValueError, "sizes input"):
            mod.resize_nodes(model)

    def test_parity_rejects_decision_flips_even_with_small_float_error(self):
        good = {"max_abs_error": 0., "mean_abs_error": 0.,
                "foreground_decision_mismatches": 0,
                "boundary_decision_mismatches": 0}
        self.assertTrue(mod.strict_parity_pass([good, good], 1e-6, 1e-7))
        bad = dict(good, foreground_decision_mismatches=1)
        self.assertFalse(mod.strict_parity_pass([good, bad], 1e-6, 1e-7))
        bad = dict(good, max_abs_error=0.001)
        self.assertFalse(mod.strict_parity_pass([good, bad], 1e-6, 1e-7))

    def test_synthetic_probe_has_locked_contract(self):
        left, right = mod.two_inputs(np)
        self.assertEqual(left.shape, (1, 3, 512, 512))
        self.assertEqual(right.shape, (1, 3, 512, 512))
        self.assertEqual(left.dtype, np.float32)
        self.assertEqual(right.dtype, np.float32)
        self.assertTrue(np.isfinite(left).all())
        self.assertTrue(np.isfinite(right).all())
        self.assertFalse(np.array_equal(left, right))
        self.assertTrue(np.all(left <= 0.5))
        self.assertTrue(np.all(left >= -0.5))


if __name__ == "__main__":
    unittest.main()
