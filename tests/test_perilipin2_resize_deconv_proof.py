"""Offline invariants for Perilipin2's deterministic nearest-to-deconvolution proof.

These tests do not need ONNX, Kneron, Docker, microscopy or the KL720 device.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "windows"
sys.path.insert(0, str(SCRIPTS))
SOURCE = SCRIPTS / "perilipin2_resize_deconv_proof.py"
SPEC = importlib.util.spec_from_file_location("perilipin2_resize_deconv_proof", str(SOURCE))
assert SPEC is not None and SPEC.loader is not None
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)


class Node:
    def __init__(self, name, op_type, inputs, outputs):
        self.name = name
        self.op_type = op_type
        self.input = list(inputs)
        self.output = list(outputs)
    def CopyFrom(self, other):
        self.name, self.op_type = other.name, other.op_type
        self.input, self.output = list(other.input), list(other.output)


class Model:
    def __init__(self, graph=None):
        self.graph = graph
    def CopyFrom(self, other):
        self.graph = copy.deepcopy(other.graph)


def geometry():
    sizes = [32, 64, 128, 256]
    channels = [96, 72, 48, 24]
    return [
        {"node_name": "/Resize_{}".format(index),
         "input_shape": [1, channels[index], size, size],
         "target_shape": [1, channels[index], size * 2, size * 2],
         "integer_scales": [1, 1, 2, 2]}
        for index, size in enumerate(sizes)
    ]


class DepthwiseDeconvProofTest(unittest.TestCase):
    def test_four_stage_geometry_and_exact_2x_only(self):
        self.assertEqual(
            [r["channels"] for r in d.validated_geometry(geometry())],
            [96, 72, 48, 24])
        bad = copy.deepcopy(geometry())
        bad[2]["target_shape"][-1] += 1
        with self.assertRaisesRegex(ValueError, "exact 2x"):
            d.validated_geometry(bad)
        bad = copy.deepcopy(geometry())
        bad[1]["node_name"] = bad[0]["node_name"]
        with self.assertRaisesRegex(ValueError, "Nonunique"):
            d.validated_geometry(bad)
        with self.assertRaisesRegex(ValueError, "four"):
            d.validated_geometry(geometry()[:3])

    def test_nonoverlapping_ones_transposed_conv_exactly_duplicates_pixels(self):
        rng = np.random.default_rng(720)
        x = rng.uniform(-0.5, 0.5, (1, 4, 5, 7)).astype(np.float32)
        expected = np.repeat(np.repeat(x, 2, axis=2), 2, axis=3)
        output = np.zeros_like(expected)
        for dy in (0, 1):
            for dx in (0, 1):
                output[:, :, dy::2, dx::2] += x
        np.testing.assert_array_equal(output, expected)

    def test_convert_removes_only_old_size_initializers_not_learned_weights(self):
        sizes = geometry()
        init = [types.SimpleNamespace(name="learned", values=np.array([0.125], np.float32))]
        nodes = []
        for index, entry in enumerate(sizes):
            name = "old_size_{}".format(index)
            init.append(types.SimpleNamespace(name=name, values=np.array(entry["target_shape"])))
            nodes.append(Node(entry["node_name"], "Resize",
                              ["feature_{}".format(index), "", "", name],
                              ["upsampled_{}".format(index)]))
        source = Model(types.SimpleNamespace(
            node=nodes, initializer=init,
            input=[types.SimpleNamespace(name="rgb_chw_normalized"),
                   types.SimpleNamespace(name="old_size_0")],
            value_info=[types.SimpleNamespace(name="old_size_0")]))
        fake_onnx = types.SimpleNamespace(
            ModelProto=Model,
            helper=types.SimpleNamespace(
                make_node=lambda kind, inputs, outputs, name, **kwargs:
                    Node(name, kind, inputs, outputs)),
            numpy_helper=types.SimpleNamespace(
                from_array=lambda array, name:
                    types.SimpleNamespace(name=name, values=array)),
            checker=types.SimpleNamespace(check_model=lambda _: None),
        )
        with mock.patch.object(d.vendor, "resize_graph_contract"):
            result, audit = d.convert(fake_onnx, np, source, sizes)
        self.assertEqual(audit["operators_before"], {"Resize": 4})
        self.assertEqual(audit["operators_after"], {"ConvTranspose": 4})
        self.assertEqual(len(audit["replacements"]), 4)
        self.assertTrue(audit["trained_parameter_values_changed"] is False)
        self.assertEqual([n.op_type for n in source.graph.node], ["Resize"] * 4)
        self.assertEqual([n.op_type for n in result.graph.node], ["ConvTranspose"] * 4)
        names = [item.name for item in result.graph.initializer]
        self.assertIn("learned", names)
        self.assertEqual([x.name for x in result.graph.input], ["rgb_chw_normalized"])
        self.assertEqual([x.name for x in source.graph.input],
                         ["rgb_chw_normalized", "old_size_0"])
        self.assertTrue(all(name not in names for name in
                            ["old_size_0", "old_size_1", "old_size_2", "old_size_3"]))
        self.assertEqual(len([name for name in names if name.startswith("m3c_deconv_")]), 4)
        for item in result.graph.initializer:
            if item.name.startswith("m3c_deconv_"):
                self.assertEqual(str(item.values.dtype), "float32")
                self.assertTrue((item.values == 1).all())
        self.assertEqual(float(result.graph.initializer[0].values[0]), 0.125)


if __name__ == "__main__":
    unittest.main()
