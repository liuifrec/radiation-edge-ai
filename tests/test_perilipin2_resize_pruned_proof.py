"""Offline safeguards for immutable dead-shape-branch ONNX pruning.

No Docker, KL720, ONNX binary, or microscopy data required.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1] / "scripts" / "windows"
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "perilipin2_resize_pruned_proof.py"
SPEC = importlib.util.spec_from_file_location("perilipin2_resize_pruned_proof", str(SOURCE))
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def node(op, inputs, outputs):
    return types.SimpleNamespace(
        op_type=op, input=list(inputs), output=list(outputs), attribute=[],
    )


def value(name):
    return types.SimpleNamespace(name=name)


def graph():
    return types.SimpleNamespace(
        input=[value("x")],
        output=[value("y")],
        initializer=[value("sizes"), value("unused_initializer")],
        value_info=[value("shape"), value("concat"), value("conv"), value("up")],
        node=[
            node("Conv", ["x"], ["conv"]),
            node("Shape", ["x"], ["shape"]),
            node("Concat", ["shape"], ["concat"]),
            node("Resize", ["conv", "", "", "sizes"], ["up"]),
            node("Conv", ["up"], ["y"]),
        ],
        sparse_initializer=[],
    )


class Model:
    def __init__(self, graph_obj=None):
        self.graph = graph_obj
    def CopyFrom(self, other):
        self.graph = copy.deepcopy(other.graph)


class PrunedGraphTest(unittest.TestCase):
    def test_reachability_drops_unused_shape_branch(self):
        g = graph()
        keep, names = module.reachable_nodes(g)
        self.assertEqual(keep, {0, 3, 4})
        self.assertIn("sizes", names)
        self.assertNotIn("concat", names)
        self.assertNotIn("shape", names)

    def test_onxx_prune_is_deletion_only_and_does_not_mutate_source(self):
        original = Model(graph())
        fake_onnx = types.SimpleNamespace(
            ModelProto=Model,
            checker=types.SimpleNamespace(check_model=lambda x: None))
        with mock.patch.dict(sys.modules, {"onnx": fake_onnx}):
            pruned, summary = module.prune_dead(original)
        self.assertEqual(summary["nodes_before"], 5)
        self.assertEqual(summary["nodes_after"], 3)
        self.assertEqual(summary["nodes_removed"], 2)
        self.assertEqual(summary["removed_operators"], {"Concat": 1, "Shape": 1})
        self.assertEqual([n.op_type for n in pruned.graph.node],
                         ["Conv", "Resize", "Conv"])
        self.assertEqual([n.op_type for n in original.graph.node],
                         ["Conv", "Shape", "Concat", "Resize", "Conv"])
        self.assertEqual([x.name for x in pruned.graph.initializer], ["sizes"])
        self.assertEqual([x.name for x in pruned.graph.value_info],
                         ["conv", "up"])

    def test_missing_live_tensor_is_rejected(self):
        g = graph()
        g.node[-1].input[0] = "unknown_input"
        with self.assertRaisesRegex(ValueError, "Unresolved live ONNX tensor"):
            module.reachable_nodes(g)

    def test_duplicate_producer_is_rejected(self):
        g = graph()
        g.node[1].output[0] = "conv"
        with self.assertRaisesRegex(ValueError, "Duplicate ONNX tensor producer"):
            module.reachable_nodes(g)

    def test_nested_control_flow_rejected(self):
        g = graph()
        g.node[0].attribute = [types.SimpleNamespace(type=5)]
        with self.assertRaisesRegex(ValueError, "Control-flow"):
            module.reachable_nodes(g)

    def test_no_unreachable_nodes_is_not_a_compatibility_experiment(self):
        g = graph()
        g.node = [g.node[0], g.node[3], g.node[4]]
        original = Model(g)
        fake_onnx = types.SimpleNamespace(
            ModelProto=Model,
            checker=types.SimpleNamespace(check_model=lambda x: None))
        with mock.patch.dict(sys.modules, {"onnx": fake_onnx}):
            with self.assertRaisesRegex(ValueError, "No unreachable nodes"):
                module.prune_dead(original)


if __name__ == "__main__":
    unittest.main()
