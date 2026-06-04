# mypy: allow-untyped-defs
from __future__ import annotations

import operator
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from torch._subclasses import FakeTensor
from torch.export.graph_signature import OutputKind
from torch.fx import GraphModule, Node

_REPO_ROOT = Path(__file__).resolve().parents[4]
_VENDORED_TORCHAO = _REPO_ROOT / "executorch" / "third-party" / "ao"
if _VENDORED_TORCHAO.is_dir() and str(_VENDORED_TORCHAO) not in sys.path:
    sys.path.insert(0, str(_VENDORED_TORCHAO))

from torchao.quantization.pt2e import HistogramObserver, MinMaxObserver
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    QuantizationConfig,
    QuantizationSpec,
    Quantizer,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY
from torchao.quantization.pt2e.utils import _fuse_conv_bn_


__all__ = [
    "XNNPACKJointTrainingQuantizer",
    "annotate_joint_backward_qdq_edges",
    "get_affine_activation_qdq_config",
    "get_symmetric_activation_qdq_config",
    "get_symmetric_gradient_qdq_config",
    "get_symmetric_weight_qdq_config",
    "insert_joint_qdq_ste_masks",
]


_QUANTIZE_PER_TENSOR = torch.ops.quantized_decomposed.quantize_per_tensor.default
_DEQUANTIZE_PER_TENSOR = torch.ops.quantized_decomposed.dequantize_per_tensor.default

_SKIP_PLACEHOLDER_TOKENS = (
    "label",
    "labels",
    "bbox",
    "bboxes",
    "target",
    "targets",
    "mask",
    "stride",
    "anchor",
)

_VIEW_LIKE_TARGETS = {
    torch.ops.aten.permute.default,
    torch.ops.aten.permute_copy.default,
    torch.ops.aten.view.default,
    torch.ops.aten.view_copy.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze_copy.dim,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.slice_copy.Tensor,
    torch.ops.aten.flatten.using_ints,
}

_METADATA_ONLY_FACTORY_TARGETS = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.full_like.default,
    torch.ops.aten.new_empty.default,
    torch.ops.aten.new_empty_strided.default,
    torch.ops.aten.new_empty_strided.out,
    torch.ops.aten.new_full.default,
    torch.ops.aten.new_ones.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.ones_like.default,
    torch.ops.aten.zeros_like.default,
}

_CONV_TARGETS = {
    torch.ops.aten.conv1d.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.conv3d.default,
    torch.ops.aten.convolution.default,
}

_SILU_TARGETS = {
    torch.ops.aten.silu.default,
}


def _fake_tensor(node: Node) -> FakeTensor | None:
    val = node.meta.get("val")
    return val if isinstance(val, FakeTensor) else None


def _is_float_tensor_node(node: object, *, include_scalar: bool = False) -> bool:
    if not isinstance(node, Node):
        return False
    val = _fake_tensor(node)
    if val is None or not torch.is_floating_point(val):
        return False
    if not include_scalar and len(val.shape) == 0:
        return False
    return True


def _is_quantizable_placeholder(node: Node) -> bool:
    if node.op != "placeholder" or not _is_float_tensor_node(node):
        return False
    name = str(node.target if node.target is not None else node.name).lower()
    if "bias" in name:
        return False
    if name.startswith("p_") or name.startswith("b_"):
        return True
    if any(tok in name for tok in _SKIP_PLACEHOLDER_TOKENS):
        return False
    return name in {"x", "input", "inputs", "image", "images"}


def _is_user_activation_placeholder(node: Node) -> bool:
    if node.op != "placeholder" or not _is_quantizable_placeholder(node):
        return False
    name = str(node.target if node.target is not None else node.name).lower()
    return not (name.startswith("p_") or name.startswith("b_"))


def _is_view_like_node(node: Node) -> bool:
    return node.op == "call_function" and node.target in _VIEW_LIKE_TARGETS


def _origin_node(node: Node) -> Node:
    cur = node
    seen: set[Node] = set()
    while _is_view_like_node(cur) and cur.args and isinstance(cur.args[0], Node):
        if cur in seen:
            break
        seen.add(cur)
        cur = cur.args[0]
    return cur


def _is_parameter_weight_node(node: Node) -> bool:
    if not _is_float_tensor_node(node):
        return False
    origin = _origin_node(node)
    if origin.op != "placeholder":
        return False
    name = str(origin.target if origin.target is not None else origin.name).lower()
    return (name.startswith("p_") or name.startswith("b_")) and "weight" in name


def _is_quantizable_input_node(node: Node) -> bool:
    if not _is_float_tensor_node(node):
        return False
    if node.op == "placeholder":
        return _is_quantizable_placeholder(node)
    return node.op in {"call_function", "call_method", "get_attr"}


def _is_compute_consumer(node: Node) -> bool:
    if node.op not in {"call_function", "call_method"}:
        return False
    if _is_view_like_node(node):
        return False
    return node.target is not operator.getitem


def _is_metadata_only_factory_node(node: Node) -> bool:
    return node.op == "call_function" and node.target in _METADATA_ONLY_FACTORY_TARGETS


def _iter_node_args(args) -> Iterable[Node]:
    if isinstance(args, Node):
        yield args
    elif isinstance(args, (tuple, list)):
        for arg in args:
            yield from _iter_node_args(arg)
    elif isinstance(args, dict):
        for arg in args.values():
            yield from _iter_node_args(arg)


def _merge_quantization_annotation(
    node: Node,
    *,
    input_qspec_map: dict[Node, QuantizationSpec] | None = None,
    output_qspec: QuantizationSpec | None = None,
    allow_implicit_sharing: bool = False,
) -> bool:
    existing = node.meta.get(Q_ANNOTATION_KEY)
    if isinstance(existing, QuantizationAnnotation):
        changed = False
        for input_node, qspec in (input_qspec_map or {}).items():
            if input_node not in existing.input_qspec_map:
                existing.input_qspec_map[input_node] = qspec
                changed = True
        if output_qspec is not None and existing.output_qspec is None:
            existing.output_qspec = output_qspec
            changed = True
        existing.allow_implicit_sharing = (
            existing.allow_implicit_sharing and allow_implicit_sharing
        )
        existing._annotated = True
        return changed

    node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map or {},
        output_qspec=output_qspec,
        allow_implicit_sharing=allow_implicit_sharing,
        _annotated=True,
    )
    return bool(input_qspec_map) or output_qspec is not None


def _make_quantization_config(
    act_quantization_spec: QuantizationSpec,
    weight_quantization_spec: QuantizationSpec,
) -> QuantizationConfig:
    return QuantizationConfig(
        act_quantization_spec,
        act_quantization_spec,
        weight_quantization_spec,
        None,
        False,
    )


def get_affine_activation_qdq_config(
    *,
    act_qmin: int = -128,
    act_qmax: int = 127,
) -> QuantizationConfig:
    act_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=act_qmin,
        quant_max=act_qmax,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=HistogramObserver.with_args(eps=2**-12),
    )
    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=2**-12),
    )
    return _make_quantization_config(act_quantization_spec, weight_quantization_spec)


def get_symmetric_activation_qdq_config() -> QuantizationConfig:
    act_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=2**-12),
    )
    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=2**-12),
    )
    return _make_quantization_config(act_quantization_spec, weight_quantization_spec)


def get_symmetric_gradient_qdq_config() -> QuantizationConfig:
    grad_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=2**-12),
    )
    return _make_quantization_config(grad_quantization_spec, grad_quantization_spec)


def get_symmetric_weight_qdq_config() -> QuantizationConfig:
    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=2**-12),
    )
    return _make_quantization_config(weight_quantization_spec, weight_quantization_spec)


def _is_conv_node(node: Node) -> bool:
    return node.op == "call_function" and node.target in _CONV_TARGETS


def _is_sigmoid_of(node: object, input_node: Node) -> bool:
    return (
        isinstance(node, Node)
        and node.op == "call_function"
        and node.target == torch.ops.aten.sigmoid.default
        and len(node.args) >= 1
        and node.args[0] is input_node
    )


def _is_silu_node(node: Node) -> bool:
    return node.op == "call_function" and node.target in _SILU_TARGETS


def _decomposed_silu_input(node: Node) -> Node | None:
    if (
        node.op != "call_function"
        or node.target != torch.ops.aten.mul.Tensor
        or len(node.args) < 2
    ):
        return None
    lhs, rhs = node.args[:2]
    if isinstance(lhs, Node) and _is_sigmoid_of(rhs, lhs):
        return lhs
    if isinstance(rhs, Node) and _is_sigmoid_of(lhs, rhs):
        return rhs
    return None


def _find_silu_qdq_nodes(gm: GraphModule) -> tuple[set[Node], set[Node], set[Node]]:
    pre_silu_sources: set[Node] = set()
    silu_outputs: set[Node] = set()
    silu_internal_sigmoids: set[Node] = set()
    for node in gm.graph.nodes:
        if _is_silu_node(node) and node.args and isinstance(node.args[0], Node):
            pre_silu_sources.add(node.args[0])
            silu_outputs.add(node)
            continue

        source = _decomposed_silu_input(node)
        if source is None:
            continue
        pre_silu_sources.add(source)
        silu_outputs.add(node)
        for arg in node.args[:2]:
            if _is_sigmoid_of(arg, source):
                silu_internal_sigmoids.add(arg)
    return pre_silu_sources, silu_outputs, silu_internal_sigmoids


def _is_decomposed_silu_internal_edge(input_node: Node, consumer_node: Node) -> bool:
    source = _decomposed_silu_input(consumer_node)
    return source is not None and _is_sigmoid_of(input_node, source)


class XNNPACKJointTrainingQuantizer(Quantizer):
    """Annotates exported joint forward/backward graphs for XNNPACK Q/DQ training."""

    def __init__(
        self,
        *,
        activation_config: QuantizationConfig | None = None,
        pre_silu_activation_config: QuantizationConfig | None = None,
        gradient_config: QuantizationConfig | None = None,
        weight_config: QuantizationConfig | None = None,
        quantize_final_outputs: bool = False,
    ) -> None:
        super().__init__()
        self.activation_config = activation_config or get_affine_activation_qdq_config()
        self.pre_silu_activation_config = (
            pre_silu_activation_config or get_symmetric_activation_qdq_config()
        )
        self.gradient_config = gradient_config or get_symmetric_gradient_qdq_config()
        self.weight_config = weight_config or get_symmetric_weight_qdq_config()
        self.quantize_final_outputs = quantize_final_outputs
        self.loss_output_names: set[str] = set()
        self.final_output_names: set[str] = set()
        self.annotation_report: dict[str, Any] = {}

    def configure_from_exported_program(self, exported_program) -> None:
        self.loss_output_names = {
            spec.arg.name
            for spec in exported_program.graph_signature.output_specs
            if spec.kind == OutputKind.LOSS_OUTPUT and hasattr(spec.arg, "name")
        }
        self.final_output_names = {
            spec.arg.name
            for spec in exported_program.graph_signature.output_specs
            if hasattr(spec.arg, "name")
        }

    def transform_for_annotation(self, model: GraphModule) -> GraphModule:
        _fuse_conv_bn_(model)
        return model

    def annotate(self, model: GraphModule) -> GraphModule:
        self.annotation_report = annotate_joint_backward_qdq_edges(
            model,
            activation_qspec=self.activation_config.input_activation,
            pre_silu_activation_qspec=self.pre_silu_activation_config.input_activation,
            gradient_qspec=self.gradient_config.input_activation,
            weight_qspec=self.weight_config.weight,
            loss_output_names=self.loss_output_names,
            final_output_names=self.final_output_names
            if self.quantize_final_outputs
            else set(),
        )
        return model

    def validate(self, model: GraphModule) -> None:
        pass

    def prepare_obs_or_fq_callback(self, model, edge_or_node_to_obs_or_fq) -> None:
        pass


def annotate_joint_backward_qdq_edges(
    gm: GraphModule,
    *,
    activation_qspec: QuantizationSpec,
    pre_silu_activation_qspec: QuantizationSpec,
    gradient_qspec: QuantizationSpec,
    weight_qspec: QuantizationSpec,
    loss_output_names: set[str],
    final_output_names: set[str],
) -> dict[str, Any]:
    pre_silu_sources, silu_outputs, silu_internal_sigmoids = _find_silu_qdq_nodes(gm)
    producer_output_qspec_nodes = pre_silu_sources | silu_outputs

    input_edges = 0
    output_nodes = 0
    annotated_nodes = 0
    skipped_view_like_consumers = 0
    in_backward = False
    by_phase = {"forward": 0, "backward": 0}
    forward_value_nodes: set[Node] = set()

    for node in gm.graph.nodes:
        if node.op not in {"call_function", "call_method"}:
            continue
        if node.name in loss_output_names:
            in_backward = True
        phase = "backward" if in_backward else "forward"

        output_qspec = None
        if node in pre_silu_sources and _is_float_tensor_node(node):
            output_qspec = pre_silu_activation_qspec
        elif node in silu_outputs and _is_float_tensor_node(node):
            output_qspec = activation_qspec
        elif node.name in final_output_names and _is_float_tensor_node(node):
            output_qspec = gradient_qspec if in_backward else activation_qspec

        if not _is_compute_consumer(node) and output_qspec is None:
            skipped_view_like_consumers += int(_is_view_like_node(node))
            if not in_backward:
                forward_value_nodes.add(node)
            continue

        input_qspec_map: dict[Node, QuantizationSpec] = {}
        if _is_compute_consumer(node):
            for input_node in _iter_node_args(node.args):
                if _is_metadata_only_factory_node(node):
                    continue
                if not _is_quantizable_input_node(input_node):
                    continue
                origin = _origin_node(input_node)
                if origin in producer_output_qspec_nodes:
                    continue
                if input_node in silu_internal_sigmoids:
                    continue
                if _is_decomposed_silu_internal_edge(input_node, node):
                    continue
                if _is_parameter_weight_node(input_node):
                    input_qspec_map[input_node] = weight_qspec
                elif in_backward and (
                    origin in forward_value_nodes
                    or _is_user_activation_placeholder(origin)
                ):
                    input_qspec_map[input_node] = activation_qspec
                else:
                    input_qspec_map[input_node] = (
                        gradient_qspec if in_backward else activation_qspec
                    )

        if not input_qspec_map and output_qspec is None:
            continue

        if _merge_quantization_annotation(
            node,
            input_qspec_map=input_qspec_map,
            output_qspec=output_qspec,
            allow_implicit_sharing=False,
        ):
            annotated_nodes += 1
            input_edges += len(input_qspec_map)
            output_nodes += int(output_qspec is not None)
            by_phase[phase] += 1
        if not in_backward:
            forward_value_nodes.add(node)

    return {
        "joint_qdq_annotated_nodes": annotated_nodes,
        "joint_qdq_input_edges": input_edges,
        "joint_qdq_output_nodes": output_nodes,
        "joint_qdq_forward_nodes": by_phase["forward"],
        "joint_qdq_backward_nodes": by_phase["backward"],
        "joint_qdq_skipped_view_like_consumers": skipped_view_like_consumers,
        "joint_qdq_pre_silu_sources": len(pre_silu_sources),
        "joint_qdq_silu_outputs": len(silu_outputs),
        "joint_qdq_silu_internal_sigmoids": len(silu_internal_sigmoids),
    }


def _is_quantize_per_tensor_node(node: object) -> bool:
    return (
        isinstance(node, Node)
        and node.op == "call_function"
        and node.target == _QUANTIZE_PER_TENSOR
    )


def _is_dequantize_per_tensor_node(node: object) -> bool:
    return (
        isinstance(node, Node)
        and node.op == "call_function"
        and node.target == _DEQUANTIZE_PER_TENSOR
    )


def _quantize_source(node: Node) -> Node | None:
    if _is_quantize_per_tensor_node(node) and node.args and isinstance(node.args[0], Node):
        return node.args[0]
    return None


def _dequantize_quantize_node(node: object) -> Node | None:
    if not _is_dequantize_per_tensor_node(node) or not isinstance(node, Node):
        return None
    quantize_node = node.args[0] if node.args else None
    return quantize_node if _is_quantize_per_tensor_node(quantize_node) else None


def _unwrap_qdq_source(node: Node) -> Node:
    quantize_node = _dequantize_quantize_node(node)
    if quantize_node is None:
        return node
    source = _quantize_source(quantize_node)
    return source if source is not None else node


def _direct_quantize_users(node: Node) -> list[Node]:
    return [user for user in node.users if _is_quantize_per_tensor_node(user)]


def _is_symmetric_qdq_quantize(node: Node) -> bool:
    return (
        _is_quantize_per_tensor_node(node)
        and len(node.args) >= 5
        and node.args[2] == 0
        and node.args[3] == -127
        and node.args[4] == 127
    )


def _is_affine_qdq_quantize(node: Node) -> bool:
    return (
        _is_quantize_per_tensor_node(node)
        and len(node.args) >= 5
        and node.args[3] == -128
        and node.args[4] == 127
    )


def _is_dequantize_of_quantize(node: object, quantize_node: Node) -> bool:
    return _dequantize_quantize_node(node) is quantize_node


def _is_sigmoid_of_dequantize_from(node: object, quantize_node: Node) -> bool:
    return (
        isinstance(node, Node)
        and node.op == "call_function"
        and node.target == torch.ops.aten.sigmoid.default
        and len(node.args) >= 1
        and _is_dequantize_of_quantize(node.args[0], quantize_node)
    )


def _is_converted_decomposed_silu_node(node: Node, quantize_node: Node) -> bool:
    if (
        node.op != "call_function"
        or node.target != torch.ops.aten.mul.Tensor
        or len(node.args) < 2
    ):
        return False
    lhs, rhs = node.args[:2]
    return (
        _is_dequantize_of_quantize(lhs, quantize_node)
        and _is_sigmoid_of_dequantize_from(rhs, quantize_node)
    ) or (
        _is_dequantize_of_quantize(rhs, quantize_node)
        and _is_sigmoid_of_dequantize_from(lhs, quantize_node)
    )


def _contains_sigmoid_of_dequantize_from(
    node: object,
    quantize_node: Node,
    *,
    max_depth: int = 8,
    seen: set[Node] | None = None,
) -> bool:
    if not isinstance(node, Node) or max_depth < 0:
        return False
    if _is_sigmoid_of_dequantize_from(node, quantize_node):
        return True
    seen = seen or set()
    if node in seen:
        return False
    seen.add(node)
    return any(
        _contains_sigmoid_of_dequantize_from(
            arg, quantize_node, max_depth=max_depth - 1, seen=seen
        )
        for arg in _iter_node_args(node.args)
    )


def _is_silu_derivative_arg(node: Node, quantize_node: Node) -> bool:
    source = _unwrap_qdq_source(node)
    return (
        source.op == "call_function"
        and source.target == torch.ops.aten.mul.Tensor
        and _contains_sigmoid_of_dequantize_from(source, quantize_node)
    )


def _ste_bounds_from_quantize_node(quantize_node: Node) -> tuple[float, float]:
    if len(quantize_node.args) < 5:
        raise RuntimeError(f"unexpected quantize_per_tensor args: {quantize_node.args}")
    scale = float(quantize_node.args[1])
    zero_point = int(quantize_node.args[2])
    quant_min = int(quantize_node.args[3])
    quant_max = int(quantize_node.args[4])
    return (quant_min - zero_point) * scale, (quant_max - zero_point) * scale


def _insert_ste_mask(
    gm: GraphModule,
    *,
    source_node: Node,
    quantize_node: Node,
    insert_before: Node,
) -> Node:
    lower, upper = _ste_bounds_from_quantize_node(quantize_node)
    with gm.graph.inserting_before(insert_before):
        ge = gm.graph.call_function(torch.ops.aten.ge.Scalar, args=(source_node, lower))
        le = gm.graph.call_function(torch.ops.aten.le.Scalar, args=(source_node, upper))
        mask_bool = gm.graph.call_function(
            torch.ops.aten.logical_and.default, args=(ge, le)
        )
        mask = gm.graph.call_function(
            torch.ops.aten._to_copy.default,
            args=(mask_bool,),
            kwargs={"dtype": torch.float32},
        )
    return mask


def _find_converted_silu_qdq(gm: GraphModule) -> list[tuple[Node, Node, Node, Node]]:
    matches: list[tuple[Node, Node, Node, Node]] = []
    for node in gm.graph.nodes:
        if not _is_quantize_per_tensor_node(node) or not _is_symmetric_qdq_quantize(node):
            continue
        qsym_quantize = node
        pre_silu_source = _quantize_source(qsym_quantize)
        if pre_silu_source is None:
            continue
        silu_nodes = [
            candidate
            for candidate in gm.graph.nodes
            if _is_converted_decomposed_silu_node(candidate, qsym_quantize)
        ]
        for silu_node in silu_nodes:
            qasym_candidates = [
                user
                for user in _direct_quantize_users(silu_node)
                if _is_affine_qdq_quantize(user)
            ]
            for qasym_quantize in qasym_candidates:
                matches.append(
                    (pre_silu_source, qsym_quantize, silu_node, qasym_quantize)
                )
    return matches


def _find_silu_backward_grads(gm: GraphModule, qsym_quantize: Node) -> list[tuple[Node, Node]]:
    candidates: list[tuple[Node, Node]] = []
    for node in gm.graph.nodes:
        if (
            node.op != "call_function"
            or node.target != torch.ops.aten.mul.Tensor
            or len(node.args) < 2
        ):
            continue
        node_args = [arg for arg in node.args[:2] if isinstance(arg, Node)]
        for derivative_arg in node_args:
            if not _is_silu_derivative_arg(derivative_arg, qsym_quantize):
                continue
            for gradient_input in node_args:
                if gradient_input is derivative_arg:
                    continue
                candidates.append((node, gradient_input))
    return candidates


def _replace_node_arg(node: Node, old_arg: Node, new_arg: Node) -> bool:
    changed = False
    new_args = []
    for arg in node.args:
        if arg is old_arg:
            new_args.append(new_arg)
            changed = True
        else:
            new_args.append(arg)
    if changed:
        node.args = tuple(new_args)
    return changed


def insert_joint_qdq_ste_masks(gm: GraphModule) -> dict[str, int]:
    qasym_masks = 0
    qsym_masks = 0
    matched_backward = 0
    matches = _find_converted_silu_qdq(gm)
    if not matches:
        return {
            "joint_ste_qasym_masks": 0,
            "joint_ste_qsym_masks": 0,
            "joint_ste_conv_silu_matches": 0,
        }

    for pre_silu_source, qsym_quantize, silu_node, qasym_quantize in matches:
        qsym_source = _quantize_source(qsym_quantize)
        qasym_source = _quantize_source(qasym_quantize)
        if qsym_source is None or qasym_source is None:
            continue
        if qsym_source is not pre_silu_source or qasym_source is not silu_node:
            continue

        silu_grad_matches = _find_silu_backward_grads(gm, qsym_quantize)
        if not silu_grad_matches:
            continue

        for silu_grad, gradient_input in silu_grad_matches:
            silu_grad_quantize_users = [
                user
                for user in _direct_quantize_users(silu_grad)
                if _is_symmetric_qdq_quantize(user)
            ]
            if not silu_grad_quantize_users:
                continue

            qasym_mask = _insert_ste_mask(
                gm,
                source_node=qasym_source,
                quantize_node=qasym_quantize,
                insert_before=silu_grad,
            )
            with gm.graph.inserting_before(silu_grad):
                masked_gradient = gm.graph.call_function(
                    torch.ops.aten.mul.Tensor, args=(gradient_input, qasym_mask)
                )
            if _replace_node_arg(silu_grad, gradient_input, masked_gradient):
                qasym_masks += 1

            qsym_mask_insert_before = silu_grad_quantize_users[0]
            qsym_mask = _insert_ste_mask(
                gm,
                source_node=qsym_source,
                quantize_node=qsym_quantize,
                insert_before=qsym_mask_insert_before,
            )
            with gm.graph.inserting_before(qsym_mask_insert_before):
                masked_silu_grad = gm.graph.call_function(
                    torch.ops.aten.mul.Tensor, args=(silu_grad, qsym_mask)
                )
            for quantize_user in silu_grad_quantize_users:
                quantize_user.args = (masked_silu_grad, *quantize_user.args[1:])
            qsym_masks += 1
            matched_backward += 1

    gm.graph.eliminate_dead_code()
    gm.recompile()
    return {
        "joint_ste_qasym_masks": qasym_masks,
        "joint_ste_qsym_masks": qsym_masks,
        "joint_ste_conv_silu_matches": matched_backward,
    }
