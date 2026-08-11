# mypy: allow-untyped-defs
from __future__ import annotations

import operator
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch._subclasses import FakeTensor
from torch.export.graph_signature import OutputKind
from torch.fx import GraphModule, Node

_REPO_ROOT = Path(__file__).resolve().parents[4]
_VENDORED_TORCHAO = _REPO_ROOT / "executorch" / "third-party" / "ao"
if _VENDORED_TORCHAO.is_dir() and str(_VENDORED_TORCHAO) not in sys.path:
    sys.path.insert(0, str(_VENDORED_TORCHAO))

from torchao.quantization.pt2e import (
    HistogramObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
)
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    QuantizationConfig,
    QuantizationSpec,
    Quantizer,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY
from torchao.quantization.pt2e.utils import _fuse_conv_bn_
from xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config


__all__ = [
    "XNNPACKJointTrainingQuantizer",
    "annotate_joint_backward_qdq_edges",
    "get_affine_activation_qdq_config",
    "get_affine_activation_int16_qdq_config",
    "get_symmetric_activation_qdq_config",
    "get_symmetric_activation_int16_qdq_config",
    "get_fp16_activation_qdq_config",
    "get_symmetric_gradient_qdq_config",
    "get_symmetric_gradient_int16_qdq_config",
    "get_symmetric_weight_qdq_config",
    "insert_joint_qdq_ste_masks",
]


_QUANTIZE_PER_TENSOR = torch.ops.quantized_decomposed.quantize_per_tensor.default
_DEQUANTIZE_PER_TENSOR = torch.ops.quantized_decomposed.dequantize_per_tensor.default

_BACKWARD_QUANTIZATION_MODES = {"all", "none", "conv_silu", "annotation_rule"}

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
    torch.ops.aten.split.Tensor,
    torch.ops.aten.split_with_sizes.default,
    torch.ops.aten.split_with_sizes_copy.default,
    torch.ops.aten.unbind.int,
    torch.ops.aten.chunk.default,
}

_LOSS_PATH_TEXT_TOKENS = (
    "label",
    "labels",
    "bbox",
    "bboxes",
    "target",
    "targets",
    "mask",
    "masks",
    "assign",
    "batch_idx",
    "gt_",
)

_LOSS_PATH_TARGETS = {
    torch.ops.aten.where.self,
    torch.ops.aten.scatter.value,
    torch.ops.aten.index_put.default,
}

_ATTENTION_TEXT_TOKENS = (
    "_attn_",
    ".attn.",
    "attention",
    "qkv",
)

_ATTENTION_TARGETS = {
    torch.ops.aten.bmm.default,
    torch.ops.aten.matmul.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.softmax.int,
}

_YOLO_DETECT_INDEX_RE = re.compile(
    r"^[pcb]_model_model_(\d+)_(?:cv2|cv3|dfl|m_|anchors|strides)(?:_|$)"
)
_YOLO_ATTENTION_INDEX_RE = re.compile(
    r"^[pb]_model_model_(\d+)_.*(?:attn|qkv)"
)
_YOLO_MODEL_PARAM_RE = re.compile(r"^[pb]_model_model_\d+_")
_NAMED_MODEL_PARAM_RE = re.compile(
    r"^[pb]_(?:backbone|neck|detect|detect_head|head|model)_"
)

_METADATA_ONLY_FACTORY_TARGETS = {
    torch.ops.aten._assert_tensor_metadata.default,
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

_ANNOTATION_RULE_BOUNDARY_CONSUMER_TARGETS = {
    torch.ops.aten.convolution_backward.default,
    torch.ops.aten.add.Tensor,
    torch.ops.aten.add.Scalar,
    torch.ops.aten.cat.default,
    torch.ops.aten.mul.Tensor,
}


@dataclass(frozen=True)
class JointBackwardAnnotationOptions:
    """Model-family knobs for the annotation_rule backward pass.

    Defaults reproduce the historical module constants exactly, so existing
    (YOLO) exports are byte-identical when no options are supplied. A family
    whose loss region legitimately uses ops like ``where.self``/``scatter.value``
    (e.g. a decomposed cross-entropy) overrides ``loss_path_targets`` /
    ``loss_path_text_tokens`` so those gradients are not skipped, and extends
    ``boundary_consumer_targets`` with its backward consumers.
    """

    loss_path_targets: frozenset = field(
        default_factory=lambda: frozenset(_LOSS_PATH_TARGETS)
    )
    loss_path_text_tokens: tuple[str, ...] = _LOSS_PATH_TEXT_TOKENS
    boundary_consumer_targets: frozenset = field(
        default_factory=lambda: frozenset(_ANNOTATION_RULE_BOUNDARY_CONSUMER_TARGETS)
    )


def _resolve_annotation_options(
    options: JointBackwardAnnotationOptions | None,
) -> JointBackwardAnnotationOptions:
    return options if options is not None else _DEFAULT_ANNOTATION_OPTIONS


_DEFAULT_ANNOTATION_OPTIONS = JointBackwardAnnotationOptions()

_LAYERWISE_EDGE_KINDS = {
    "forward_activation",
    "forward_pre_silu",
    "forward_silu_output",
    "backward_saved_activation",
    "backward_silu_input",
    "backward_gradient",
    "backward_conv_input",
    "final_forward_output",
    "final_backward_output",
}

_LAYERWISE_FORWARD_EDGE_KINDS = {
    "forward_activation",
    "forward_pre_silu",
    "forward_silu_output",
    "final_forward_output",
}

_LAYERWISE_FORWARD_FORMATS = {"int8", "int16"}
_LAYERWISE_BACKWARD_FORMATS = {"int8", "int16", "fp16"}

_LAYERWISE_AFFINE_ACTIVATION_EDGE_KINDS = {
    "forward_activation",
    "forward_pre_silu",
    "forward_silu_output",
    "backward_saved_activation",
    "backward_silu_input",
    "final_forward_output",
}

_LAYERWISE_GRADIENT_EDGE_KINDS = {
    "backward_gradient",
    "backward_conv_input",
    "final_backward_output",
}


def _as_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (tuple, list)):
        return [str(item) for item in value]
    return [str(value)]


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
    if node.op != "call_function":
        return False
    if node.target in _VIEW_LIKE_TARGETS:
        return True
    return (
        node.target is operator.getitem
        and node.args
        and isinstance(node.args[0], Node)
        and _is_view_like_node(node.args[0])
    )


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


def _existing_output_qspec(node: Node):
    """The output qspec a previous annotation pass already put on ``node``."""

    annotation = node.meta.get(Q_ANNOTATION_KEY)
    return getattr(annotation, "output_qspec", None) if annotation is not None else None


def _concrete_qspec(qspec, depth: int = 8):
    """Follow ``SharedQuantizationSpec`` links to the spec that carries dtypes.

    A shared spec names another edge or node instead of describing a wire
    format, so it has no ``dtype`` of its own; comparing it directly makes every
    field read ``None``.
    """

    for _ in range(depth):
        edge_or_node = getattr(qspec, "edge_or_node", None)
        if edge_or_node is None:
            return qspec
        if isinstance(edge_or_node, tuple):
            input_node, owner = edge_or_node
            annotation = owner.meta.get(Q_ANNOTATION_KEY)
            qspec = (getattr(annotation, "input_qspec_map", None) or {}).get(input_node)
        else:
            annotation = edge_or_node.meta.get(Q_ANNOTATION_KEY)
            qspec = getattr(annotation, "output_qspec", None)
        if qspec is None:
            return None
    return None


def _qspec_interchangeable(produced, consumed) -> bool:
    """Can a consumer read ``produced`` instead of observing its own copy?

    Only the wire format has to agree -- dtype and range.  Observer identity and
    qscheme details differ freely between a symmetric weight-side config and an
    affine activation config that still exchange the same int8 codes.
    """

    produced = _concrete_qspec(produced)
    consumed = _concrete_qspec(consumed)
    if produced is None or consumed is None:
        return False
    return all(
        getattr(produced, field, None) == getattr(consumed, field, None)
        for field in ("dtype", "quant_min", "quant_max", "qscheme")
    )


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


def get_affine_activation_int16_qdq_config() -> QuantizationConfig:
    act_quantization_spec = QuantizationSpec(
        dtype=torch.int16,
        quant_min=-32768,
        quant_max=32767,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=1e-12),
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


def get_symmetric_activation_int16_qdq_config() -> QuantizationConfig:
    act_quantization_spec = QuantizationSpec(
        dtype=torch.int16,
        quant_min=-32767,
        quant_max=32767,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=1e-12),
    )
    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=2**-12),
    )
    return _make_quantization_config(act_quantization_spec, weight_quantization_spec)


def get_fp16_activation_qdq_config() -> QuantizationConfig:
    act_quantization_spec = QuantizationSpec(
        dtype=torch.float16,
        observer_or_fake_quant_ctr=PlaceholderObserver.with_args(dtype=torch.float16),
    )
    return _make_quantization_config(act_quantization_spec, act_quantization_spec)


def get_symmetric_gradient_qdq_config() -> QuantizationConfig:
    grad_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=2**-12),
    )
    return _make_quantization_config(grad_quantization_spec, grad_quantization_spec)


def get_symmetric_gradient_int16_qdq_config() -> QuantizationConfig:
    grad_quantization_spec = QuantizationSpec(
        dtype=torch.int16,
        quant_min=-32767,
        quant_max=32767,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=1e-12),
    )
    return _make_quantization_config(grad_quantization_spec, grad_quantization_spec)


def get_symmetric_weight_qdq_config(*, per_channel: bool = True) -> QuantizationConfig:
    """Weight qspec for trainable conv weights.

    Must stay byte-equivalent to the vendored per-channel branch in
    ``xnnpack_quantizer.get_symmetric_quantization_config`` (qscheme, ch_axis,
    observer, eps, range). The forward and backward phases annotate the *same*
    weight tensor through two independent observers -- forward via
    ``XNNPACKQuantizer`` + ``forward_quantization_config``, backward via this
    config, because ``annotate_forward_compute_edges`` is False whenever the
    XNNPACK forward quantizer is in use. If the two disagree,
    ``_collect_pt2e_weight_qparams`` raises "conflicting PT2E weight qparams".

    ``per_channel=False`` is kept for A/B measurement and legacy reproduction
    only; per-output-channel axis 0 is the production configuration.
    """

    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=(
            torch.per_channel_symmetric if per_channel else torch.per_tensor_symmetric
        ),
        ch_axis=0,
        is_dynamic=False,
        observer_or_fake_quant_ctr=(
            PerChannelMinMaxObserver if per_channel else MinMaxObserver
        ).with_args(eps=2**-12),
    )
    # _make_quantization_config also places the weight spec in the activation
    # slot; only `.weight` is ever read from this config, so that is inert.
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


def _node_contains_target(
    node: object,
    target: object,
    *,
    max_depth: int = 8,
    seen: set[Node] | None = None,
) -> bool:
    if not isinstance(node, Node) or max_depth < 0:
        return False
    if node.op == "call_function" and node.target == target:
        return True
    seen = seen or set()
    if node in seen:
        return False
    seen.add(node)
    return any(
        _node_contains_target(arg, target, max_depth=max_depth - 1, seen=seen)
        for arg in _iter_node_args(node.args)
    )


def _is_plain_silu_derivative(node: object) -> bool:
    return (
        isinstance(node, Node)
        and node.op == "call_function"
        and node.target == torch.ops.aten.mul.Tensor
        and _node_contains_target(node, torch.ops.aten.sigmoid.default)
        and _node_contains_target(node, torch.ops.aten.add.Scalar)
        and _node_contains_target(node, torch.ops.aten.sub.Tensor)
    )


def _find_plain_silu_conv_backward_edges(gm: GraphModule) -> list[tuple[Node, Node, Node]]:
    edges: list[tuple[Node, Node, Node]] = []
    for conv_backward in gm.graph.nodes:
        if (
            conv_backward.op != "call_function"
            or conv_backward.target != torch.ops.aten.convolution_backward.default
            or not conv_backward.args
        ):
            continue
        silu_grad = conv_backward.args[0]
        if (
            not isinstance(silu_grad, Node)
            or silu_grad.op != "call_function"
            or silu_grad.target != torch.ops.aten.mul.Tensor
            or len(silu_grad.args) < 2
        ):
            continue
        node_args = [arg for arg in silu_grad.args[:2] if isinstance(arg, Node)]
        if len(node_args) != 2:
            continue
        lhs, rhs = node_args
        if _is_plain_silu_derivative(lhs):
            edges.append((silu_grad, rhs, conv_backward))
        elif _is_plain_silu_derivative(rhs):
            edges.append((silu_grad, lhs, conv_backward))
    return edges


def _forward_node_names(gm: GraphModule, loss_output_names: set[str]) -> set[str]:
    names: set[str] = set()
    in_backward = False
    for node in gm.graph.nodes:
        if node.name in loss_output_names:
            in_backward = True
        if not in_backward:
            names.add(node.name)
    return names


def _model_forward_node_names(
    gm: GraphModule, user_output_names: set[str]
) -> set[str]:
    by_name = {node.name: node for node in gm.graph.nodes}
    stack = [by_name[name] for name in user_output_names if name in by_name]
    seen: set[Node] = set()
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(_iter_node_args(node.args))
        stack.extend(_iter_node_args(node.kwargs))
    return {node.name for node in seen}


def _clear_quantization_annotations(
    gm: GraphModule, keep_fn: Callable[[Node], bool]
) -> None:
    for node in gm.graph.nodes:
        if not keep_fn(node):
            node.meta.pop(Q_ANNOTATION_KEY, None)


def _node_module_stack_strings(node: Node) -> tuple[list[str], list[str]]:
    module_paths: list[str] = []
    module_types: list[str] = []
    nn_module_stack = node.meta.get("nn_module_stack")
    if isinstance(nn_module_stack, dict):
        for key, value in nn_module_stack.items():
            module_paths.append(str(key))
            if isinstance(value, (tuple, list)):
                if value:
                    module_paths.append(str(value[0]))
                if len(value) > 1:
                    module_type = value[1]
                    module_types.append(getattr(module_type, "__name__", str(module_type)))
            else:
                module_paths.append(str(value))
    target = str(node.target if node.target is not None else node.name)
    if target.startswith(("p_", "b_")):
        module_paths.append(target)
        module_paths.append(target[2:])
        module_paths.append(target[2:].replace("_", "."))
    return module_paths, module_types


def _collect_related_nodes(
    nodes: Iterable[Node],
    *,
    max_depth: int = 3,
) -> list[Node]:
    related: list[Node] = []
    seen: set[Node] = set()
    stack: list[tuple[Node, int]] = [(node, 0) for node in nodes]
    while stack:
        node, depth = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        related.append(node)
        if depth >= max_depth:
            continue
        for arg in _iter_node_args(node.args):
            stack.append((arg, depth + 1))
        for arg in _iter_node_args(node.kwargs):
            stack.append((arg, depth + 1))
    return related


def _node_match_strings(node: Node) -> dict[str, list[str]]:
    module_paths, module_types = _node_module_stack_strings(node)
    target = str(node.target if node.target is not None else node.name)
    fields = {
        "name": [node.name],
        "target": [target],
        "module": module_paths,
        "type": module_types,
    }
    fields["any"] = [
        text
        for values in fields.values()
        for text in values
        if text
    ]
    return fields


def _lower_match_texts(nodes: Iterable[Node], *, max_depth: int = 3) -> list[str]:
    return [
        text.lower()
        for node in _collect_related_nodes(nodes, max_depth=max_depth)
        for text in _node_match_strings(node)["any"]
    ]


def _detect_head_model_indices(gm: GraphModule) -> set[str]:
    indices: set[str] = set()
    for node in gm.graph.nodes:
        if node.op not in {"placeholder", "get_attr"}:
            continue
        for text in _lower_match_texts((node,), max_depth=0):
            match = _YOLO_DETECT_INDEX_RE.search(text)
            if match is not None:
                indices.add(match.group(1))
    return indices


def _attention_model_indices(gm: GraphModule) -> set[str]:
    indices: set[str] = set()
    for node in gm.graph.nodes:
        if node.op not in {"placeholder", "get_attr"}:
            continue
        for text in _lower_match_texts((node,), max_depth=0):
            match = _YOLO_ATTENTION_INDEX_RE.search(text)
            if match is not None:
                indices.add(match.group(1))
    return indices


def _is_loss_path_related(
    nodes: Iterable[Node],
    loss_only_node_names: set[str] | None,
    *,
    max_depth: int = 3,
    options: JointBackwardAnnotationOptions | None = None,
) -> bool:
    options = _resolve_annotation_options(options)
    related = _collect_related_nodes(nodes, max_depth=max_depth)
    if loss_only_node_names is not None and any(
        node.name in loss_only_node_names for node in related
    ):
        return True
    if any(
        node.op == "call_function" and node.target in options.loss_path_targets
        for node in related
    ):
        return True
    for node in related:
        if node.op != "placeholder":
            continue
        if any(
            token in text
            for text in _lower_match_texts((node,), max_depth=0)
            for token in options.loss_path_text_tokens
        ):
            return True
    return False


def _is_attention_related(
    nodes: Iterable[Node],
    *,
    attention_model_indices: set[str] | None = None,
    max_depth: int = 3,
) -> bool:
    related = _collect_related_nodes(nodes, max_depth=max_depth)
    if any(
        node.op == "call_function" and node.target in _ATTENTION_TARGETS
        for node in related
    ):
        return True
    texts = _lower_match_texts(related, max_depth=0)
    token_texts = _lower_match_texts(
        _collect_related_nodes(nodes, max_depth=min(max_depth, 4)),
        max_depth=0,
    )
    if any(
        token in text
        for text in token_texts
        for token in _ATTENTION_TEXT_TOKENS
    ):
        return True
    attention_model_indices = attention_model_indices or set()
    return any(
        f"model_model_{index}_" in text
        for text in texts
        for index in attention_model_indices
    )


def _is_detect_head_related(text: str, detect_head_indices: set[str]) -> bool:
    if "detect_head" in text or ".detect." in text or "_detect_" in text:
        return True
    return any(f"model_model_{index}_" in text for index in detect_head_indices)


def _is_model_parameter_text(text: str) -> bool:
    return bool(
        _YOLO_MODEL_PARAM_RE.match(text) or _NAMED_MODEL_PARAM_RE.match(text)
    )


def _is_model_path_related(
    nodes: Iterable[Node],
    *,
    detect_head_indices: set[str],
    max_depth: int = 4,
) -> bool:
    texts = _lower_match_texts(nodes, max_depth=max_depth)
    if any(_is_detect_head_related(text, detect_head_indices) for text in texts):
        return True
    return any(_is_model_parameter_text(text) for text in texts)


def _annotation_rule_edge_decision(
    *,
    node: Node,
    input_node: Node,
    origin: Node,
    detect_head_indices: set[str],
    model_backward_node_names: set[str],
    attention_model_indices: set[str],
    loss_only_node_names: set[str] | None,
    options: JointBackwardAnnotationOptions | None = None,
) -> str:
    if _is_loss_path_related((node,), loss_only_node_names, max_depth=0, options=options):
        return "skip_loss_path"
    if _is_loss_path_related(
        (input_node, origin),
        loss_only_node_names,
        max_depth=1,
        options=options,
    ):
        return "skip_loss_path"
    related = (node, input_node, origin)
    if _is_attention_related(
        related,
        attention_model_indices=attention_model_indices,
        max_depth=4,
    ):
        return "skip_attention"
    if any(
        item.name in model_backward_node_names for item in (node, input_node, origin)
    ):
        return "quantize"
    if not _is_model_path_related(
        related,
        detect_head_indices=detect_head_indices,
        max_depth=5,
    ):
        return "skip_non_model"
    return "quantize"


def _annotation_rule_output_decision(
    *,
    node: Node,
    detect_head_indices: set[str],
    model_backward_node_names: set[str],
    attention_model_indices: set[str],
    loss_only_node_names: set[str] | None,
    options: JointBackwardAnnotationOptions | None = None,
) -> str:
    related = (node,)
    if _is_loss_path_related(related, loss_only_node_names, max_depth=1, options=options):
        return "skip_loss_path"
    if _is_attention_related(
        related,
        attention_model_indices=attention_model_indices,
        max_depth=8,
    ):
        return "skip_attention"
    if node.name in model_backward_node_names:
        return "quantize"
    if not _is_model_path_related(
        related,
        detect_head_indices=detect_head_indices,
        max_depth=5,
    ):
        return "skip_non_model"
    return "quantize"


def _annotation_rule_model_backward_node_names(
    gm: GraphModule,
    *,
    phase_forward_node_names: set[str] | None,
    loss_output_names: set[str],
    detect_head_indices: set[str],
    attention_model_indices: set[str],
    loss_only_node_names: set[str] | None,
    options: JointBackwardAnnotationOptions | None = None,
) -> set[str]:
    roots: list[Node] = []
    in_backward = False
    for node in gm.graph.nodes:
        if phase_forward_node_names is None:
            if node.name in loss_output_names:
                in_backward = True
        else:
            in_backward = node.name not in phase_forward_node_names
        if not in_backward:
            continue
        if node.name in loss_output_names:
            continue
        if _is_loss_path_related((node,), loss_only_node_names, max_depth=1, options=options):
            continue
        if _is_attention_related(
            (node,),
            attention_model_indices=attention_model_indices,
            max_depth=4,
        ):
            continue
        if _is_model_path_related(
            (node,),
            detect_head_indices=detect_head_indices,
            max_depth=5,
        ):
            roots.append(node)

    seen: set[Node] = set()
    stack = list(roots)
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        if node.name in loss_output_names:
            continue
        if _is_loss_path_related((node,), loss_only_node_names, max_depth=1, options=options):
            continue
        if _is_attention_related(
            (node,),
            attention_model_indices=attention_model_indices,
            max_depth=4,
        ):
            continue
        seen.add(node)
        stack.extend(_iter_node_args(node.args))
        stack.extend(_iter_node_args(node.kwargs))
    return {node.name for node in seen}


def _has_annotation_rule_boundary_consumer(
    node: Node,
    model_backward_node_names: set[str],
    options: JointBackwardAnnotationOptions | None = None,
) -> bool:
    options = _resolve_annotation_options(options)
    return any(
        user.name in model_backward_node_names
        and user.op == "call_function"
        and user.target in options.boundary_consumer_targets
        for user in node.users
    )


def _normalize_layerwise_edge_formats(section: object) -> dict[str, str]:
    if section is None:
        return {}
    if isinstance(section, str):
        section = {"format": section}
    if not isinstance(section, dict):
        raise TypeError("layer quantization config entries must be strings or objects")

    formats: dict[str, str] = {}
    default_format = section.get("format")
    forward_format = section.get("forward", default_format)
    backward_format = section.get("backward", default_format)
    for edge_kind in _LAYERWISE_EDGE_KINDS:
        phase_format = (
            forward_format
            if edge_kind in _LAYERWISE_FORWARD_EDGE_KINDS
            else backward_format
        )
        if phase_format is not None:
            formats[edge_kind] = str(phase_format)

    edge_overrides = section.get("edges", {})
    if not isinstance(edge_overrides, dict):
        raise TypeError("layer quantization config 'edges' must be an object")
    for edge_kind, quant_format in edge_overrides.items():
        if edge_kind not in _LAYERWISE_EDGE_KINDS:
            raise ValueError(f"unsupported layer quantization edge kind: {edge_kind}")
        formats[str(edge_kind)] = str(quant_format)

    for edge_kind in _LAYERWISE_EDGE_KINDS:
        if edge_kind in section:
            formats[edge_kind] = str(section[edge_kind])

    for edge_kind, quant_format in formats.items():
        allowed_formats = (
            _LAYERWISE_FORWARD_FORMATS
            if edge_kind in _LAYERWISE_FORWARD_EDGE_KINDS
            else _LAYERWISE_BACKWARD_FORMATS
        )
        if quant_format not in allowed_formats:
            raise ValueError(
                f"layer quantization format for {edge_kind} must be one of "
                f"{sorted(allowed_formats)}, got {quant_format!r}"
            )
    return formats


class _LayerWiseQuantRule:
    def __init__(self, index: int, raw_rule: dict[str, Any]) -> None:
        self.index = index
        self.name = str(raw_rule.get("name", f"rule_{index}"))
        match = raw_rule.get("match", {})
        if not isinstance(match, dict):
            raise TypeError("layer quantization rule 'match' must be an object")
        self.matchers: dict[str, list[re.Pattern[str]]] = {}
        for raw_key, raw_patterns in match.items():
            key = str(raw_key)
            if key.endswith("_regex"):
                key = key[: -len("_regex")]
            if key == "scope":
                key = "module"
            if key not in {"name", "target", "module", "type", "any"}:
                raise ValueError(f"unsupported layer quantization match key: {raw_key}")
            patterns = [
                re.compile(pattern)
                for pattern in _as_string_list(raw_patterns)
            ]
            self.matchers[key] = patterns
        if not self.matchers:
            raise ValueError("layer quantization rule must contain at least one matcher")
        self.edge_formats = _normalize_layerwise_edge_formats(raw_rule)

    def matches(self, nodes: Iterable[Node], *, max_depth: int = 3) -> bool:
        field_values: dict[str, list[str]] = {
            "name": [],
            "target": [],
            "module": [],
            "type": [],
            "any": [],
        }
        for node in _collect_related_nodes(nodes, max_depth=max_depth):
            for key, values in _node_match_strings(node).items():
                field_values[key].extend(values)

        for key, patterns in self.matchers.items():
            values = field_values[key]
            if not all(
                any(pattern.search(value) for value in values)
                for pattern in patterns
            ):
                return False
        return True

    def format_for(self, edge_kind: str) -> str | None:
        return self.edge_formats.get(edge_kind)


class _LayerWiseQuantFormatResolver:
    def __init__(self, config: dict[str, Any]) -> None:
        if not isinstance(config, dict):
            raise TypeError("layer quantization config must be a JSON object")
        self.default_formats = _normalize_layerwise_edge_formats(
            config.get("default", {})
        )
        rules = config.get("rules", [])
        if not isinstance(rules, list):
            raise TypeError("layer quantization config 'rules' must be a list")
        self.rules = [
            _LayerWiseQuantRule(index, rule)
            for index, rule in enumerate(rules)
            if isinstance(rule, dict)
        ]
        if len(self.rules) != len(rules):
            raise TypeError("layer quantization rules must be objects")
        self._activation_qspecs = {
            "int8": get_affine_activation_qdq_config().input_activation,
            "int16": get_affine_activation_int16_qdq_config().input_activation,
        }
        self._gradient_qspecs = {
            "int8": get_symmetric_gradient_qdq_config().input_activation,
            "int16": get_symmetric_gradient_int16_qdq_config().input_activation,
        }
        self._fp16_qspec = get_fp16_activation_qdq_config().input_activation
        self.reset_report()

    def reset_report(self) -> None:
        self.format_counts: dict[str, dict[str, int]] = {}
        self.rule_counts: dict[str, int] = {}

    def _qspec_for_format(
        self,
        edge_kind: str,
        quant_format: str,
    ) -> QuantizationSpec:
        if quant_format == "fp16":
            if edge_kind in _LAYERWISE_FORWARD_EDGE_KINDS:
                raise ValueError(f"fp16 is only supported for backward edges: {edge_kind}")
            return self._fp16_qspec
        if edge_kind in _LAYERWISE_AFFINE_ACTIVATION_EDGE_KINDS:
            return self._activation_qspecs[quant_format]
        if edge_kind in _LAYERWISE_GRADIENT_EDGE_KINDS:
            return self._gradient_qspecs[quant_format]
        raise ValueError(f"unsupported layer quantization edge kind: {edge_kind}")

    def __call__(
        self,
        edge_kind: str,
        fallback_qspec: QuantizationSpec,
        nodes: tuple[Node, ...],
    ) -> QuantizationSpec:
        if edge_kind not in _LAYERWISE_EDGE_KINDS:
            raise ValueError(f"unsupported layer quantization edge kind: {edge_kind}")
        quant_format = self.default_formats.get(edge_kind)
        matched_rule_name: str | None = None
        match_depth = 1 if edge_kind in {
            "backward_silu_input",
            "backward_conv_input",
        } else 3
        for rule in self.rules:
            if not rule.matches(nodes, max_depth=match_depth):
                continue
            rule_format = rule.format_for(edge_kind)
            if rule_format is not None:
                quant_format = rule_format
                matched_rule_name = rule.name
        if quant_format is None:
            return fallback_qspec

        edge_counts = self.format_counts.setdefault(edge_kind, {})
        edge_counts[quant_format] = edge_counts.get(quant_format, 0) + 1
        if matched_rule_name is not None:
            self.rule_counts[matched_rule_name] = (
                self.rule_counts.get(matched_rule_name, 0) + 1
            )
        return self._qspec_for_format(edge_kind, quant_format)

    def report(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "format_counts": self.format_counts,
            "rule_match_counts": self.rule_counts,
        }


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
        backward_quantization_mode: str = "all",
        use_xnnpack_forward_quantizer: bool = True,
        annotate_silu_edges: bool = True,
        annotate_pre_silu_edges: bool = True,
        silu_output_filter_fn: Callable[[Node], bool] | None = None,
        forward_quantization_config: QuantizationConfig | None = None,
        forward_filter_fn: Callable[[Node], bool] | None = None,
        layer_quantization_config: dict[str, Any] | None = None,
        annotation_options: JointBackwardAnnotationOptions | None = None,
    ) -> None:
        super().__init__()
        if backward_quantization_mode not in _BACKWARD_QUANTIZATION_MODES:
            raise ValueError(
                "backward_quantization_mode must be one of "
                f"{sorted(_BACKWARD_QUANTIZATION_MODES)}"
            )
        self.activation_config = activation_config or get_affine_activation_qdq_config()
        self.pre_silu_activation_config = (
            pre_silu_activation_config or get_affine_activation_qdq_config()
        )
        self.gradient_config = gradient_config or get_symmetric_gradient_qdq_config()
        self.weight_config = weight_config or get_symmetric_weight_qdq_config()
        self.quantize_final_outputs = quantize_final_outputs
        self.backward_quantization_mode = backward_quantization_mode
        self.use_xnnpack_forward_quantizer = use_xnnpack_forward_quantizer
        self.annotate_silu_edges = annotate_silu_edges
        self.annotate_pre_silu_edges = annotate_pre_silu_edges
        self.silu_output_filter_fn = silu_output_filter_fn
        self.forward_quantization_config = (
            forward_quantization_config
            or get_symmetric_quantization_config(is_per_channel=False)
        )
        self.forward_filter_fn = forward_filter_fn
        self.annotation_options = annotation_options
        self.layer_quantization_resolver = (
            _LayerWiseQuantFormatResolver(layer_quantization_config)
            if layer_quantization_config is not None
            else None
        )
        self.loss_output_names: set[str] = set()
        self.user_output_names: set[str] = set()
        self.final_output_names: set[str] = set()
        self.annotation_report: dict[str, Any] = {}

    def configure_from_exported_program(self, exported_program) -> None:
        self.loss_output_names = {
            spec.arg.name
            for spec in exported_program.graph_signature.output_specs
            if spec.kind == OutputKind.LOSS_OUTPUT and hasattr(spec.arg, "name")
        }
        self.user_output_names = {
            spec.arg.name
            for spec in exported_program.graph_signature.output_specs
            if spec.kind == OutputKind.USER_OUTPUT and hasattr(spec.arg, "name")
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
        user_forward_node_names = _model_forward_node_names(
            model, self.user_output_names
        )
        loss_forward_node_names = _model_forward_node_names(
            model, self.loss_output_names
        )
        if self.use_xnnpack_forward_quantizer:
            forward_node_names = user_forward_node_names or _forward_node_names(
                model, self.loss_output_names
            )
            forward_quantizer = XNNPACKQuantizer().set_global(
                self.forward_quantization_config
            )
            forward_quantizer.set_filter_function(
                lambda node: node.name in forward_node_names
                and (
                    self.forward_filter_fn is None or self.forward_filter_fn(node)
                )
            )
            model = forward_quantizer.annotate(model)
            if self.forward_filter_fn is not None:
                _clear_quantization_annotations(
                    model,
                    lambda node: node.name in forward_node_names
                    and self.forward_filter_fn is not None
                    and self.forward_filter_fn(node),
                )

        if self.layer_quantization_resolver is not None:
            self.layer_quantization_resolver.reset_report()

        self.annotation_report = annotate_joint_backward_qdq_edges(
            model,
            annotation_options=self.annotation_options,
            activation_qspec=self.activation_config.input_activation,
            pre_silu_activation_qspec=self.pre_silu_activation_config.input_activation,
            gradient_qspec=self.gradient_config.input_activation,
            weight_qspec=self.weight_config.weight,
            loss_output_names=self.loss_output_names,
            final_output_names=self.final_output_names
            if self.quantize_final_outputs
            else set(),
            backward_quantization_mode=self.backward_quantization_mode,
            annotate_forward_compute_edges=not self.use_xnnpack_forward_quantizer,
            annotate_silu_edges=self.annotate_silu_edges,
            annotate_pre_silu_edges=self.annotate_pre_silu_edges,
            silu_output_filter_fn=self.silu_output_filter_fn,
            phase_forward_node_names=loss_forward_node_names | user_forward_node_names,
            loss_only_node_names=loss_forward_node_names - user_forward_node_names,
            qspec_resolver=self.layer_quantization_resolver,
        )
        if self.layer_quantization_resolver is not None:
            self.annotation_report["joint_qdq_layerwise_config"] = (
                self.layer_quantization_resolver.report()
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
    annotation_options: JointBackwardAnnotationOptions | None = None,
    backward_quantization_mode: str = "all",
    annotate_forward_compute_edges: bool = True,
    annotate_silu_edges: bool = True,
    annotate_pre_silu_edges: bool = True,
    silu_output_filter_fn: Callable[[Node], bool] | None = None,
    phase_forward_node_names: set[str] | None = None,
    loss_only_node_names: set[str] | None = None,
    qspec_resolver: Callable[
        [str, QuantizationSpec, tuple[Node, ...]], QuantizationSpec
    ]
    | None = None,
) -> dict[str, Any]:
    def resolve_qspec(
        edge_kind: str,
        fallback_qspec: QuantizationSpec,
        *nodes: Node,
    ) -> QuantizationSpec:
        if qspec_resolver is None:
            return fallback_qspec
        return qspec_resolver(edge_kind, fallback_qspec, tuple(nodes))

    if annotate_silu_edges:
        pre_silu_sources, silu_outputs, silu_internal_sigmoids = _find_silu_qdq_nodes(gm)
        if silu_output_filter_fn is not None:
            silu_outputs = {node for node in silu_outputs if silu_output_filter_fn(node)}
        if not annotate_pre_silu_edges:
            pre_silu_sources = set()
            silu_internal_sigmoids = set()
    else:
        pre_silu_sources, silu_outputs, silu_internal_sigmoids = set(), set(), set()
    producer_output_qspec_nodes = pre_silu_sources | silu_outputs
    annotation_rule_enabled = backward_quantization_mode == "annotation_rule"
    detect_head_indices = (
        _detect_head_model_indices(gm) if annotation_rule_enabled else set()
    )
    attention_model_indices = (
        _attention_model_indices(gm) if annotation_rule_enabled else set()
    )
    model_backward_node_names = (
        _annotation_rule_model_backward_node_names(
            gm,
            options=annotation_options,
            phase_forward_node_names=phase_forward_node_names,
            loss_output_names=loss_output_names,
            detect_head_indices=detect_head_indices,
            attention_model_indices=attention_model_indices,
            loss_only_node_names=loss_only_node_names,
        )
        if annotation_rule_enabled
        else set()
    )
    annotation_rule_counts = {
        "quantized_input_edges": 0,
        "quantized_output_nodes": 0,
        "skip_loss_path": 0,
        "skip_attention": 0,
        "skip_non_model": 0,
        "reuse_producer_output_qspec": 0,
    }

    input_edges = 0
    requested_output_qdq_nodes: set[Node] = set()
    annotated_nodes = 0
    skipped_view_like_consumers = 0
    by_phase = {"forward": 0, "backward": 0}
    conv_silu_backward_edges = 0
    forward_value_nodes: set[Node] = {
        node
        for node in gm.graph.nodes
        if phase_forward_node_names is not None and node.name in phase_forward_node_names
    }
    in_backward = False

    for node in gm.graph.nodes:
        if node.op not in {"call_function", "call_method"}:
            continue
        if phase_forward_node_names is None:
            if node.name in loss_output_names:
                in_backward = True
        else:
            in_backward = node.name not in phase_forward_node_names
        phase = "backward" if in_backward else "forward"

        output_qspec = None
        if node in pre_silu_sources and _is_float_tensor_node(node):
            output_qspec = resolve_qspec(
                "forward_pre_silu", pre_silu_activation_qspec, node
            )
        elif node in silu_outputs and _is_float_tensor_node(node):
            output_qspec = resolve_qspec(
                "forward_silu_output", activation_qspec, node
            )
        elif node.name in final_output_names and _is_float_tensor_node(node):
            if in_backward:
                output_qspec = resolve_qspec(
                    "final_backward_output", gradient_qspec, node
                )
            else:
                output_qspec = resolve_qspec(
                    "final_forward_output", activation_qspec, node
                )
        elif (
            annotation_rule_enabled
            and in_backward
            and _is_float_tensor_node(node)
            and not _is_metadata_only_factory_node(node)
            and not _is_view_like_node(node)
            and _has_annotation_rule_boundary_consumer(
                node,
                model_backward_node_names,
                options=annotation_options,
            )
        ):
            output_qspec = resolve_qspec("backward_gradient", gradient_qspec, node)
        if output_qspec is not None and annotation_rule_enabled:
            decision = _annotation_rule_output_decision(
                options=annotation_options,
                node=node,
                detect_head_indices=detect_head_indices,
                model_backward_node_names=model_backward_node_names,
                attention_model_indices=attention_model_indices,
                loss_only_node_names=loss_only_node_names,
            )
            if decision != "quantize":
                annotation_rule_counts[decision] += 1
                output_qspec = None
            else:
                annotation_rule_counts["quantized_output_nodes"] += 1
        if output_qspec is not None:
            requested_output_qdq_nodes.add(node)
            producer_output_qspec_nodes.add(node)

        if not _is_compute_consumer(node) and output_qspec is None:
            skipped_view_like_consumers += int(_is_view_like_node(node))
            if phase_forward_node_names is None and not in_backward:
                forward_value_nodes.add(node)
            continue

        input_qspec_map: dict[Node, QuantizationSpec] = {}
        if _is_compute_consumer(node):
            for input_node in _iter_node_args(node.args):
                if node.kwargs:
                    continue
                if in_backward and backward_quantization_mode == "none":
                    continue
                if in_backward and backward_quantization_mode == "conv_silu":
                    continue
                if not in_backward and not annotate_forward_compute_edges:
                    continue
                if in_backward:
                    origin = _origin_node(input_node)
                    if input_node.name in loss_output_names or (
                        loss_only_node_names is not None
                        and origin.name in loss_only_node_names
                    ):
                        continue
                if _is_metadata_only_factory_node(node):
                    continue
                if not _is_quantizable_input_node(input_node):
                    continue
                origin = _origin_node(input_node)
                if _is_metadata_only_factory_node(
                    input_node
                ) or _is_metadata_only_factory_node(origin):
                    continue
                if origin in producer_output_qspec_nodes:
                    annotation_rule_counts["reuse_producer_output_qspec"] += 1
                    continue
                # Which qspec this edge would otherwise get; mirrors the
                # three-way choice made further down.
                saved_activation_edge = in_backward and (
                    origin in forward_value_nodes
                    or _is_user_activation_placeholder(origin)
                )
                if _is_parameter_weight_node(input_node):
                    candidate_qspec = weight_qspec
                elif saved_activation_edge:
                    candidate_qspec = activation_qspec
                elif in_backward:
                    candidate_qspec = gradient_qspec
                else:
                    candidate_qspec = activation_qspec
                # A producer annotated by an earlier pass already emits a
                # quantized output.  Observing it again here gives each
                # consumer its own scale, and consumers that must agree on the
                # wire (the DFL loss backward and the decode backward both read
                # cat_17) then disagree once calibration sees enough samples to
                # separate them.  Reuse the producer's qspec when the wire
                # format matches.
                if _qspec_interchangeable(
                    _existing_output_qspec(origin), candidate_qspec
                ):
                    annotation_rule_counts["reuse_producer_output_qspec"] += 1
                    producer_output_qspec_nodes.add(origin)
                    continue
                if input_node in silu_internal_sigmoids:
                    continue
                if _is_decomposed_silu_internal_edge(input_node, node):
                    continue
                if annotation_rule_enabled:
                    decision = _annotation_rule_edge_decision(
                        options=annotation_options,
                        node=node,
                        input_node=input_node,
                        origin=origin,
                        detect_head_indices=detect_head_indices,
                        model_backward_node_names=model_backward_node_names,
                        attention_model_indices=attention_model_indices,
                        loss_only_node_names=loss_only_node_names,
                    )
                    if decision != "quantize":
                        annotation_rule_counts[decision] += 1
                        continue
                    annotation_rule_counts["quantized_input_edges"] += 1
                if _is_parameter_weight_node(input_node):
                    input_qspec_map[input_node] = weight_qspec
                elif in_backward and (
                    origin in forward_value_nodes
                    or _is_user_activation_placeholder(origin)
                ):
                    input_qspec_map[input_node] = resolve_qspec(
                        "backward_saved_activation",
                        activation_qspec,
                        node,
                        input_node,
                        origin,
                    )
                else:
                    edge_kind = (
                        "backward_gradient" if in_backward else "forward_activation"
                    )
                    input_qspec_map[input_node] = resolve_qspec(
                        edge_kind,
                        gradient_qspec if in_backward else activation_qspec,
                        node,
                        input_node,
                        origin,
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
            by_phase[phase] += 1
        if phase_forward_node_names is None and not in_backward:
            forward_value_nodes.add(node)

    if backward_quantization_mode == "conv_silu":
        for silu_grad, gradient_input, conv_backward in _find_plain_silu_conv_backward_edges(gm):
            if _merge_quantization_annotation(
                silu_grad,
                input_qspec_map={
                    gradient_input: resolve_qspec(
                        "backward_silu_input",
                        activation_qspec,
                        silu_grad,
                        gradient_input,
                        conv_backward,
                    )
                },
                allow_implicit_sharing=False,
            ):
                annotated_nodes += 1
                input_edges += 1
                by_phase["backward"] += 1
            if _merge_quantization_annotation(
                conv_backward,
                input_qspec_map={
                    silu_grad: resolve_qspec(
                        "backward_conv_input",
                        gradient_qspec,
                        conv_backward,
                        silu_grad,
                        gradient_input,
                    )
                },
                allow_implicit_sharing=False,
            ):
                annotated_nodes += 1
                input_edges += 1
                by_phase["backward"] += 1
            conv_silu_backward_edges += 1

    return {
        "joint_qdq_annotated_nodes": annotated_nodes,
        "joint_qdq_input_edges": input_edges,
        "joint_qdq_output_nodes": len(requested_output_qdq_nodes),
        "joint_qdq_forward_nodes": by_phase["forward"],
        "joint_qdq_backward_nodes": by_phase["backward"],
        "joint_qdq_skipped_view_like_consumers": skipped_view_like_consumers,
        "joint_qdq_pre_silu_sources": len(pre_silu_sources),
        "joint_qdq_silu_outputs": len(silu_outputs),
        "joint_qdq_silu_internal_sigmoids": len(silu_internal_sigmoids),
        "joint_qdq_conv_silu_backward_edges": conv_silu_backward_edges,
        "joint_qdq_annotation_rule_detect_head_indices": sorted(detect_head_indices),
        "joint_qdq_annotation_rule_attention_indices": sorted(
            attention_model_indices
        ),
        "joint_qdq_annotation_rule_model_backward_nodes": len(
            model_backward_node_names
        ),
        "joint_qdq_annotation_rule": annotation_rule_counts,
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
    if not _is_quantize_per_tensor_node(node) or len(node.args) < 5:
        return False
    zero_point = int(node.args[2])
    quant_min = int(node.args[3])
    quant_max = int(node.args[4])
    return zero_point == 0 and quant_min in {-127, -32767} and -quant_min == quant_max


def _is_affine_qdq_quantize(node: Node) -> bool:
    if not _is_quantize_per_tensor_node(node) or len(node.args) < 5:
        return False
    quant_min = int(node.args[3])
    quant_max = int(node.args[4])
    return (quant_min, quant_max) in {(-128, 127), (-32768, 32767)}


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


def _is_alias_of(node: object, source: Node) -> bool:
    if node is source:
        return True
    if not isinstance(node, Node):
        return False
    quantize_node = _dequantize_quantize_node(node)
    if quantize_node is not None:
        qdq_source = _quantize_source(quantize_node)
        return qdq_source is not None and _is_alias_of(qdq_source, source)
    if node.op == "call_function" and node.target == torch.ops.aten.clone.default:
        return bool(node.args and _is_alias_of(node.args[0], source))
    return False


def _contains_sigmoid_of_alias(
    node: object,
    source: Node,
    *,
    max_depth: int = 8,
    seen: set[Node] | None = None,
) -> bool:
    if not isinstance(node, Node) or max_depth < 0:
        return False
    if (
        node.op == "call_function"
        and node.target == torch.ops.aten.sigmoid.default
        and node.args
        and _is_alias_of(node.args[0], source)
    ):
        return True
    seen = seen or set()
    if node in seen:
        return False
    seen.add(node)
    return any(
        _contains_sigmoid_of_alias(
            arg, source, max_depth=max_depth - 1, seen=seen
        )
        for arg in _iter_node_args(node.args)
    )


def _is_converted_decomposed_silu_node(node: Node, quantize_node: Node) -> bool:
    if (
        node.op != "call_function"
        or node.target != torch.ops.aten.mul.Tensor
        or len(node.args) < 2
    ):
        return False
    qdq_source = _quantize_source(quantize_node)
    if qdq_source is None:
        return False
    lhs, rhs = node.args[:2]
    return (
        _is_alias_of(lhs, qdq_source)
        and _contains_sigmoid_of_alias(rhs, qdq_source)
    ) or (
        _is_alias_of(rhs, qdq_source)
        and _contains_sigmoid_of_alias(lhs, qdq_source)
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
    qdq_source = _quantize_source(quantize_node)
    if qdq_source is None:
        return False
    return (
        source.op == "call_function"
        and source.target == torch.ops.aten.mul.Tensor
        and (
            _contains_sigmoid_of_dequantize_from(source, quantize_node)
            or _contains_sigmoid_of_alias(source, qdq_source)
        )
    )


def _node_shape(node: Node) -> tuple[int, ...] | None:
    source = _unwrap_qdq_source(node)
    val = source.meta.get("val")
    if hasattr(val, "shape"):
        return tuple(val.shape)
    tensor_meta = source.meta.get("tensor_meta")
    if hasattr(tensor_meta, "shape"):
        return tuple(tensor_meta.shape)
    return None


def _same_known_shape(lhs: Node, rhs: Node) -> bool:
    lhs_shape = _node_shape(lhs)
    rhs_shape = _node_shape(rhs)
    if lhs_shape is None or rhs_shape is None:
        return True
    return lhs_shape == rhs_shape


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
        if not _is_quantize_per_tensor_node(node) or not _is_affine_qdq_quantize(node):
            continue
        pre_silu_quantize = node
        pre_silu_source = _quantize_source(pre_silu_quantize)
        if pre_silu_source is None:
            continue
        silu_nodes = [
            candidate
            for candidate in gm.graph.nodes
            if _is_converted_decomposed_silu_node(candidate, pre_silu_quantize)
        ]
        for silu_node in silu_nodes:
            qasym_candidates = [
                user
                for user in _direct_quantize_users(silu_node)
                if _is_affine_qdq_quantize(user)
            ]
            for qasym_quantize in qasym_candidates:
                matches.append(
                    (pre_silu_source, pre_silu_quantize, silu_node, qasym_quantize)
                )
    return matches


def _find_silu_backward_grads(
    gm: GraphModule, pre_silu_quantize: Node
) -> list[tuple[Node, Node]]:
    candidates: list[tuple[Node, Node]] = []
    pre_silu_source = _quantize_source(pre_silu_quantize)
    if pre_silu_source is None:
        return candidates
    for node in gm.graph.nodes:
        if (
            node.op != "call_function"
            or node.target != torch.ops.aten.mul.Tensor
            or len(node.args) < 2
        ):
            continue
        node_args = [arg for arg in node.args[:2] if isinstance(arg, Node)]
        if any(
            arg.op == "call_function"
            and arg.target == torch.ops.aten._to_copy.default
            and isinstance(arg.args[0], Node)
            and arg.args[0].target == torch.ops.aten.logical_and.default
            for arg in node_args
        ):
            continue
        for derivative_arg in node_args:
            if not _is_silu_derivative_arg(derivative_arg, pre_silu_quantize):
                continue
            for gradient_input in node_args:
                if gradient_input is derivative_arg:
                    continue
                if not _same_known_shape(gradient_input, pre_silu_source):
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
    pre_silu_masks = 0
    matched_backward = 0
    matches = _find_converted_silu_qdq(gm)
    if not matches:
        return {
            "joint_ste_qasym_masks": 0,
            "joint_ste_qsym_masks": 0,
            "joint_ste_conv_silu_matches": 0,
        }

    for pre_silu_source, pre_silu_quantize, silu_node, qasym_quantize in matches:
        pre_silu_quantize_source = _quantize_source(pre_silu_quantize)
        qasym_source = _quantize_source(qasym_quantize)
        if pre_silu_quantize_source is None or qasym_source is None:
            continue
        if pre_silu_quantize_source is not pre_silu_source or qasym_source is not silu_node:
            continue

        silu_grad_matches = _find_silu_backward_grads(gm, pre_silu_quantize)
        if not silu_grad_matches:
            continue

        for silu_grad, gradient_input in silu_grad_matches:
            silu_grad_quantize_users = [
                user
                for user in _direct_quantize_users(silu_grad)
                if _is_symmetric_qdq_quantize(user)
            ]
            conv_backward_users = [
                user
                for user in silu_grad.users
                if user.op == "call_function"
                and user.target == torch.ops.aten.convolution_backward.default
                and user.args
                and user.args[0] is silu_grad
            ]
            if not silu_grad_quantize_users and not conv_backward_users:
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

            pre_silu_mask_insert_before = (
                silu_grad_quantize_users[0]
                if silu_grad_quantize_users
                else conv_backward_users[0]
            )
            pre_silu_mask = _insert_ste_mask(
                gm,
                source_node=pre_silu_quantize_source,
                quantize_node=pre_silu_quantize,
                insert_before=pre_silu_mask_insert_before,
            )
            with gm.graph.inserting_before(pre_silu_mask_insert_before):
                masked_silu_grad = gm.graph.call_function(
                    torch.ops.aten.mul.Tensor, args=(silu_grad, pre_silu_mask)
                )
            for quantize_user in silu_grad_quantize_users:
                quantize_user.args = (masked_silu_grad, *quantize_user.args[1:])
            for conv_backward in conv_backward_users:
                _replace_node_arg(conv_backward, silu_grad, masked_silu_grad)
            if silu_grad_quantize_users or conv_backward_users:
                pre_silu_masks += 1
                matched_backward += 1

    gm.graph.eliminate_dead_code()
    gm.recompile()
    return {
        "joint_ste_qasym_masks": qasym_masks,
        "joint_ste_qsym_masks": pre_silu_masks,
        "joint_ste_conv_silu_matches": matched_backward,
    }
