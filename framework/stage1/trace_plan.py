"""Trace-boundary planning primitives for Stage1.

This module is the first autonomy layer above the old AutoTraceAdapter
registry.  It inspects a full model module tree, tags modules with conservative
heuristics, and emits a TracePlan that can be embedded in Stage1 manifests.
The plan is an auditable boundary proposal; it is not a model-level
separability verdict.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

try:
    import torch_pruning as tp
except Exception:  # noqa: BLE001
    tp = None


TRACE_PLAN_SCHEMA = "stage1_trace_plan_v1"

HETER_BASELINE_MODELS = {"fcooper", "attfuse", "where2comm", "v2vnet", "disconet"}

_DENSE_NAME_HINTS = (
    "backbone",
    "resnet",
    "base_bev_backbone",
    "pyramid_backbone",
    "neck",
    "deblock",
    "shrink",
    "shrinker",
)

_HEAD_NAME_HINTS = (
    "cls_head",
    "reg_head",
    "dir_head",
    "single_head",
    "occ_head",
    "occupancy",
    "aux_head",
)

_SPARSE_HINTS = (
    "pillar_vfe",
    "vfe",
    "voxel",
    "scatter",
    "sparse",
    "quickcumsum",
    "quicksum",
    "cumsum",
    "lift",
    "splat",
    "geometry",
)

_FUSION_HINTS = (
    "fusion",
    "fuse",
    "warp",
    "affine",
    "pairwise_t_matrix",
    "record_len",
    "collab",
    "maxfusion",
)

_ATTENTION_HINTS = (
    "attention",
    "attfusion",
    "transformer",
    "hmsa",
    "mswin",
    "attfuse",
)

_ROUTING_HINTS = (
    "where2comm",
    "v2v",
    "v2vnet",
    "disco",
    "communication",
    "routing",
    "message_passing",
)

_POSTPROCESS_HINTS = (
    "postprocess",
    "post_process",
    "nms",
    "decode",
    "box_coder",
    "proposal",
)

_TRAINING_HINTS = (
    "loss",
    "assigner",
    "target",
    "metric",
    "eval",
)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _module_path(module_path: str) -> str:
    return module_path or "<root>"


def _text_for(path: str, type_name: str, class_name: str = "") -> str:
    return f"{path} {type_name} {class_name}".lower()


def _has_any(text: str, hints: Iterable[str]) -> bool:
    return any(hint in text for hint in hints)


def _param_count(module: nn.Module, *, recurse: bool) -> int:
    return int(sum(p.numel() for p in module.parameters(recurse=recurse)))


def _is_descendant(path: str, parent: str) -> bool:
    return path != parent and path.startswith(parent + ".")


def _is_under_any(path: str, parents: Iterable[str]) -> bool:
    return any(_is_descendant(path, parent) for parent in parents)


def _sort_paths(paths: Iterable[str]) -> list[str]:
    return sorted(set(paths), key=lambda p: (p.count("."), p))


def _get_module_by_path(model: nn.Module, path: str) -> nn.Module:
    if not path:
        return model
    module: nn.Module = model
    for part in path.split("."):
        if isinstance(module, nn.ModuleDict) and part in module:
            module = module[part]
        elif part.isdigit() and isinstance(module, (nn.Sequential, nn.ModuleList)):
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def _tensor_shapes(value: Any) -> list[list[int]]:
    if torch.is_tensor(value):
        return [list(value.shape)]
    if isinstance(value, dict):
        out: list[list[int]] = []
        for item in value.values():
            out.extend(_tensor_shapes(item))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_tensor_shapes(item))
        return out
    return []


@dataclass
class ModuleRecord:
    """One module-tree inventory row."""

    path: str
    type_name: str
    class_name: str
    depth: int
    n_children: int
    params_direct: int
    params_recursive: int
    tags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = _module_path(self.path)
        return data


@dataclass
class TraceCandidate:
    """A candidate dense trace boundary before wrapper validation."""

    candidate_id: str
    entry: str
    input_shape: list[int] | None
    included_modules: list[str]
    ignored_layers: list[str]
    skipped_subgraphs: list[str]
    confidence: str
    selection_reason: str
    wrapper_kind: str = "module_path"
    status: str = "candidate"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModuleTreeScanner:
    """Collect module inventory from a full nn.Module."""

    def scan(self, model: nn.Module) -> list[ModuleRecord]:
        records: list[ModuleRecord] = []
        for path, module in model.named_modules():
            class_name = type(module).__name__
            type_name = f"{type(module).__module__}.{class_name}"
            records.append(
                ModuleRecord(
                    path=path,
                    type_name=type_name,
                    class_name=class_name,
                    depth=0 if not path else path.count(".") + 1,
                    n_children=len(list(module.children())),
                    params_direct=_param_count(module, recurse=False),
                    params_recursive=_param_count(module, recurse=True),
                )
            )
        return records


class HeuristicTagger:
    """Conservative module tags used by TraceBoundaryDetector."""

    def tag(self, record: ModuleRecord) -> ModuleRecord:
        text = _text_for(record.path, record.type_name, record.class_name)
        tags: set[str] = set(record.tags)
        reasons: list[str] = list(record.reasons)

        if any(op in record.class_name for op in ("Conv2d", "ConvTranspose2d", "Linear", "BatchNorm2d")):
            tags.add("dense_op")
            reasons.append("module type is a dense torch op")
        if _has_any(text, _DENSE_NAME_HINTS):
            tags.add("dense_path_candidate")
            reasons.append("name matches dense backbone/neck/shrinker hints")
        if _has_any(text, _HEAD_NAME_HINTS):
            tags.add("head_candidate")
            tags.add("ignored_candidate")
            tags.add("dense_path_candidate")
            reasons.append("name matches output head hints")
        if _has_any(text, _SPARSE_HINTS):
            tags.add("skipped_candidate")
            tags.add("sparse_or_geometry_preprocess")
            reasons.append("name/type matches sparse or geometry preprocess hints")
        if _has_any(text, _ATTENTION_HINTS):
            tags.add("skipped_candidate")
            tags.add("attention_or_routing_fusion")
            reasons.append("name/type matches attention or transformer hints")
        if _has_any(text, _ROUTING_HINTS):
            tags.add("skipped_candidate")
            tags.add("attention_or_routing_fusion")
            reasons.append("name/type matches routing or message-passing hints")
        if _has_any(text, _FUSION_HINTS):
            tags.add("skipped_candidate")
            tags.add("fusion_or_alignment")
            reasons.append("name/type matches fusion or alignment hints")
        if _has_any(text, _POSTPROCESS_HINTS):
            tags.add("skipped_candidate")
            tags.add("postprocess_or_decode")
            reasons.append("name/type matches postprocess/decode hints")
        if _has_any(text, _TRAINING_HINTS):
            tags.add("skipped_candidate")
            tags.add("training_or_eval_only")
            reasons.append("name/type matches training/eval-only hints")
        if not tags and record.n_children == 0 and record.params_recursive > 0:
            tags.add("unknown_param_leaf")
            reasons.append("parameterized leaf did not match known Stage1 boundary hints")

        return ModuleRecord(
            path=record.path,
            type_name=record.type_name,
            class_name=record.class_name,
            depth=record.depth,
            n_children=record.n_children,
            params_direct=record.params_direct,
            params_recursive=record.params_recursive,
            tags=sorted(tags),
            reasons=sorted(set(reasons)),
        )

    def tag_many(self, records: Iterable[ModuleRecord]) -> list[ModuleRecord]:
        return [self.tag(record) for record in records]

    def skipped_type(self, record: ModuleRecord, *, model_name: str = "") -> tuple[str, bool, str | None]:
        text = _text_for(record.path, record.type_name, record.class_name)
        model = str(model_name).lower()
        is_fusion_boundary = _has_any(text, _FUSION_HINTS) or record.path.endswith("fusion_net")

        if _has_any(text, _SPARSE_HINTS):
            return "sparse_or_geometry_preprocess", False, "trace_sparse_frontend_boundary"
        if _has_any(text, _POSTPROCESS_HINTS):
            return "postprocess_or_decode", True, "postprocess_coverage_gate"
        if _has_any(text, _TRAINING_HINTS):
            return "training_or_eval_only", False, None
        if _has_any(text, _ATTENTION_HINTS):
            return "attention_or_routing_fusion", True, "attention_fusion_coverage_anchor"
        if _has_any(text, _ROUTING_HINTS):
            return "attention_or_routing_fusion", True, "routing_fusion_coverage_anchor"
        if "maxfusion" in text or "max_fusion" in text or (is_fusion_boundary and model == "fcooper"):
            return "fusion_or_alignment", False, "maxfusion_coverage_anchor"
        if is_fusion_boundary and model in {"where2comm", "v2vnet", "disconet"}:
            return "attention_or_routing_fusion", True, "routing_fusion_coverage_anchor"
        if is_fusion_boundary and model == "attfuse":
            return "attention_or_routing_fusion", True, "attention_fusion_coverage_anchor"
        if _has_any(text, _FUSION_HINTS):
            return "fusion_or_alignment", True, "routing_fusion_coverage_anchor"
        return "custom_untraced_subgraph", True, "custom_subgraph_coverage_gate"


class DensePathFinder:
    """Generate dense trace candidates from tagged module inventory."""

    def find(
        self,
        *,
        records: list[ModuleRecord],
        records_by_path: dict[str, ModuleRecord],
        tagger: HeuristicTagger,
        model_name: str,
        input_shape: Iterable[int] | None = None,
    ) -> tuple[list[TraceCandidate], list[dict[str, Any]], list[dict[str, Any]]]:
        modality = _infer_modality(records_by_path)
        if modality:
            return self._find_heter_baseline(
                records=records,
                records_by_path=records_by_path,
                tagger=tagger,
                model_name=model_name,
                modality=modality,
                input_shape=list(input_shape) if input_shape is not None else None,
            )
        return self._find_generic(
            records=records,
            records_by_path=records_by_path,
            tagger=tagger,
            model_name=model_name,
            input_shape=list(input_shape) if input_shape is not None else None,
        )

    def _top_skipped_paths(
        self,
        *,
        records: list[ModuleRecord],
        included_paths: list[str],
    ) -> list[str]:
        skip_records = [
            record
            for record in records
            if record.path
            and "skipped_candidate" in record.tags
            and not _is_under_any(record.path, included_paths)
        ]
        top_skip_paths: list[str] = []
        for path in _sort_paths(record.path for record in skip_records):
            if not _is_under_any(path, top_skip_paths):
                top_skip_paths.append(path)
        return top_skip_paths

    def _find_heter_baseline(
        self,
        *,
        records: list[ModuleRecord],
        records_by_path: dict[str, ModuleRecord],
        tagger: HeuristicTagger,
        model_name: str,
        modality: str,
        input_shape: list[int] | None,
    ) -> tuple[list[TraceCandidate], list[dict[str, Any]], list[dict[str, Any]]]:
        backbone = f"backbone_{modality}"
        shrinker = f"shrinker_{modality}"
        included_paths = [
            path
            for path in (backbone, shrinker, "cls_head", "reg_head", "dir_head")
            if path in records_by_path
        ]
        ignored_paths = [
            path
            for path in ("cls_head", "reg_head", "dir_head")
            if path in records_by_path
        ]
        top_skip_paths = self._top_skipped_paths(records=records, included_paths=included_paths)
        skipped = [
            _skip_ref(tagger, records_by_path[path], model_name=model_name)
            for path in top_skip_paths
        ]

        rejected = []
        if backbone not in records_by_path:
            rejected.append(
                {
                    "candidate_id": f"heter_baseline_dense_{modality}",
                    "status": "rejected",
                    "failed_at": "module_tree_scan",
                    "error": f"missing {backbone}",
                    "suggested_override": "provide a manual dense entry or model-specific boundary override",
                }
            )
        if not included_paths:
            rejected.append(
                {
                    "candidate_id": "heter_baseline_dense_core",
                    "status": "rejected",
                    "failed_at": "candidate_selection",
                    "error": "no backbone/shrinker/head dense path was found",
                    "suggested_override": "review module names or add a detector plugin",
                }
            )

        candidates = []
        if included_paths:
            candidates.append(
                TraceCandidate(
                    candidate_id=f"heter_baseline_dense_{modality}",
                    entry="post_scatter_bev",
                    input_shape=input_shape,
                    included_modules=included_paths,
                    ignored_layers=ignored_paths,
                    skipped_subgraphs=[item["name"] for item in skipped],
                    confidence="medium",
                    selection_reason=(
                        "HeterModelBaseline dense path selected from module tree: "
                        f"{backbone} -> {shrinker if shrinker in included_paths else 'heads'}."
                    ),
                    wrapper_kind="heter_baseline_dense_path",
                )
            )
        return candidates, skipped, rejected

    def _find_generic(
        self,
        *,
        records: list[ModuleRecord],
        records_by_path: dict[str, ModuleRecord],
        tagger: HeuristicTagger,
        model_name: str,
        input_shape: list[int] | None,
    ) -> tuple[list[TraceCandidate], list[dict[str, Any]], list[dict[str, Any]]]:
        skipped_paths = self._top_skipped_paths(records=records, included_paths=[])
        skipped = [
            _skip_ref(tagger, records_by_path[path], model_name=model_name)
            for path in skipped_paths
        ]
        included_paths = [
            record.path
            for record in records
            if record.path
            and "dense_path_candidate" in record.tags
            and record.depth <= 2
            and not _is_under_any(record.path, skipped_paths)
        ]
        ignored_paths = [
            record.path
            for record in records
            if record.path
            and "ignored_candidate" in record.tags
            and not _is_under_any(record.path, skipped_paths)
        ]
        rejected = []
        candidates = []
        if included_paths:
            candidates.append(
                TraceCandidate(
                    candidate_id="generic_dense_path",
                    entry="unknown_dense_entry",
                    input_shape=input_shape,
                    included_modules=_sort_paths(included_paths),
                    ignored_layers=_sort_paths(ignored_paths),
                    skipped_subgraphs=[item["name"] for item in skipped],
                    confidence="medium",
                    selection_reason="generic dense path hints from module names",
                    wrapper_kind="module_path",
                )
            )
        else:
            rejected.append(
                {
                    "candidate_id": "generic_dense_path",
                    "status": "rejected",
                    "failed_at": "candidate_selection",
                    "error": "generic detector found no dense path candidate",
                    "suggested_override": "add a detector plugin or provide a manual TraceAdapter fallback",
                }
            )
        return candidates, skipped, rejected


class GeneratedTraceWrapper(nn.Module):
    """Wrapper generated from a TraceCandidate.

    First version supports the common BEV dense path:
    backbone(dict I/O) -> optional shrinker -> output heads.
    """

    def __init__(self, full_model: nn.Module, candidate: TraceCandidate | dict[str, Any]):
        super().__init__()
        data = candidate.to_dict() if isinstance(candidate, TraceCandidate) else dict(candidate)
        self.candidate = data
        self.wrapper_kind = str(data.get("wrapper_kind") or "module_path")
        included = [str(item) for item in _as_list(data.get("included_modules"))]
        ignored = set(str(item) for item in _as_list(data.get("ignored_layers")))
        body_paths = [
            path
            for path in included
            if path not in ignored and "head" not in path.lower()
        ]
        head_paths = [
            path
            for path in included
            if path in ignored or "head" in path.lower()
        ]
        self.body_path_names = list(body_paths)
        self.head_path_names = list(head_paths)
        self.body = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        for path in body_paths:
            self.body[self._alias(path)] = _get_module_by_path(full_model, path)
        for path in head_paths:
            self.heads[self._alias(path)] = _get_module_by_path(full_model, path)

    @staticmethod
    def _alias(path: str) -> str:
        return path.replace(".", "__")

    @staticmethod
    def _extract_dense_feature(value: Any) -> Any:
        if isinstance(value, dict):
            for key in ("spatial_features_2d", "spatial_features", "bev_feature", "features"):
                if key in value:
                    return value[key]
        return value

    def _run_module(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        try:
            y = module({"spatial_features": x})
        except Exception:
            y = module(x)
        return self._extract_dense_feature(y)

    def forward(self, spatial_features: torch.Tensor):
        feat = spatial_features
        for path in self.body_path_names:
            feat = self._run_module(self.body[self._alias(path)], feat)
        outs = []
        for path in self.head_path_names:
            outs.append(self.heads[self._alias(path)](feat))
        if outs:
            return tuple(outs)
        return feat


class WrapperSynthesizer:
    """Build executable wrappers from trace candidates."""

    def synthesize(
        self,
        full_model: nn.Module,
        candidate: TraceCandidate | dict[str, Any],
    ) -> nn.Module:
        return GeneratedTraceWrapper(full_model, candidate)


class BoundaryValidator:
    """Validate generated wrapper candidates with Stage1 dry-run gates."""

    def validate(
        self,
        wrapper: nn.Module,
        example_input: torch.Tensor,
        *,
        ignored_layers: list[nn.Module] | None = None,
        run_prune: bool = True,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "full_model_load_sanity": "not_applicable_wrapper_only",
            "wrapper_forward_dryrun": "pending",
            "output_shape_sanity": "pending",
            "depgraph_build": "pending",
            "prune_dryrun": "pending" if run_prune else "skipped",
            "interface_invariant_check": "pending",
            "latency_coverage_annotation": "trace_net_only",
        }
        with torch.no_grad():
            outputs = wrapper(example_input)
        out_shapes = _tensor_shapes(outputs)
        result["wrapper_forward_dryrun"] = "ok"
        result["output_shape_sanity"] = "ok" if out_shapes else "no_tensor_output"
        result["out_shapes"] = out_shapes
        result["interface_invariant_check"] = "ok" if out_shapes else "needs_review"

        if tp is None:
            result["depgraph_build"] = "unavailable_torch_pruning"
            result["prune_dryrun"] = "unavailable_torch_pruning"
            return result

        ignored_layers = ignored_layers or []
        dg = tp.DependencyGraph().build_dependency(wrapper, example_inputs=example_input)
        groups = list(
            dg.get_all_groups(
                root_module_types=[nn.Conv2d, nn.ConvTranspose2d, nn.Linear],
                ignored_layers=ignored_layers,
            )
        )
        result["depgraph_build"] = "ok"
        result["n_prunable_groups"] = len(groups)

        if not run_prune:
            return result

        wrapper2 = copy.deepcopy(wrapper)
        x2 = example_input.detach().clone()
        ignored2 = [
            _get_module_by_path(wrapper2, name)
            for name, _module in wrapper.named_modules()
            if any(_module is ignored for ignored in ignored_layers)
        ]
        try:
            pruner = tp.pruner.MetaPruner(
                wrapper2,
                x2,
                importance=tp.importance.MagnitudeImportance(p=1),
                pruning_ratio=0.5,
                round_to=32,
                global_pruning=False,
                iterative_steps=1,
                ignored_layers=ignored2,
            )
            pruner.step()
            with torch.no_grad():
                pruned_outputs = wrapper2(x2)
            result["prune_dryrun"] = "ok"
            result["pruned_out_shapes"] = _tensor_shapes(pruned_outputs)
        except Exception as e:  # noqa: BLE001
            result["prune_dryrun"] = "fail"
            result["prune_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return result


def _module_ref(records_by_path: dict[str, ModuleRecord], path: str, role: str, reason: str) -> dict[str, Any]:
    record = records_by_path.get(path)
    if record is None:
        return {"name": path, "role": role, "reason": reason}
    return {
        "name": _module_path(path),
        "type": record.class_name,
        "role": role,
        "param_count": record.params_recursive,
        "tags": list(record.tags),
        "reason": reason,
    }


def _skip_ref(tagger: HeuristicTagger, record: ModuleRecord, *, model_name: str) -> dict[str, Any]:
    skip_type, blocker, gate = tagger.skipped_type(record, model_name=model_name)
    return {
        "name": _module_path(record.path),
        "type": skip_type,
        "description": f"{_module_path(record.path)} ({record.class_name}) auto-skipped by module-tree heuristic",
        "full_model_verdict_blocker": bool(blocker),
        "blocker_gate": gate,
        "source": "trace_boundary_detector.module_tree",
        "tags": list(record.tags),
        "reason": "; ".join(record.reasons) or "matched Stage1 skipped boundary heuristic",
    }


def _infer_modality(records_by_path: dict[str, ModuleRecord]) -> str | None:
    for modality in ("m1", "m2", "m3", "m4"):
        if f"backbone_{modality}" in records_by_path:
            return modality
    for path in records_by_path:
        if path.startswith("backbone_") and len(path) >= len("backbone_m1"):
            return path.split("_", 1)[1].split(".", 1)[0]
    return None


def _infer_model_key(model_name: str, config_path: str = "", full_model: nn.Module | None = None) -> str:
    text = f"{model_name} {config_path} {type(full_model).__name__ if full_model is not None else ''}".lower()
    for key in ("where2comm", "v2vnet", "disconet", "attfuse", "fcooper"):
        if key in text:
            return key
    if "att" in text and "fusion" in text:
        return "attfuse"
    return model_name or "unknown"


def _ckpt_status_from_path(ckpt_path: str | Path | None, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if not ckpt_path:
        return "missing_architecture_scan_only"
    return "ok" if Path(ckpt_path).is_file() else "missing_architecture_scan_only"


class TraceBoundaryDetector:
    """Produce TracePlan dictionaries from full-model module trees."""

    def __init__(
        self,
        scanner: ModuleTreeScanner | None = None,
        tagger: HeuristicTagger | None = None,
        path_finder: DensePathFinder | None = None,
    ):
        self.scanner = scanner or ModuleTreeScanner()
        self.tagger = tagger or HeuristicTagger()
        self.path_finder = path_finder or DensePathFinder()

    def detect(
        self,
        full_model: nn.Module,
        *,
        model_name: str,
        config_path: str = "",
        ckpt_path: str = "",
        ckpt_status: str | None = None,
        input_shape: Iterable[int] | None = None,
        manual_override_used: bool = False,
    ) -> dict[str, Any]:
        records = self.tagger.tag_many(self.scanner.scan(full_model))
        records_by_path = {record.path: record for record in records}
        model_key = _infer_model_key(model_name, config_path, full_model)
        candidates, skipped, rejected = self.path_finder.find(
            records=records,
            records_by_path=records_by_path,
            tagger=self.tagger,
            model_name=model_key,
            input_shape=input_shape,
        )
        if model_key in HETER_BASELINE_MODELS or _infer_modality(records_by_path):
            return self._plan_from_candidates(
                records=records,
                records_by_path=records_by_path,
                full_model=full_model,
                model_name=model_key,
                config_path=config_path,
                ckpt_path=ckpt_path,
                ckpt_status=_ckpt_status_from_path(ckpt_path, ckpt_status),
                manual_override_used=manual_override_used,
                detector_name="TraceBoundaryDetector.heter_baseline_v1",
                candidates=candidates,
                skipped=skipped,
                rejected=rejected,
            )
        return self._plan_from_candidates(
            records=records,
            records_by_path=records_by_path,
            full_model=full_model,
            model_name=model_key,
            config_path=config_path,
            ckpt_path=ckpt_path,
            ckpt_status=_ckpt_status_from_path(ckpt_path, ckpt_status),
            manual_override_used=manual_override_used,
            detector_name="TraceBoundaryDetector.generic_v1",
            candidates=candidates,
            skipped=skipped,
            rejected=rejected,
        )

    def _plan_from_candidates(
        self,
        *,
        records: list[ModuleRecord],
        records_by_path: dict[str, ModuleRecord],
        full_model: nn.Module,
        model_name: str,
        config_path: str,
        ckpt_path: str,
        ckpt_status: str,
        manual_override_used: bool,
        detector_name: str,
        candidates: list[TraceCandidate],
        skipped: list[dict[str, Any]],
        rejected: list[dict[str, Any]],
    ) -> dict[str, Any]:
        selected = candidates[0] if candidates else None
        blocking_skip = any(item.get("full_model_verdict_blocker") for item in skipped)
        has_missing_ckpt = "missing" in str(ckpt_status)
        review_reasons = []
        if skipped:
            review_reasons.append("skipped_subgraphs_require_user_review")
        if blocking_skip:
            review_reasons.append("fusion_attention_or_routing_boundary_not_closed")
        if has_missing_ckpt:
            review_reasons.append("missing_checkpoint_architecture_only")
        if rejected:
            review_reasons.append("candidate_rejection_present")

        if selected is None:
            selected_candidate = {
                "candidate_id": "no_dense_candidate",
                "status": "no_candidate",
                "entry": "unknown",
                "input_shape": None,
                "included_modules": [],
                "ignored_layers": [],
                "skipped_subgraphs": [item["name"] for item in skipped],
                "selection_reason": "no candidate generated by DensePathFinder",
                "wrapper_kind": "none",
                "validation": {
                    "full_model_module_tree_scan": "ok",
                    "wrapper_forward_dryrun": "not_run_no_candidate",
                    "depgraph_build": "not_run_no_candidate",
                    "prune_dryrun": "not_run_no_candidate",
                },
            }
            included_paths = []
            ignored_paths = []
            trace_confidence = "low"
        else:
            selected_candidate = selected.to_dict()
            selected_candidate["status"] = "selected" if not rejected else "selected_with_rejections"
            selected_candidate["validation"] = {
                "full_model_module_tree_scan": "ok",
                "wrapper_forward_dryrun": "pending_graph_scan",
                "depgraph_build": "pending_graph_scan",
                "prune_dryrun": "pending_graph_scan",
            }
            included_paths = list(selected.included_modules)
            ignored_paths = list(selected.ignored_layers)
            trace_confidence = selected.confidence if not rejected else "low"

        return {
            "schema": TRACE_PLAN_SCHEMA,
            "model": model_name,
            "model_class": type(full_model).__name__,
            "config_path": config_path,
            "ckpt_path": ckpt_path,
            "ckpt_status": ckpt_status,
            "detector": detector_name,
            "manual_override_used": bool(manual_override_used),
            "trace_confidence": trace_confidence,
            "coverage_scope": "dense_core_only" if skipped else "full_dense_path_candidate",
            "review_required": bool(review_reasons),
            "review_reasons": sorted(set(review_reasons)),
            "candidates": [candidate.to_dict() for candidate in candidates],
            "selected_candidate": selected_candidate,
            "included_modules": [
                _module_ref(records_by_path, path, "included", "selected dense trace path")
                for path in included_paths
            ],
            "ignored_layers": [
                _module_ref(records_by_path, path, "ignored", "head/interface layer kept in forward but ignored by pruning")
                for path in ignored_paths
            ],
            "skipped_subgraphs": skipped,
            "rejected_candidates": rejected,
            "module_inventory": [record.to_dict() for record in records],
        }


def attach_runtime_validation(
    trace_plan: dict[str, Any] | None,
    *,
    forward_status: str | None = None,
    depgraph_status: str | None = None,
    prune_status: str | None = None,
    n_prunable_groups: int | None = None,
    output_shapes: list[list[int]] | None = None,
) -> dict[str, Any] | None:
    """Add graph_scan validation results to a TracePlan dict."""

    if not isinstance(trace_plan, dict):
        return trace_plan
    plan = dict(trace_plan)
    candidate = dict(plan.get("selected_candidate", {}) or {})
    validation = dict(candidate.get("validation", {}) or {})
    if forward_status is not None:
        validation["wrapper_forward_dryrun"] = forward_status
        validation["boundary_validator"] = "BoundaryValidator.graph_scan_integrated_v1"
    if depgraph_status is not None:
        validation["depgraph_build"] = depgraph_status
    if prune_status is not None:
        validation["prune_dryrun"] = prune_status
        validation["interface_invariant_check"] = "ok" if prune_status == "ok" else "needs_review"
    if n_prunable_groups is not None:
        validation["n_prunable_groups"] = int(n_prunable_groups)
    if output_shapes is not None:
        validation["output_shape_sanity"] = "ok" if output_shapes else "no_tensor_output"
        validation["out_shapes"] = output_shapes
    validation.setdefault("latency_coverage_annotation", "trace_net_only")
    candidate["validation"] = validation
    plan["selected_candidate"] = candidate
    return plan


def legacy_trace_plan_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Normalize old adapter manifests into the TracePlan schema."""

    trace = dict(manifest.get("trace", {}) or {})
    model = str(manifest.get("model") or "unknown")
    skipped = []
    for idx, item in enumerate(_as_list(trace.get("skipped_subgraphs"))):
        if isinstance(item, dict):
            skipped.append(dict(item))
    for idx, raw in enumerate(_as_list(trace.get("skipped_modules"))):
        text = str(raw)
        low = text.lower()
        if any(key in low for key in _ATTENTION_HINTS + _ROUTING_HINTS):
            typ, blocker, gate = "attention_or_routing_fusion", True, "attention_fusion_coverage_anchor"
            if any(key in low for key in _ROUTING_HINTS):
                gate = "routing_fusion_coverage_anchor"
        elif any(key in low for key in _FUSION_HINTS):
            if "maxfusion" in low or "max pooling" in low:
                typ, blocker, gate = "fusion_or_alignment", False, "maxfusion_coverage_anchor"
            else:
                typ, blocker, gate = "fusion_or_alignment", True, "routing_fusion_coverage_anchor"
        elif any(key in low for key in _SPARSE_HINTS):
            typ, blocker, gate = "sparse_or_geometry_preprocess", False, "trace_sparse_frontend_boundary"
        else:
            typ, blocker, gate = "custom_untraced_subgraph", True, "custom_subgraph_coverage_gate"
        skipped.append(
            {
                "name": text.split("(", 1)[0].strip() or f"skipped_{idx}",
                "type": typ,
                "description": text,
                "full_model_verdict_blocker": blocker,
                "blocker_gate": gate,
                "source": "legacy_trace_adapter_manifest",
            }
        )

    ckpt_status = str(manifest.get("ckpt_status") or "unknown")
    review_reasons = ["legacy_adapter_boundary_requires_review"]
    if "missing" in ckpt_status:
        review_reasons.append("missing_checkpoint_architecture_only")
    if any(item.get("full_model_verdict_blocker") for item in skipped):
        review_reasons.append("fusion_attention_or_routing_boundary_not_closed")
    return {
        "schema": TRACE_PLAN_SCHEMA,
        "model": model,
        "model_class": manifest.get("model_class") or "",
        "config_path": manifest.get("config") or manifest.get("config_path") or "",
        "ckpt_path": manifest.get("ckpt") or manifest.get("ckpt_path") or "",
        "ckpt_status": ckpt_status,
        "detector": "legacy_trace_adapter_manifest_normalizer",
        "manual_override_used": True,
        "trace_confidence": "low" if "missing" in ckpt_status else "medium",
        "coverage_scope": "dense_core_only" if skipped else "trace_adapter_scope",
        "review_required": True,
        "review_reasons": sorted(set(review_reasons)),
        "selected_candidate": {
            "candidate_id": "legacy_trace_adapter_boundary",
            "status": "selected_legacy",
            "entry": "trace_adapter_declared_entry",
            "input_shape": trace.get("entry_shape"),
            "included_modules": [],
            "ignored_layers": [],
            "skipped_subgraphs": [item.get("name") for item in skipped],
            "selection_reason": "normalized from existing TraceAdapter manifest fields",
            "validation": {
                "full_model_module_tree_scan": "not_available_legacy_manifest",
                "wrapper_forward_dryrun": "already_run_by_graph_scan" if manifest.get("scan_status") else "unknown",
                "depgraph_build": "already_run_by_graph_scan" if manifest.get("view_b1_prune_groups") else "unknown",
                "prune_dryrun": (
                    str((manifest.get("checks", {}) or {}).get("dryrun_prune05", {}).get("status"))
                    if isinstance(manifest.get("checks"), dict)
                    else "unknown"
                ),
            },
        },
        "included_modules": [],
        "ignored_layers": [],
        "skipped_subgraphs": skipped,
        "rejected_candidates": [],
        "module_inventory": [],
    }


__all__ = [
    "TRACE_PLAN_SCHEMA",
    "ModuleRecord",
    "TraceCandidate",
    "ModuleTreeScanner",
    "HeuristicTagger",
    "DensePathFinder",
    "GeneratedTraceWrapper",
    "WrapperSynthesizer",
    "BoundaryValidator",
    "TraceBoundaryDetector",
    "attach_runtime_validation",
    "legacy_trace_plan_from_manifest",
]
