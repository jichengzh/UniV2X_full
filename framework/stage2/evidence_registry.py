"""Stage2 evidence registry contract.

The registry is an internal cost-model input contract. It is deliberately
separate from Stage2Input: users still provide only the Stage1 manifest and
classification report at the public CLI boundary.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


STAGE2_EVIDENCE_REGISTRY_SCHEMA = "stage2_evidence_registry_v1"
MEASURED_HARDWARE_BACKEND = "h800_tvm"
MEASURED_ENERGY_BACKEND = "h800_tvm_power_telemetry"
HISTORICAL_TRT_BACKEND = "historical_trt"
VALID_STATUSES = {
    "measured",
    "imported",
    "predicted",
    "demo",
    "estimated",
    "historical",
    "proxy",
    "not_available",
    "not_done",
}
HARDWARE_EVIDENCE_SOURCES = {"latency_lut", "quant_evidence", "energy_lut"}


class EvidenceRegistryError(ValueError):
    """Raised when Stage2 evidence is missing, unsafe, or out of scope."""


@dataclass(frozen=True)
class EvidenceSource:
    name: str
    path: Path | None
    measurement_status: str
    backend: str
    scope: Any
    provenance: str
    coverage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        name: str,
        data: dict[str, Any],
        *,
        base_dir: Path,
    ) -> "EvidenceSource":
        status = str(data.get("measurement_status", "not_available"))
        if status not in VALID_STATUSES:
            raise EvidenceRegistryError(f"{name}: unknown measurement_status {status!r}")

        raw_path = data.get("path")
        path = None
        if raw_path:
            path = Path(str(raw_path))
            if not path.is_absolute():
                path = base_dir / path

        backend = str(data.get("backend", "unknown"))
        if backend == HISTORICAL_TRT_BACKEND and status == "measured":
            raise EvidenceRegistryError(
                f"{name}: historical_trt cannot be measured evidence"
            )
        if (
            name == "energy_lut"
            and status == "measured"
            and backend != MEASURED_ENERGY_BACKEND
        ):
            raise EvidenceRegistryError(
                f"{name}: measured evidence must use backend={MEASURED_ENERGY_BACKEND}"
            )
        if (
            name in HARDWARE_EVIDENCE_SOURCES
            and name != "energy_lut"
            and status == "measured"
            and backend != MEASURED_HARDWARE_BACKEND
        ):
            raise EvidenceRegistryError(
                f"{name}: measured evidence must use backend={MEASURED_HARDWARE_BACKEND}"
            )

        coverage = {
            "expected_cells": int((data.get("coverage") or {}).get("expected_cells", 0)),
            "measured_cells": int((data.get("coverage") or {}).get("measured_cells", 0)),
            "failed_cells": int((data.get("coverage") or {}).get("failed_cells", 0)),
            "proxy_cells": int((data.get("coverage") or {}).get("proxy_cells", 0)),
        }
        metadata = {
            key: value
            for key, value in data.items()
            if key
            not in {
                "path",
                "measurement_status",
                "backend",
                "scope",
                "provenance",
                "coverage",
            }
        }
        return cls(
            name=name,
            path=path,
            measurement_status=status,
            backend=backend,
            scope=data.get("scope"),
            provenance=str(data.get("provenance", "")),
            coverage=coverage,
            metadata=metadata,
        )

    @property
    def available(self) -> bool:
        return (
            self.path is not None
            and self.measurement_status not in {"not_available", "not_done"}
        )

    @property
    def promotable_to_measured(self) -> bool:
        return (
            self.measurement_status == "measured"
            and self.backend == MEASURED_HARDWARE_BACKEND
        )

    def require_available(self) -> None:
        if not self.available:
            raise EvidenceRegistryError(f"{self.name}: required evidence is missing")
        if self.path is not None and not self.path.exists():
            raise EvidenceRegistryError(f"{self.name}: path does not exist: {self.path}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": None if self.path is None else str(self.path),
            "measurement_status": self.measurement_status,
            "backend": self.backend,
            "scope": self.scope,
            "provenance": self.provenance,
            "coverage": dict(self.coverage),
            **self.metadata,
        }


@dataclass(frozen=True)
class DSQueryResult:
    ds_mid: float
    ds_low: float
    ds_high: float
    mode: str
    flags: list[str]
    prediction_chain: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "DS_mid": self.ds_mid,
            "DS_low": self.ds_low,
            "DS_high": self.ds_high,
            "mode": self.mode,
            "flags": list(self.flags),
            "prediction_chain": self.prediction_chain,
        }


@dataclass(frozen=True)
class Stage2CostInputs:
    latency_lut: Any
    ap_model: Any
    q_lookup: Any
    energy_lut: EvidenceSource
    downstream_objective: EvidenceSource
    energy_claim_allowed: bool


class _DownstreamDSMap:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._values: dict[tuple[float, float], float] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                latency = float(row["latency_ms"])
                ap50 = float(row["ap50"])
                self._values[(ap50, latency)] = float(row["honest_DS"])
        if not self._values:
            raise EvidenceRegistryError(f"downstream_objective: empty DS map {path}")
        self.ap_axis = sorted({ap for ap, _ in self._values})
        self.latency_axis = sorted({lat for _, lat in self._values})

    @staticmethod
    def _bracket(axis: list[float], value: float, label: str) -> tuple[float, float]:
        if value < axis[0] or value > axis[-1]:
            raise EvidenceRegistryError(
                f"{label}={value} outside measured rectangle [{axis[0]}, {axis[-1]}]"
            )
        if value in axis:
            return value, value
        lo = max(item for item in axis if item <= value)
        hi = min(item for item in axis if item >= value)
        return lo, hi

    @staticmethod
    def _lerp(lo: float, hi: float, value: float, ylo: float, yhi: float) -> float:
        if lo == hi:
            return ylo
        t = (value - lo) / (hi - lo)
        return (1 - t) * ylo + t * yhi

    def predict(self, ap50: float, latency_ms: float) -> float:
        ap_lo, ap_hi = self._bracket(self.ap_axis, ap50, "ap50")
        lat_lo, lat_hi = self._bracket(self.latency_axis, latency_ms, "latency_ms")
        q11 = self._values[(ap_lo, lat_lo)]
        q12 = self._values[(ap_lo, lat_hi)]
        q21 = self._values[(ap_hi, lat_lo)]
        q22 = self._values[(ap_hi, lat_hi)]
        low_ap = self._lerp(lat_lo, lat_hi, latency_ms, q11, q12)
        high_ap = self._lerp(lat_lo, lat_hi, latency_ms, q21, q22)
        return self._lerp(ap_lo, ap_hi, ap50, low_ap, high_ap)


@dataclass(frozen=True)
class Stage2EvidenceRegistry:
    schema: str
    model: str
    hardware_target: dict[str, Any]
    manifest: EvidenceSource
    latency_lut: EvidenceSource
    ap_anchors: EvidenceSource
    quant_evidence: EvidenceSource
    energy_lut: EvidenceSource
    downstream_objective: EvidenceSource
    unsupported_conclusions_base: list[str] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> "Stage2EvidenceRegistry":
        registry_path = Path(path)
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        return cls.from_dict(data, base_dir=registry_path.parent)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        base_dir: str | Path = ".",
    ) -> "Stage2EvidenceRegistry":
        schema = str(data.get("schema", ""))
        if schema != STAGE2_EVIDENCE_REGISTRY_SCHEMA:
            raise EvidenceRegistryError(f"unexpected schema: {schema!r}")
        base = Path(base_dir)
        return cls(
            schema=schema,
            model=str(data.get("model", "unknown")),
            hardware_target=dict(data.get("hardware_target", {})),
            manifest=EvidenceSource.from_dict(
                "manifest", data.get("manifest", {}), base_dir=base
            ),
            latency_lut=EvidenceSource.from_dict(
                "latency_lut", data.get("latency_lut", {}), base_dir=base
            ),
            ap_anchors=EvidenceSource.from_dict(
                "ap_anchors", data.get("ap_anchors", {}), base_dir=base
            ),
            quant_evidence=EvidenceSource.from_dict(
                "quant_evidence", data.get("quant_evidence", {}), base_dir=base
            ),
            energy_lut=EvidenceSource.from_dict(
                "energy_lut", data.get("energy_lut", {}), base_dir=base
            ),
            downstream_objective=EvidenceSource.from_dict(
                "downstream_objective",
                data.get("downstream_objective", {}),
                base_dir=base,
            ),
            unsupported_conclusions_base=list(data.get("unsupported_conclusions", [])),
        )

    def source(self, name: str) -> EvidenceSource:
        try:
            return getattr(self, name)
        except AttributeError as exc:
            raise EvidenceRegistryError(f"unknown evidence source: {name}") from exc

    @property
    def energy_claim_allowed(self) -> bool:
        if self.energy_lut.measurement_status != "measured":
            return False
        if self.energy_lut.backend != MEASURED_ENERGY_BACKEND:
            return False
        if self.energy_lut.path is None or not self.energy_lut.path.exists():
            return False
        try:
            from framework.stage2.lut_productization import (
                energy_claim_allowed_from_rows,
                read_jsonl,
            )

            return energy_claim_allowed_from_rows(read_jsonl(self.energy_lut.path))
        except ValueError:
            return False
        except OSError:
            return False

    @property
    def unsupported_conclusions(self) -> list[str]:
        unsupported = set(self.unsupported_conclusions_base)
        if not self.energy_claim_allowed:
            unsupported.add("energy_improvement_without_energy_lut")
        return sorted(unsupported)

    def require_search_ready(self) -> None:
        self.latency_lut.require_available()
        self.ap_anchors.require_available()

    def load_latency_lut(self):
        from framework.search_three_arm import LatencyLUT

        self.latency_lut.require_available()
        return LatencyLUT(self.latency_lut.path)

    def load_ap_model(self):
        from framework.search_three_arm import APModel

        self.ap_anchors.require_available()
        return APModel(self.ap_anchors.path)

    def load_q_lookup(self):
        from framework.search_three_arm import QLookup

        if not self.quant_evidence.available:
            return QLookup()
        return QLookup(self.quant_evidence.path)

    def load_cost_inputs(self) -> Stage2CostInputs:
        return Stage2CostInputs(
            latency_lut=self.load_latency_lut(),
            ap_model=self.load_ap_model(),
            q_lookup=self.load_q_lookup(),
            energy_lut=self.energy_lut,
            downstream_objective=self.downstream_objective,
            energy_claim_allowed=self.energy_claim_allowed,
        )

    def _validate_downstream_scope(self) -> None:
        scope = self.downstream_objective.scope
        if not isinstance(scope, dict):
            raise EvidenceRegistryError("downstream_objective: scope must be a dict")
        if not (
            str(scope.get("model")) == "codriving"
            and str(scope.get("town")) == "Town05"
            and str(scope.get("routes")) == "clean6"
        ):
            raise EvidenceRegistryError(
                "downstream_objective is only valid for CoDriving/Town05/clean6"
            )

    @staticmethod
    def _status_label(status: str) -> str:
        return "measured" if status == "measured" else "predicted"

    def query_downstream_ds(
        self,
        *,
        ap50: float,
        latency_ms: float,
        ap_input_status: str,
        latency_input_status: str,
        ap_interval: tuple[float, float] | None = None,
        latency_interval_ms: tuple[float, float] | None = None,
    ) -> DSQueryResult:
        self.downstream_objective.require_available()
        self._validate_downstream_scope()
        ds_map = _DownstreamDSMap(self.downstream_objective.path)
        ds_mid = ds_map.predict(float(ap50), float(latency_ms))

        ap_values = [float(ap50)]
        if ap_interval is not None:
            ap_values.extend([float(ap_interval[0]), float(ap_interval[1])])
        latency_values = [float(latency_ms)]
        if latency_interval_ms is not None:
            latency_values.extend(
                [float(latency_interval_ms[0]), float(latency_interval_ms[1])]
            )
        samples = [
            ds_map.predict(ap_value, latency_value)
            for ap_value in ap_values
            for latency_value in latency_values
        ]

        flags: set[str] = set()
        measured_inputs = ap_input_status == "measured" and latency_input_status == "measured"
        mode = "constraint" if measured_inputs else "report_only"
        if not measured_inputs:
            flags.add("predicted_ds_report_only")
            flags.add("downstream_validation_required")

        cliff = self.downstream_objective.metadata.get("cliff_band_ms", [600, 650])
        if latency_interval_ms is not None:
            lo, hi = sorted([float(latency_interval_ms[0]), float(latency_interval_ms[1])])
            cliff_lo, cliff_hi = float(cliff[0]), float(cliff[1])
            if lo < cliff_hi and hi > cliff_lo:
                flags.add("uncertain_due_to_cliff")

        prediction_chain = (
            f"{self._status_label(ap_input_status)}_ap+"
            f"{self._status_label(latency_input_status)}_latency->measured_ds_lut"
        )
        return DSQueryResult(
            ds_mid=round(ds_mid, 6),
            ds_low=round(min(samples), 6),
            ds_high=round(max(samples), 6),
            mode=mode,
            flags=sorted(flags),
            prediction_chain=prediction_chain,
        )

    def coverage_summary(self) -> dict[str, Any]:
        names = [
            "latency_lut",
            "ap_anchors",
            "quant_evidence",
            "energy_lut",
            "downstream_objective",
        ]
        summary: dict[str, Any] = {}
        totals = {
            "expected_cells": 0,
            "measured_cells": 0,
            "failed_cells": 0,
            "proxy_cells": 0,
        }
        for name in names:
            coverage = dict(self.source(name).coverage)
            summary[name] = coverage
            for key in totals:
                totals[key] += int(coverage.get(key, 0))
        summary["total_expected_cells"] = totals["expected_cells"]
        summary["total_measured_cells"] = totals["measured_cells"]
        summary["total_failed_cells"] = totals["failed_cells"]
        summary["total_proxy_cells"] = totals["proxy_cells"]
        return summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "model": self.model,
            "hardware_target": dict(self.hardware_target),
            "manifest": self.manifest.to_dict(),
            "latency_lut": self.latency_lut.to_dict(),
            "ap_anchors": self.ap_anchors.to_dict(),
            "quant_evidence": self.quant_evidence.to_dict(),
            "energy_lut": self.energy_lut.to_dict(),
            "downstream_objective": self.downstream_objective.to_dict(),
            "unsupported_conclusions": self.unsupported_conclusions,
            "coverage_summary": self.coverage_summary(),
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _count_json_rows(path: Path, keys: tuple[str, ...]) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return len(value)
    return 0


def _count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _coverage(
    *,
    expected: int,
    measured: int,
    failed: int = 0,
    proxy: int = 0,
) -> dict[str, int]:
    return {
        "expected_cells": int(expected),
        "measured_cells": int(measured),
        "failed_cells": int(failed),
        "proxy_cells": int(proxy),
    }


def build_default_registry_dict(
    model: str,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a repo-local Pyramid/CoDriving evidence registry fixture."""

    base = Path(root) if root is not None else _repo_root()
    ds_map = base / "multi_agent/real_test/ds_ap_latency_all_measured.csv"
    if model == "pyramid_lidar":
        manifest = base / "framework/partitions/pyramid_lidar_partition.yaml"
        latency = base / "results/latency_lut_pyramid.json"
        ap = base / "results/ap70_model_pyramid.json"
        q = base / "results/latency_lut_pyramid_q.json"
        latency_count = _count_json_rows(latency, ("widths", "grid"))
        ap_count = _count_json_rows(ap, ("table", "anchors", "fit_points"))
        q_count = _count_json_rows(q, ("widths", "grid"))
        downstream = {
            "path": None,
            "measurement_status": "not_available",
            "backend": "not_available",
            "scope": {
                "model": "pyramid_lidar",
                "reason": "CoDriving/Town05/clean6 DS map is not cross-model evidence",
            },
            "provenance": "not_collected_for_pyramid",
            "mode_policy": "blocked_cross_model",
            "coverage": _coverage(expected=0, measured=0),
        }
        return {
            "schema": STAGE2_EVIDENCE_REGISTRY_SCHEMA,
            "model": model,
            "hardware_target": {"name": "H800 Hopper", "backend": MEASURED_HARDWARE_BACKEND},
            "manifest": {
                "path": str(manifest),
                "scope": "rsu_dense_core",
                "provenance": "stage1_partition_manifest",
            },
            "latency_lut": {
                "path": str(latency),
                "measurement_status": "measured",
                "backend": MEASURED_HARDWARE_BACKEND,
                "scope": "dense_core",
                "provenance": "H800 TVM Relax/MetaSchedule latency LUT",
                "coverage": _coverage(expected=latency_count, measured=latency_count),
            },
            "ap_anchors": {
                "path": str(ap),
                "measurement_status": "measured",
                "backend": "model_eval",
                "scope": "model_accuracy",
                "provenance": "Pyramid AP70 anchor/model file",
                "metric": "AP70",
                "coverage": _coverage(expected=ap_count, measured=ap_count),
            },
            "quant_evidence": {
                "path": str(q),
                "measurement_status": "historical",
                "backend": HISTORICAL_TRT_BACKEND,
                "scope": "backbone_only",
                "provenance": "historical RTX 4090 TRT quant context; not H800 measured",
                "coverage": _coverage(expected=q_count, measured=0, proxy=q_count),
            },
            "energy_lut": {
                "path": None,
                "measurement_status": "not_available",
                "backend": "not_available",
                "scope": "dense_core",
                "provenance": "not_collected_yet",
                "coverage": _coverage(expected=latency_count, measured=0),
            },
            "downstream_objective": downstream,
            "unsupported_conclusions": [
                "historical_trt_as_h800_measured_quant_evidence",
                "cross_model_ds_map_for_pyramid",
            ],
        }
    if model == "codriving":
        manifest = base / "framework/partitions/codriving_partition.yaml"
        latency = base / "results/latency_lut_codriving.json"
        ap = base / "results/ap70_model_codriving.json"
        latency_count = _count_json_rows(latency, ("widths", "grid"))
        ap_count = _count_json_rows(ap, ("table", "anchors", "fit_points"))
        ds_count = _count_csv_rows(ds_map)
        return {
            "schema": STAGE2_EVIDENCE_REGISTRY_SCHEMA,
            "model": model,
            "hardware_target": {"name": "H800 Hopper", "backend": MEASURED_HARDWARE_BACKEND},
            "manifest": {
                "path": str(manifest),
                "scope": "rsu_dense_core",
                "provenance": "stage1_partition_manifest",
            },
            "latency_lut": {
                "path": str(latency),
                "measurement_status": "estimated",
                "backend": MEASURED_HARDWARE_BACKEND,
                "scope": "dense_core",
                "provenance": "CoDriving H800 TVM LUT with real base/p50 and estimated p25/p75",
                "coverage": _coverage(
                    expected=latency_count,
                    measured=2,
                    proxy=max(latency_count - 2, 0),
                ),
            },
            "ap_anchors": {
                "path": str(ap),
                "measurement_status": "measured",
                "backend": "model_eval",
                "scope": "model_accuracy",
                "provenance": "CoDriving iso-budget AP70 anchors",
                "metric": "AP70",
                "coverage": _coverage(expected=ap_count, measured=ap_count),
            },
            "quant_evidence": {
                "path": None,
                "measurement_status": "not_done",
                "backend": "not_done",
                "scope": "dense_core",
                "provenance": "CoDriving Q evidence not registered as formal H800 evidence",
                "coverage": _coverage(expected=0, measured=0),
            },
            "energy_lut": {
                "path": None,
                "measurement_status": "not_available",
                "backend": "not_available",
                "scope": "dense_core",
                "provenance": "not_collected_yet",
                "coverage": _coverage(expected=latency_count, measured=0),
            },
            "downstream_objective": {
                "path": str(ds_map),
                "measurement_status": "measured",
                "backend": "closed_loop_sim",
                "scope": {
                    "model": "codriving",
                    "town": "Town05",
                    "routes": "clean6",
                    "traffic": "full_traffic_1",
                },
                "provenance": "CoDriving AP x latency measured DS map v2",
                "mode_policy": "report_only_unless_measured_inputs",
                "coverage": _coverage(expected=ds_count, measured=ds_count),
                "cliff_band_ms": [600, 650],
            },
            "unsupported_conclusions": [],
        }
    raise EvidenceRegistryError(f"unsupported default registry model: {model}")


def build_default_registry(
    model: str,
    *,
    root: str | Path | None = None,
) -> Stage2EvidenceRegistry:
    return Stage2EvidenceRegistry.from_dict(
        build_default_registry_dict(model, root=root),
        base_dir=Path(root) if root is not None else _repo_root(),
    )


def write_default_registry(
    model: str,
    out_path: str | Path,
    *,
    root: str | Path | None = None,
) -> Stage2EvidenceRegistry:
    data = build_default_registry_dict(model, root=root)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return Stage2EvidenceRegistry.from_file(out)


__all__ = [
    "DSQueryResult",
    "EvidenceRegistryError",
    "EvidenceSource",
    "STAGE2_EVIDENCE_REGISTRY_SCHEMA",
    "Stage2CostInputs",
    "Stage2EvidenceRegistry",
    "build_default_registry",
    "build_default_registry_dict",
    "write_default_registry",
]
