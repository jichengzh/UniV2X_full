"""Schema-driven structure identity for Stage5 search genomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


MODEL_WIDTH_SCHEMAS: dict[str, tuple[str, ...]] = {
    "pyramid": ("w0", "w1", "w2"),
    "codriving": ("w0", "w1", "w2"),
    "fcooper": (
        "backbone.s0",
        "backbone.s1",
        "backbone.s2",
        "neck.deblock",
        "neck.output",
    ),
}


@dataclass(frozen=True)
class StructureIdentity:
    model: str
    group_id: str
    width: tuple[int, ...]
    width_schema: tuple[str, ...]
    structure_widths: dict[str, int]


def width_schema_for_model(model: str) -> tuple[str, ...]:
    normalized = str(model).lower()
    try:
        return MODEL_WIDTH_SCHEMAS[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported target model: {model}") from exc


def canonical_group_id(
    model: str, width: Sequence[int], width_schema: Sequence[str] | None = None
) -> str:
    normalized = str(model).lower()
    schema = tuple(width_schema or width_schema_for_model(normalized))
    values = tuple(int(value) for value in width)
    if len(values) != len(schema):
        raise ValueError("structure identity width/schema cardinality mismatch")
    if normalized in {"pyramid", "codriving"}:
        return f"{normalized}|{'x'.join(map(str, values))}"
    return normalized + "|" + "|".join(
        f"{name}={value}" for name, value in zip(schema, values)
    )


def validate_structure_identity(row: Mapping[str, Any]) -> StructureIdentity:
    model = str(row.get("model") or "").lower()
    expected_schema = width_schema_for_model(model)
    supplied_schema = tuple(row.get("width_schema") or expected_schema)
    if supplied_schema != expected_schema:
        raise ValueError(
            f"structure identity width_schema mismatch for {model}: "
            f"{supplied_schema} != {expected_schema}"
        )
    width = tuple(int(value) for value in row.get("width") or [])
    if len(width) != len(expected_schema):
        raise ValueError(
            f"structure identity width/schema cardinality mismatch for {model}"
        )
    structure_widths = {
        str(name): int(value)
        for name, value in zip(expected_schema, width)
    }
    supplied_widths = row.get("structure_widths")
    if supplied_widths is not None and dict(supplied_widths) != structure_widths:
        raise ValueError(f"structure identity named widths mismatch for {model}")
    group_id = str(row.get("group_id") or "")
    expected_group_id = canonical_group_id(model, width, expected_schema)
    if group_id != expected_group_id:
        raise ValueError(
            f"structure identity group_id mismatch: {group_id} != {expected_group_id}"
        )
    return StructureIdentity(
        model=model,
        group_id=group_id,
        width=width,
        width_schema=expected_schema,
        structure_widths=structure_widths,
    )
