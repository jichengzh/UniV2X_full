# Three Comparison Tables on One Page — Design

## Goal

Place the TensorRT/H800, TensorRT/Orin, and TVM/H800 comparison tables on one
physical paper page while preserving three independent table numbers and
cross-references.

## Layout

Use one page-level `table*` float containing three consecutive tabular blocks.
Each block has its own `\caption` and `\label`, so the paper continues to expose
Table 1, Table 2, and Table 3 even though LaTeX places them as one indivisible
float. In accordance with the selected AAAI format, each caption remains below
its table. Every caption is a single-line title; measurement-scope explanations
remain in the surrounding prose. The AAAI class controls caption alignment and
Arabic table numbering.

## Table Content

- Table 1 remains the existing TensorRT/H800 comparison.
- Table 2 remains the measured TensorRT/Orin multiscale-backbone comparison.
- Table 3 uses the TVM/H800 `DeltaAP_max=0.10` results from sections 11.5 and
  14.5 of the 31-series handoff document.
- All three tables retain the same five routes: Original/default, Compression
  only, Schedule only, Compress to Tune, and Joint.
- Table 3 leaves Original/default empty because the user does not want the
  native PyTorch/cuDNN baseline used in the TVM-only comparison.
- Table 3 leaves Pyramid Schedule only empty because the source records an
  FP32 numerical-contract failure; the caption discloses this failure.
- F-Cooper remains reserved with `--` wherever measurements are unavailable.
- Every displayed numeric cell is rounded to two digits after the decimal
  point.

## Validation

Verify the three labels are unique and numbered consecutively, compare all TVM
cells against the handoff source after two-decimal rounding, compile the paper,
and visually confirm that all three tables appear together, all three captions
fit on one line, and no content is clipped or overlapping.
