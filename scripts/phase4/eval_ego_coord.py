"""阶段 4.12 — Plan C: ego-coord 独立 IoU eval

绕过 nuScenes 评估器,直接拿原始 det.json + val pkl GT,
把 detection 从 global → ego 后,在 ego 坐标系下计算 BEV IoU + 中心距离的 AP。

输入:
- det:   /home/jichengzhi/UniV2X/test/univ2x_sub_vehicle_tiny/<run>/results_nusc_det.json
- gt:    /home/jichengzhi/UniV2X/data/infos/V2X-Seq-SPD-New/vehicle-side/spd_infos_temporal_val.pkl

输出:
- 每类 mAP @ center_dist {0.5,1.0,2.0,4.0} (nuScenes 风格)
- 每类 mAP @ BEV IoU {0.25,0.5}
- overall mAP

用法:
  python scripts/phase4/eval_ego_coord.py \\
      --det test/univ2x_sub_vehicle_tiny/Tue_Apr_28_23_52_53_2026/results_nusc_det.json \\
      --gt  data/infos/V2X-Seq-SPD-New/vehicle-side/spd_infos_temporal_val.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import Polygon

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_ego")


CENTER_DIST_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
IOU_THRESHOLDS = (0.25, 0.5)
CLASS_NAMES = ("car", "bicycle", "pedestrian")


@dataclass(frozen=True)
class Box3D:
    """BEV-relevant box in ego frame: center (x,y,z), size (w,l,h), yaw."""
    x: float
    y: float
    z: float
    w: float
    l: float
    h: float
    yaw: float
    name: str
    score: float = 1.0

    def bev_polygon(self) -> Polygon:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        # local corners: w along x, l along y (mmdet3d LiDAR convention: gt[3]=dx=w, gt[4]=dy=l)
        dx, dy = self.w / 2.0, self.l / 2.0
        local = np.array([
            [dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy],
        ])
        rot = np.array([[c, -s], [s, c]])
        global_xy = local @ rot.T + np.array([self.x, self.y])
        return Polygon(global_xy)


def quat_to_yaw(q: List[float]) -> float:
    """[w,x,y,z] quaternion → yaw around z (mmdet3d convention)."""
    return Quaternion(q).yaw_pitch_roll[0]


def global_to_lidar(
    t_global: np.ndarray,
    yaw_global: float,
    e2g_t: np.ndarray,
    e2g_q: Quaternion,
    l2e_t: np.ndarray,
    l2e_q: Quaternion,
) -> Tuple[np.ndarray, float]:
    """world → ego → lidar (matches mmdet3d gt_boxes frame).

    Pipeline that produced the box (in spd_vehicle_e2e_dataset.lidar_nusc_box_to_global):
        lidar  --rotate(l2e_R) translate(l2e_t)--> ego
        ego    --rotate(e2g_R) translate(e2g_t)--> global
    Inversion:
        global --(p - e2g_t) inv-rot e2g_R--> ego
        ego    --(p - l2e_t) inv-rot l2e_R--> lidar
    """
    p = e2g_q.inverse.rotate(t_global - e2g_t)
    p = l2e_q.inverse.rotate(p - l2e_t)
    yaw = yaw_global - e2g_q.yaw_pitch_roll[0] - l2e_q.yaw_pitch_roll[0]
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    return p, yaw


def bev_iou(a: Box3D, b: Box3D) -> float:
    pa, pb = a.bev_polygon(), b.bev_polygon()
    if not pa.is_valid or not pb.is_valid:
        return 0.0
    inter = pa.intersection(pb).area
    if inter <= 0:
        return 0.0
    union = pa.area + pb.area - inter
    return float(inter / union) if union > 0 else 0.0


def center_dist(a: Box3D, b: Box3D) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def load_detections(path: Path) -> Dict[str, List[Dict]]:
    with path.open() as f:
        d = json.load(f)
    return d["results"]


def load_gt(path: Path) -> Dict[str, Dict]:
    with path.open("rb") as f:
        d = pickle.load(f)
    return {info["token"]: info for info in d["infos"]}


def build_gt_boxes(info: Dict) -> List[Box3D]:
    out: List[Box3D] = []
    boxes = info["gt_boxes"]
    names = info["gt_names"]
    if boxes is None or len(boxes) == 0:
        return out
    valid = info.get("valid_flag")
    for i, (b, n) in enumerate(zip(boxes, names)):
        if valid is not None and not bool(valid[i]):
            continue
        if n not in CLASS_NAMES:
            continue
        x, y, z, dx, dy, dz, yaw = b[:7]
        out.append(Box3D(
            x=float(x), y=float(y), z=float(z),
            w=float(dx), l=float(dy), h=float(dz),
            yaw=float(yaw), name=str(n), score=1.0,
        ))
    return out


def build_det_boxes_in_lidar(dets: List[Dict], info: Dict) -> List[Box3D]:
    e2g_t = np.asarray(info["ego2global_translation"], dtype=np.float64)
    e2g_q = Quaternion(info["ego2global_rotation"])
    l2e_t = np.asarray(info["lidar2ego_translation"], dtype=np.float64)
    l2e_q = Quaternion(info["lidar2ego_rotation"])
    out: List[Box3D] = []
    for d in dets:
        if d["detection_name"] not in CLASS_NAMES:
            continue
        t_g = np.asarray(d["translation"], dtype=np.float64)
        yaw_g = quat_to_yaw(d["rotation"])
        p_l, yaw_l = global_to_lidar(t_g, yaw_g, e2g_t, e2g_q, l2e_t, l2e_q)
        sz = d["size"]
        w, l, h = float(sz[0]), float(sz[1]), float(sz[2])
        out.append(Box3D(
            x=float(p_l[0]), y=float(p_l[1]), z=float(p_l[2]),
            w=w, l=l, h=h, yaw=float(yaw_l),
            name=d["detection_name"], score=float(d["detection_score"]),
        ))
    return out


def match_and_score(
    dets: List[Box3D],
    gts: List[Box3D],
    metric: str,
    threshold: float,
) -> Tuple[List[int], List[float], int]:
    """Greedy matching by descending score. metric in {'dist','iou'}.

    Returns (tp_flags, scores_sorted_desc, num_gt) so caller can build PR curve.
    A detection is TP if there exists an unmatched GT of same class with
    cost <= threshold (dist) or score >= threshold (iou).
    """
    gt_used = [False] * len(gts)
    order = sorted(range(len(dets)), key=lambda i: -dets[i].score)
    tp_flags: List[int] = []
    scores: List[float] = []
    for i in order:
        det = dets[i]
        best_j = -1
        if metric == "dist":
            best_val = float("inf")
            for j, gt in enumerate(gts):
                if gt_used[j] or gt.name != det.name:
                    continue
                d = center_dist(det, gt)
                if d < best_val and d <= threshold:
                    best_val, best_j = d, j
        else:  # iou — pick highest IoU above threshold
            best_val = -1.0
            for j, gt in enumerate(gts):
                if gt_used[j] or gt.name != det.name:
                    continue
                iou = bev_iou(det, gt)
                if iou > best_val and iou >= threshold:
                    best_val, best_j = iou, j
        scores.append(det.score)
        if best_j >= 0:
            gt_used[best_j] = True
            tp_flags.append(1)
        else:
            tp_flags.append(0)
    return tp_flags, scores, sum(1 for g in gts if True)


def compute_ap(tp_flags: List[int], scores: List[float], num_gt: int) -> float:
    """All-point AP (PASCAL VOC style)."""
    if num_gt == 0:
        return 0.0
    if len(tp_flags) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tps = np.asarray(tp_flags)[order]
    fps = 1 - tps
    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)
    recall = cum_tp / max(num_gt, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    # all-point interpolation
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(
    detections: Dict[str, List[Dict]],
    gts: Dict[str, Dict],
    max_range: float | None = None,
) -> Dict[str, Dict[str, float]]:
    per_class_per_metric: Dict[str, Dict[str, Dict]] = {
        cls: {f"dist_{t}": {"tp": [], "score": [], "ngt": 0} for t in CENTER_DIST_THRESHOLDS}
        for cls in CLASS_NAMES
    }
    for cls in CLASS_NAMES:
        for t in IOU_THRESHOLDS:
            per_class_per_metric[cls][f"iou_{t}"] = {"tp": [], "score": [], "ngt": 0}

    skipped = 0
    n_det_total = 0
    n_gt_total = 0
    for token, info in gts.items():
        if token not in detections:
            skipped += 1
            continue
        gt_boxes = build_gt_boxes(info)
        det_boxes = build_det_boxes_in_lidar(detections[token], info)
        if max_range is not None:
            gt_boxes = [b for b in gt_boxes if np.hypot(b.x, b.y) <= max_range]
            det_boxes = [b for b in det_boxes if np.hypot(b.x, b.y) <= max_range]
        n_det_total += len(det_boxes)
        n_gt_total += len(gt_boxes)
        for cls in CLASS_NAMES:
            cls_gt = [g for g in gt_boxes if g.name == cls]
            cls_det = [d for d in det_boxes if d.name == cls]
            for t in CENTER_DIST_THRESHOLDS:
                tps, scs, _ = match_and_score(cls_det, cls_gt, "dist", t)
                slot = per_class_per_metric[cls][f"dist_{t}"]
                slot["tp"].extend(tps)
                slot["score"].extend(scs)
                slot["ngt"] += len(cls_gt)
            for t in IOU_THRESHOLDS:
                tps, scs, _ = match_and_score(cls_det, cls_gt, "iou", t)
                slot = per_class_per_metric[cls][f"iou_{t}"]
                slot["tp"].extend(tps)
                slot["score"].extend(scs)
                slot["ngt"] += len(cls_gt)

    if skipped:
        log.warning("skipped %d GT samples (no det)", skipped)
    log.info("processed %d samples, %d dets, %d gts (filtered to %s)",
             len(gts) - skipped, n_det_total, n_gt_total, CLASS_NAMES)

    out: Dict[str, Dict[str, float]] = {}
    for cls in CLASS_NAMES:
        out[cls] = {}
        for key, slot in per_class_per_metric[cls].items():
            out[cls][key] = compute_ap(slot["tp"], slot["score"], slot["ngt"])
            out[cls][f"{key}_ngt"] = slot["ngt"]
    return out


def report(results: Dict[str, Dict[str, float]]) -> None:
    print("=" * 78)
    print("Per-class AP @ center distance (nuScenes-style, ego frame)")
    print("=" * 78)
    header = f"{'class':<12}" + "".join(f"  d={t:<5}" for t in CENTER_DIST_THRESHOLDS) + "  mAP_d   |  ngt"
    print(header)
    overall_d = []
    for cls in CLASS_NAMES:
        row = f"{cls:<12}"
        vals = []
        for t in CENTER_DIST_THRESHOLDS:
            v = results[cls][f"dist_{t}"]
            vals.append(v)
            row += f"  {v:.4f}"
        m = float(np.mean(vals)) if vals else 0.0
        overall_d.append(m)
        ngt = int(results[cls][f"dist_{CENTER_DIST_THRESHOLDS[0]}_ngt"])
        row += f"  {m:.4f}  |  {ngt}"
        print(row)
    print(f"{'overall':<12}" + " " * (10 * len(CENTER_DIST_THRESHOLDS)) + f"  {np.mean(overall_d):.4f}")

    print()
    print("=" * 78)
    print("Per-class AP @ BEV IoU (ego frame)")
    print("=" * 78)
    header = f"{'class':<12}" + "".join(f"  iou={t:<5}" for t in IOU_THRESHOLDS) + "  mAP_iou"
    print(header)
    overall_i = []
    for cls in CLASS_NAMES:
        row = f"{cls:<12}"
        vals = []
        for t in IOU_THRESHOLDS:
            v = results[cls][f"iou_{t}"]
            vals.append(v)
            row += f"  {v:.4f}"
        m = float(np.mean(vals)) if vals else 0.0
        overall_i.append(m)
        row += f"  {m:.4f}"
        print(row)
    print(f"{'overall':<12}" + " " * (12 * len(IOU_THRESHOLDS)) + f"  {np.mean(overall_i):.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--det", type=Path, required=True, help="results_nusc_det.json")
    p.add_argument("--gt", type=Path, required=True, help="spd_infos_temporal_val.pkl")
    p.add_argument("--out", type=Path, default=None, help="optional JSON output of metrics")
    p.add_argument("--max-range", type=float, default=None,
                   help="filter both GT and det to ego radius <= R (m)")
    args = p.parse_args()

    log.info("loading det: %s", args.det)
    detections = load_detections(args.det)
    log.info("  %d sample tokens", len(detections))
    log.info("loading gt: %s", args.gt)
    gts = load_gt(args.gt)
    log.info("  %d gt tokens", len(gts))

    if args.max_range is not None:
        log.info("filtering GT and det to ego radius <= %.1f m", args.max_range)
    results = evaluate(detections, gts, max_range=args.max_range)
    report(results)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump(results, f, indent=2)
        log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
