#!/usr/bin/env python3
"""
bench2drive_metrics.py — offline Efficiency + Comfort metrics for V2Xverse closed-loop.

Ports B2D efficiency_smoothness_benchmark.py (Thinklab-SJTU/Bench2Drive) to V2Xverse.
Input: metric_info.json produced by pnp_agent_e2e.py (METRIC_LOG=1).
Output: Efficiency (%), Comfort (0-1), with intermediate diagnostics.

Thresholds (§9.2, verbatim from B2D source / NeurIPS24 paper arXiv 2406.03877v3):
  lon_acc  : [-4.05, 2.40] m/s²
  lat_acc  : [-4.89, 4.89] m/s²
  jerk_mag : [-8.37, 8.37] m/s³
  lon_jerk : [-4.13, 4.13] m/s³
  yaw_acc  : [-1.93, 1.93] rad/s²
  yaw_rate : [-0.95, 0.95] rad/s

dt = 0.05s (V2Xverse 20Hz ticks; B2D original is 10Hz/dt=0.1, thresholds are physical units so kept verbatim)
Savitzky-Golay: window=7, polyorder=2; jerk/yaw_acc use deriv=1 with delta=dt.
"""

import json
import sys
import os
import math
import argparse
import numpy as np
from scipy.signal import savgol_filter

# ─────────────────────────────────────────────
# Physical thresholds (B2D verbatim)
# ─────────────────────────────────────────────
LON_ACC_MIN, LON_ACC_MAX   = -4.05, 2.40   # m/s²
LAT_ACC_MAX                = 4.89          # ± m/s²
JERK_MAG_MAX               = 8.37          # ± m/s³
LON_JERK_MAX               = 4.13          # ± m/s³
YAW_ACC_MAX                = 1.93          # ± rad/s²
YAW_RATE_MAX               = 0.95          # ± rad/s

SEGMENT_LEN = 20            # frames per comfort segment
SG_WINDOW   = 7             # Savitzky-Golay window (must be odd, ≤ SEGMENT_LEN)
SG_POLY     = 2             # Savitzky-Golay polyorder
DT          = 0.05          # seconds per frame (20Hz V2Xverse)

EFF_CAP     = 10.0          # 1000% cap → 10.0× ratio


def savgol_smooth(arr, deriv=0):
    """Apply savgol_filter. For deriv>0 also divide by dt^deriv to get physical units."""
    if len(arr) < SG_WINDOW:
        return arr
    filtered = savgol_filter(arr, window_length=SG_WINDOW, polyorder=SG_POLY,
                              deriv=deriv, delta=DT)
    return filtered


def compute_comfort(frames):
    """
    Compute route comfort score from a list of per-frame dicts.
    Returns (comfort_score, total_segments, comfortable_segments, diagnostics_dict).
    """
    n = len(frames)
    if n < SEGMENT_LEN:
        return None, 0, 0, {}

    # Extract kinematic arrays
    acc_x  = np.array([f['acceleration'][0]    for f in frames])
    acc_y  = np.array([f['acceleration'][1]    for f in frames])
    fwd_x  = np.array([f['forward_vector'][0]  for f in frames])
    fwd_y  = np.array([f['forward_vector'][1]  for f in frames])
    rgt_x  = np.array([f['right_vector'][0]    for f in frames])
    rgt_y  = np.array([f['right_vector'][1]    for f in frames])
    yaw_r  = np.array([f['angular_velocity'][2] for f in frames])  # rad/s, z component

    # Longitudinal and lateral acceleration (2D projection)
    lon_acc_full = acc_x * fwd_x + acc_y * fwd_y
    lat_acc_full = acc_x * rgt_x + acc_y * rgt_y
    acc_mag_full = np.sqrt(acc_x**2 + acc_y**2)

    # Phase-unwrap yaw rate (handles discontinuities)
    yaw_r_unwrapped = np.unwrap(yaw_r)

    total_segs = 0
    comfortable_segs = 0
    seg_diagnostics = []

    num_segs = n // SEGMENT_LEN
    for s in range(num_segs):
        sl = slice(s * SEGMENT_LEN, (s + 1) * SEGMENT_LEN)
        seg_lon = lon_acc_full[sl]
        seg_lat = lat_acc_full[sl]
        seg_mag = acc_mag_full[sl]
        seg_yaw = yaw_r_unwrapped[sl]

        # Smooth with Savitzky-Golay
        lon_acc_s = savgol_smooth(seg_lon)
        lat_acc_s = savgol_smooth(seg_lat)
        mag_s     = savgol_smooth(seg_mag)
        yaw_s     = savgol_smooth(seg_yaw)

        # Jerk = derivative of smoothed quantities
        jerk_mag_s  = savgol_smooth(seg_mag,  deriv=1)   # d|a|/dt
        lon_jerk_s  = savgol_smooth(seg_lon,  deriv=1)   # d(lon_acc)/dt
        yaw_acc_s   = savgol_smooth(seg_yaw,  deriv=1)   # d(yaw_rate)/dt

        # Check all 6 bounds (every sample in segment must be in bounds)
        ok_lon_acc  = np.all((lon_acc_s  >= LON_ACC_MIN) & (lon_acc_s  <= LON_ACC_MAX))
        ok_lat_acc  = np.all(np.abs(lat_acc_s) <= LAT_ACC_MAX)
        ok_jerk_mag = np.all(np.abs(jerk_mag_s) <= JERK_MAG_MAX)
        ok_lon_jerk = np.all(np.abs(lon_jerk_s) <= LON_JERK_MAX)
        ok_yaw_acc  = np.all(np.abs(yaw_acc_s)  <= YAW_ACC_MAX)
        ok_yaw_rate = np.all(np.abs(yaw_s)      <= YAW_RATE_MAX)

        is_comfortable = all([ok_lon_acc, ok_lat_acc, ok_jerk_mag, ok_lon_jerk, ok_yaw_acc, ok_yaw_rate])

        total_segs += 1
        if is_comfortable:
            comfortable_segs += 1

        seg_diagnostics.append({
            'seg': s,
            'comfortable': is_comfortable,
            'mean_lon_acc': float(np.mean(lon_acc_s)),
            'max_abs_lat_acc': float(np.max(np.abs(lat_acc_s))),
            'max_abs_jerk_mag': float(np.max(np.abs(jerk_mag_s))),
            'max_abs_lon_jerk': float(np.max(np.abs(lon_jerk_s))),
            'max_abs_yaw_acc': float(np.max(np.abs(yaw_acc_s))),
            'max_abs_yaw_rate': float(np.max(np.abs(yaw_s))),
            'violations': {
                'lon_acc': not ok_lon_acc,
                'lat_acc': not ok_lat_acc,
                'jerk_mag': not ok_jerk_mag,
                'lon_jerk': not ok_lon_jerk,
                'yaw_acc': not ok_yaw_acc,
                'yaw_rate': not ok_yaw_rate,
            }
        })

    comfort_score = comfortable_segs / total_segs if total_segs > 0 else None
    diag = {
        'total_segs': total_segs,
        'comfortable_segs': comfortable_segs,
        'segments': seg_diagnostics,
    }
    return comfort_score, total_segs, comfortable_segs, diag


def compute_efficiency(frames):
    """
    Compute route efficiency score from per-frame dicts.
    speed_pct = ego_speed / mean(nearby_speeds), capped at EFF_CAP.
    Returns (mean_speed_pct_as_fraction, n_valid_frames, diagnostics).
    """
    valid_ratios = []
    skipped_no_nearby = 0
    skipped_zero_ref = 0

    for f in frames:
        nearby = f.get('nearby_speeds', [])
        if not nearby:
            skipped_no_nearby += 1
            continue
        ref_speed = float(np.mean(nearby))
        if ref_speed < 0.1:  # near-zero reference → skip (div by ~0)
            skipped_zero_ref += 1
            continue
        ratio = f['ego_speed'] / ref_speed
        ratio = min(ratio, EFF_CAP)
        valid_ratios.append(ratio)

    if not valid_ratios:
        return None, 0, {'skipped_no_nearby': skipped_no_nearby, 'skipped_zero_ref': skipped_zero_ref}

    mean_ratio = float(np.mean(valid_ratios))
    diag = {
        'n_valid': len(valid_ratios),
        'skipped_no_nearby': skipped_no_nearby,
        'skipped_zero_ref': skipped_zero_ref,
        'mean_ego_speed': float(np.mean([f['ego_speed'] for f in frames])),
        'min_ratio': float(np.min(valid_ratios)),
        'max_ratio': float(np.max(valid_ratios)),
        'mean_ratio': mean_ratio,
    }
    return mean_ratio, len(valid_ratios), diag


def run(metric_info_path, verbose=True):
    with open(metric_info_path) as f:
        frames = json.load(f)

    print(f"Loaded {len(frames)} frames from {metric_info_path}")
    print(f"dt = {DT}s (20Hz), segment_len = {SEGMENT_LEN} frames = {SEGMENT_LEN * DT:.2f}s per segment")

    # ── Comfort ──────────────────────────────────────────────
    comfort, total_segs, comf_segs, diag_c = compute_comfort(frames)
    print(f"\n── Comfort ──────────────────────────────────────")
    if comfort is None:
        print(f"  NOT COMPUTED: need >= {SEGMENT_LEN} frames, got {len(frames)}")
    else:
        print(f"  Total segments:       {total_segs}")
        print(f"  Comfortable segments: {comf_segs}")
        print(f"  Comfort score:        {comfort:.4f}  ({comfort*100:.1f}%)")
        if verbose and diag_c['segments']:
            print(f"\n  Per-segment diagnostics (first 5):")
            for seg in diag_c['segments'][:5]:
                viols = [k for k, v in seg['violations'].items() if v]
                status = "COMF" if seg['comfortable'] else f"FAIL({','.join(viols)})"
                print(f"    seg {seg['seg']:3d}: {status:30s}  mean_lon_acc={seg['mean_lon_acc']:+.3f}  "
                      f"max_lat={seg['max_abs_lat_acc']:.3f}  max_jerk={seg['max_abs_jerk_mag']:.3f}  "
                      f"max_yaw_rate={seg['max_abs_yaw_rate']:.3f}")
            if len(diag_c['segments']) > 5:
                print(f"    ... ({len(diag_c['segments'])-5} more segments)")

        # Summary statistics across all segments
        all_mean_lon = [s['mean_lon_acc'] for s in diag_c['segments']]
        print(f"\n  Mean lon_acc across all segs: {np.mean(all_mean_lon):+.4f} m/s²")
        violation_counts = {}
        for seg in diag_c['segments']:
            for k, v in seg['violations'].items():
                if v:
                    violation_counts[k] = violation_counts.get(k, 0) + 1
        if violation_counts:
            print(f"  Violation counts by type: {violation_counts}")
        else:
            print(f"  No violations found (all segments comfortable)")

    # ── Efficiency ────────────────────────────────────────────
    eff_ratio, n_valid, diag_e = compute_efficiency(frames)
    print(f"\n── Efficiency ───────────────────────────────────")
    print(f"  Total frames logged:    {len(frames)}")
    print(f"  Valid frames (nearby>0, ref_spd>0.1): {n_valid}")
    print(f"  Skipped (no nearby):    {diag_e.get('skipped_no_nearby', 0)}")
    print(f"  Skipped (zero ref):     {diag_e.get('skipped_zero_ref', 0)}")
    if eff_ratio is None:
        print(f"  Efficiency: NOT COMPUTED (no valid frames with nearby vehicles)")
    else:
        print(f"  Mean ego speed (all):   {diag_e['mean_ego_speed']:.3f} m/s")
        print(f"  speed_pct range:        [{diag_e['min_ratio']*100:.1f}%, {diag_e['max_ratio']*100:.1f}%]")
        print(f"  Efficiency (mean):      {eff_ratio*100:.2f}%")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n══ SUMMARY ══════════════════════════════════════")
    if comfort is not None:
        print(f"  Comfort:    {comfort:.4f}  ({comf_segs}/{total_segs} comfortable segments)")
    else:
        print(f"  Comfort:    N/A (insufficient frames)")
    if eff_ratio is not None:
        print(f"  Efficiency: {eff_ratio*100:.2f}%  (from {n_valid} valid frames)")
    else:
        print(f"  Efficiency: N/A (no nearby vehicles logged)")
    print(f"  dt used:    {DT}s (20Hz V2Xverse; B2D original dt=0.1s/10Hz, thresholds kept verbatim)")

    return comfort, eff_ratio


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='B2D Efficiency+Comfort offline metrics')
    parser.add_argument('metric_info', help='Path to metric_info.json produced by pnp_agent_e2e.py')
    parser.add_argument('--quiet', action='store_true', help='Suppress per-segment diagnostics')
    args = parser.parse_args()
    run(args.metric_info, verbose=not args.quiet)
