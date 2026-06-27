#!/bin/bash
# Collect INT8 results from all GPU workers into official result files.
# Run this after tuning completes to assemble q_int8_base_gate.csv and q_int8_pairs.csv.
# Usage: bash collect_int8_results.sh

RESULT_DIR=/home/jichengzhi/V2X/results
FP16_BASE=6319.98   # H800 TVM FP16 tuned base reference (latency_lut_pyramid.json)
FP16_LUT=/home/jichengzhi/V2X/results/latency_lut_pyramid.json

echo "=== INT8 Results Collector ==="
echo "$(date)"

# ----------- Q0 base gate -----------
echo ""
echo "--- Q0 base gate (GPU 1) ---"
Q0_JSON=/exdata/jichengzhi/s2_tvm/q0_int8_gate_fixed_result.json
if [ ! -f $Q0_JSON ]; then
    echo "ERROR: Q0 result not found at $Q0_JSON"
else
    INT8_DEF=$(python3 -c "import json; d=json.load(open('$Q0_JSON')); print(d.get('int8_default_us',-1))" 2>/dev/null)
    INT8_TUN=$(python3 -c "import json; d=json.load(open('$Q0_JSON')); print(d.get('int8_tuned_us',-1))" 2>/dev/null)
    echo "base INT8 default: ${INT8_DEF} us"
    echo "base INT8 tuned:   ${INT8_TUN} us"

    if [ "$INT8_TUN" != "-1" ] && [ "$INT8_TUN" != "" ]; then
        SPEEDUP=$(python3 -c "print(f'{$FP16_BASE/$INT8_TUN:.3f}')" 2>/dev/null)
        echo "INT8 vs FP16 tuned speedup: ${SPEEDUP}x"
    fi

    cat > $RESULT_DIR/q_int8_base_gate.csv << EOF
width,label,prec,sched,lat_us,fp16_tuned_us,int8_speedup,source,notes
"[64,128,256]",base,int8,default,${INT8_DEF},${FP16_BASE},,H800_TVM_int8,q0_gate_fixed GPU1 300trials
"[64,128,256]",base,int8,tuned,${INT8_TUN},${FP16_BASE},,H800_TVM_int8,q0_gate_fixed GPU1 300trials
EOF
    echo "Written: $RESULT_DIR/q_int8_base_gate.csv"
fi

# ----------- Q1 pairs -----------
echo ""
echo "--- Q1 pairs ---"
# Merge csv files from GPU 0, 2, 3
COMBINED=/tmp/q1_int8_combined.csv
echo "label,prec,sched,lat_us,source,notes" > $COMBINED

for F in \
    /exdata/jichengzhi/s2_tvm/q1_int8_pairs_gpu0.csv \
    /exdata/jichengzhi/s2_tvm/q1_int8_pairs_gpu2.csv \
    /exdata/jichengzhi/s2_tvm/q1_int8_pairs_gpu3.csv; do
    if [ -f "$F" ]; then
        tail -n +2 "$F" >> $COMBINED  # skip header
        echo "  Merged: $F ($(wc -l < $F) rows)"
    else
        echo "  MISSING: $F"
    fi
done

echo ""
echo "Combined pairs:"
cat $COMBINED

echo ""
# Write to official location
cp $COMBINED $RESULT_DIR/q_int8_pairs.csv
echo "Written: $RESULT_DIR/q_int8_pairs.csv"

# ----------- Analysis -----------
echo ""
echo "--- Analysis: H800 TVM INT8 vs FP16 speedup ---"
python3 << 'PYEOF'
import json, os, csv

FP16_TUNED = {
    "base":   6319.98,
    "trap25": 21615.0,
    "pad64":  6152.0,
    "mix_b":  19405.0,
    "s1_64":  5895.0,
    "mix_d":  None,   # need to fill from latency_lut
    "s2_128": None,
}

# Try to load FP16 LUT for mix_d and s2_128
lut_path = "/home/jichengzhi/V2X/results/latency_lut_pyramid.json"
if os.path.exists(lut_path):
    lut = json.load(open(lut_path))
    for label, v in lut.items():
        if label in FP16_TUNED and FP16_TUNED[label] is None:
            FP16_TUNED[label] = v.get("tuned_us")

pairs_csv = "/home/jichengzhi/V2X/results/q_int8_pairs.csv"
if not os.path.exists(pairs_csv):
    print("pairs CSV not found yet")
else:
    rows = list(csv.DictReader(open(pairs_csv)))
    by_label = {}
    for r in rows:
        l = r['label']
        s = r['sched']
        lat = float(r['lat_us'])
        if l not in by_label:
            by_label[l] = {}
        by_label[l][s] = lat

    print(f"\n{'Label':<8} {'INT8_def':>10} {'INT8_tun':>10} {'FP16_tun':>10} {'speedup_def':>12} {'speedup_tun':>12}")
    for label, scheds in sorted(by_label.items()):
        fp16 = FP16_TUNED.get(label, None)
        def_lat = scheds.get('default', None)
        tun_lat = scheds.get('tuned', None)
        speedup_def = f"{fp16/def_lat:.3f}x" if fp16 and def_lat else "N/A"
        speedup_tun = f"{fp16/tun_lat:.3f}x" if fp16 and tun_lat else "pending"
        print(f"{label:<8} {def_lat or 'N/A':>10} {tun_lat or 'pending':>10} {fp16 or 'N/A':>10} {speedup_def:>12} {speedup_tun:>12}")

    # Key s0-mismatch analysis
    print("\n--- S0-MISMATCH ANALYSIS (trap25 vs pad64): ---")
    t_def = by_label.get('trap25', {}).get('default')
    p_def = by_label.get('pad64', {}).get('default')
    t_tun = by_label.get('trap25', {}).get('tuned')
    p_tun = by_label.get('pad64', {}).get('tuned')
    if t_def and p_def:
        print(f"  Default ratio trap25/pad64 = {t_def/p_def:.3f}x (>1.2 = s0-mismatch confirmed under INT8)")
    if t_tun and p_tun:
        print(f"  Tuned ratio trap25/pad64   = {t_tun/p_tun:.3f}x")

    # mix_b vs s1_64
    print("\n--- S0-MISMATCH ANALYSIS (mix_b vs s1_64): ---")
    m_def = by_label.get('mix_b', {}).get('default')
    s_def = by_label.get('s1_64', {}).get('default')
    m_tun = by_label.get('mix_b', {}).get('tuned')
    s_tun = by_label.get('s1_64', {}).get('tuned')
    if m_def and s_def:
        print(f"  Default ratio mix_b/s1_64 = {m_def/s_def:.3f}x (>1.2 = s0-mismatch confirmed under INT8)")
    if m_tun and s_tun:
        print(f"  Tuned ratio mix_b/s1_64   = {m_tun/s_tun:.3f}x")

PYEOF

echo ""
echo "=== DONE ==="
