"""Patch inference_multiclass.py -> inference_apcalib.py:
   adds env-gated per-class conf-ranked box drop (same operator as closed-loop),
   test_dir override, frame cap, and a clean APCALIB summary print.
"""
import os
base = "/data/jichengzhi_v2x/V2Xverse/opencood/tools"
src = open(os.path.join(base, "inference_multiclass.py")).read()

# --- 1. module-level drop helper (insert after first 'def main():') ---
helper = '''
def _ap_drop_boxes(det_all, score_all, drop):
    import math, torch
    if drop <= 0 or det_all is None:
        return det_all, score_all
    nb, ns = [], []
    for c in range(len(det_all)):
        b, s = det_all[c], score_all[c]
        if b is None or s is None or len(s) == 0:
            nb.append(b); ns.append(s); continue
        n = len(s); k = int(math.floor(n * drop))
        if k <= 0:
            nb.append(b); ns.append(s); continue
        order = torch.argsort(s)          # conf ascending
        keep = order[k:]                  # drop lowest-conf k
        nb.append(b[keep]); ns.append(s[keep])
    return nb, ns

'''
assert "def main():" in src
src = src.replace("def main():", helper + "def main():", 1)

# --- 2. test_dir override right after hypes load ---
anchor2 = "    hypes = yaml_utils.load_yaml(None, opt)"
assert anchor2 in src
src = src.replace(anchor2, anchor2 + '''
    _td = os.environ.get('APCALIB_TESTDIR', '')
    if _td:
        hypes['test_dir'] = _td; hypes['root_dir'] = _td
        print('[apcalib] test_dir ->', _td)''', 1)

# --- 3. frame cap + drop injection right after pred_score is read ---
anchor3 = "            pred_score = infer_result['pred_score']"
assert anchor3 in src
src = src.replace(anchor3, anchor3 + '''
            if i >= int(os.environ.get('APCALIB_NMAX', '250')):
                break
            pred_box_tensor, pred_score = _ap_drop_boxes(
                pred_box_tensor, pred_score, float(os.environ.get('AP_DROP', '0')))
            infer_result['pred_box_tensor'] = pred_box_tensor
            infer_result['pred_score'] = pred_score''', 1)

# --- 4. clean summary print at end of main ---
anchor4 = "    log_file.close()"
assert anchor4 in src
src = src.replace(anchor4, anchor4 + '''
    _drop = float(os.environ.get('AP_DROP', '0'))
    print('APCALIB_RESULT drop=%.3f veh_ap50=%.4f veh_ap70=%.4f ped_ap50=%.4f cyc_ap50=%.4f' % (
        _drop, all_class_results[0]['ap50'], all_class_results[0]['ap70'],
        all_class_results[1]['ap50'], all_class_results[3]['ap50']))''', 1)

out = os.path.join(base, "inference_apcalib.py")
open(out, "w").write(src)
print("wrote", out)
