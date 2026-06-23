import os
base = "/data/jichengzhi_v2x/V2Xverse/simulation/leaderboard/team_code/agent_config"
src = open(os.path.join(base, "pnp_config_codriving_tp0_l1.yaml")).read()
# arm code -> (ap_drop, ap_featmask)
arms = {
    "ap0":   (0.0, 1),   # baseline (no-op)
    "ap50":  (0.5, 1),   # box + feature degraded
    "ap55":  (0.5, 0),   # box only (feature-bypass test)
    "ap100": (1.0, 1),   # blind sanity (DS must collapse)
}
for name, (drop, fm) in arms.items():
    out_lines = []
    for ln in src.splitlines():
        out_lines.append(ln)
        if ln.strip().startswith("tau_perc_ms:"):
            indent = ln[:len(ln) - len(ln.lstrip())]
            out_lines.append("%sap_drop: %s" % (indent, drop))
            out_lines.append("%sap_featmask: %d" % (indent, fm))
    txt = "\n".join(out_lines) + "\n"
    assert "ap_drop:" in txt, name
    out = os.path.join(base, "pnp_config_codriving_%s_l1.yaml" % name)
    open(out, "w").write(txt)
    print("wrote %s  (ap_drop=%s ap_featmask=%d)" % (os.path.basename(out), drop, fm))
# clean-route file (from known tp0-clean signal pool)
open("/data/jichengzhi_v2x/ap_routes.txt", "w").write("\n".join(["3", "17", "104", "136", "18"]) + "\n")
print("wrote ap_routes.txt: 3,17,104,136,18")
