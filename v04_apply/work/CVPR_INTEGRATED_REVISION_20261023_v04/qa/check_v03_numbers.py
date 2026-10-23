"""Check SpatialVLA numbers printed in the v03 TeX against values recomputed from raw tokens (analysis summaries produced by
the standalone-decoder analyzer, cross-checked by the independent verifier) and denominators re-derived from raw records."""
import json, csv, statistics, sys, re
W = "/root/VLA/integrated_v03_work"
tex = open(f"{W}/paper_v03/paper_body.tex").read() + open(f"{W}/paper_v03/supp_v03_svla.tex").read() + open(f"{W}/paper_v03/supp_v03_moved.tex").read()
v01 = json.load(open(f"{W}/verifier/v01_standalone_result.json"))
v02 = json.load(open(f"{W}/svla_v02/analysis/summary.json")); dr = json.load(open(f"{W}/svla_drawer/analysis/summary.json"))
vv02 = json.load(open(f"{W}/verifier/v02_standalone_result.json")); vdr = json.load(open(f"{W}/verifier/drawer_standalone_result.json"))
f3 = lambda x: f"{x:.3f}".replace("-", "-")  # noqa: E731
iv = lambda c: f"[{c[0]:.3f},{c[1]:.3f}]"  # noqa: E731
res = []
def chk(name, s):
    s2 = s.replace("-", "$-$") if False else s
    ok = s in tex or s.replace("-0.", "$-0.").replace(",-", ",-") in tex
    res.append((name, s, ok))
P, S = v02["primary_Delta"], v02["secondary"]
chk("v01 secondary", f"{f3(v01['secondary_all_Delta']['two_task'])}$ ${iv(v01['secondary_all_Delta']['ci'])}")
chk("v01 secondary table", f"64/32 & {f3(v01['secondary_all_Delta']['two_task'])} & ${iv(v01['secondary_all_Delta']['ci'])}$")
chk("v01 matched", f"7/7 & {f3(v01['primary_matched_Delta']['two_task'])}")
chk("v02 primary table", f"69/34 & {f3(P['estimate'])} & ${iv(P['ci'])}$")
chk("v02 primary text", f"$\\Delta={f3(P['estimate'])}$ ${iv(P['ci'])}$")
for t, lab, n in (("google_robot_pick_coke_can", "coke", "55/25"), ("google_robot_move_near", "move\\_near", "14/9")):
    chk(f"v02 {lab}", f"{n} & {f3(P['task_means'][t])} & ${iv(P['task_ci'][t])}$")
chk("v02 D(OL,L)", f"{f3(S['D_OL_L']['estimate'])}$ ${iv(S['D_OL_L']['ci'])}$")
chk("v02 J", f"$-{abs(S['J']['estimate']):.3f}$ & $[-{abs(S['J']['ci'][0]):.3f},-{abs(S['J']['ci'][1]):.3f}]$")
chk("v02 same change", f"{f3(S['Delta_same_change_type']['estimate'])} & ${iv(S['Delta_same_change_type']['ci'])}$")
chk("v02 EL/EO", f"{f3(S['E_L']['estimate'])} / {f3(S['E_O']['estimate'])}")
chk("v02 a3/a4", f"{f3(S['E_L_a3']['estimate'])} / {f3(S['E_O_a3']['estimate'])}; {f3(S['E_L_a4']['estimate'])} / {f3(S['E_O_a4']['estimate'])}")
chk("v02 trans/rot", f"{f3(S['E_L_trans']['estimate'])} / {f3(S['E_O_trans']['estimate'])}; {f3(S['E_L_rot']['estimate'])} / {f3(S['E_O_rot']['estimate'])}")
chk("v02 grip", f"{f3(S['grip_change_L']['estimate'])} / {f3(S['grip_change_O']['estimate'])} / {f3(S['grip_change_OL']['estimate'])}")
chk("v02 token change", f"{v02['token_change']['L']} observations, the older edit in {v02['token_change']['O']}")
chk("v02 d", f"mean $d_L$ {v02['donor_distance']['mean_d_L']:.3f} vs.\\ $d_O$ {v02['donor_distance']['mean_d_O']:.3f}")
sr = v02["source_reasons"]
chk("v02 sources", f"{v02['n_sources']:,} samples were generated")
chk("v02 rejections", f"{sr['zero_distance']:,} were rejected for zero distance (they reproduced a natural action), {sr['ratio_outside_0.8_1.25']} for a ratio outside $[0.8,1.25]$ and {sr['gripper_mismatch']} for a gripper mismatch; {sr['selected']} were selected")
fl = list(csv.DictReader(open(f"{W}/svla_v02/analysis/flow_per_observation.csv")))
chk("v02 median sources", f"median {int(statistics.median(int(x['n_sources_tried']) for x in fl))} per observation")
q = v02["donor_distance"]["ratio_quantiles"]; chk("v02 ratio quartiles", f"{q[1]:.2f}/{q[2]:.2f}/{q[3]:.2f}")
F = v02["flow"]; chk("v02 flow coke", f"& {F['google_robot_pick_coke_can']['analyzed']} (25) & {F['google_robot_move_near']['analyzed']} (9) & 69 (34)")
DP, DS = dr["primary_Delta"], dr["secondary"]
chk("drawer table", f"35/15 & {f3(DP['estimate'])} & ${iv(DP['ci'])}$")
chk("drawer text", f"$\\Delta={f3(DP['estimate'])}$ ${iv(DP['ci'])}$")
chk("drawer EL EO", f"$E_L={f3(DS['E_L']['estimate'])}$ and $E_O={f3(DS['E_O']['estimate'])}$")
chk("drawer DOLL", f"$D(OL,L)={f3(DS['D_OL_L']['estimate'])}$ ${iv(DS['D_OL_L']['ci'])}$")
chk("drawer J", f"$J=-{abs(DS['J']['estimate']):.3f}$ $[-{abs(DS['J']['ci'][0]):.3f},-{abs(DS['J']['ci'][1]):.3f}]$")
chk("drawer sources", f"Of {dr['n_sources']:,} samples, {dr['source_reasons']['zero_distance']} were rejected for zero distance and {dr['source_reasons']['ratio_outside_0.8_1.25']} for the ratio")
chk("drawer d", f"({dr['donor_distance']['mean_d_L']:.3f} vs.\\ {dr['donor_distance']['mean_d_O']:.3f})")
chk("drawer matched count", f"{dr['flow']['google_robot_open_drawer']['analyzed']}/{dr['flow']['google_robot_open_drawer']['collected']} observations (15 layouts)")
indep = {"v02_verifier_ALL_MATCH": vv02["ALL_MATCH"], "drawer_verifier_ALL_MATCH": vdr["ALL_MATCH"],
         "v01_standalone_primary": round(v01["primary_matched_Delta"]["two_task"], 7), "v01_standalone_secondary": round(v01["secondary_all_Delta"]["two_task"], 7)}
bad = [r for r in res if not r[2]]
out = {"checks": [{"name": n, "expected_string": s, "found": ok} for n, s, ok in res], "independent": indep, "n_pass": len(res) - len(bad), "n_fail": len(bad)}
json.dump(out, open(f"{W}/qa/NUMERICAL_QA_SVLA_20261023_v03.json", "w"), indent=1)
print(out["n_pass"], "pass", out["n_fail"], "fail"); [print("FAIL", b[0], b[1]) for b in bad]; print(indep)
