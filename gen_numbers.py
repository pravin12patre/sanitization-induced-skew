"""Write LaTeX macros (numbers.tex) and tables for the paper straight from the
result files, so no number in the paper is typed by hand."""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DST = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "paper")
macros = {}
match_rows = {}


def J(ds, name):
    return json.load(open(os.path.join(HERE, f"out_{ds}", name)))


def f2(x): return f"{x:.2f}"
def f3(x): return f"{x:.3f}"
def f4(x): return f"{x:.4f}"
def pct(x, nd=0): return f"{100 * x:.{nd}f}\\%"
def ci(v, fmt=f3): return f"{fmt(v[0])} ({fmt(v[1])}--{fmt(v[2])})"
def num(x): return f"{x:,}".replace(",", "{,}")


TFDV_FEATS = ["recency_days", "frequency", "monetary", "freq_90d", "spend_90d",
              "tenure_days", "weekend_share", "q4_share", "last_dow"]
NAMES = {"recency_days": "Recency", "frequency": "Frequency", "monetary": "Monetary",
         "freq_90d": "Count, 90 days", "spend_90d": "Spend, 90 days",
         "tenure_days": "Tenure", "weekend_share": "Weekend share",
         "q4_share": "Nov--Dec share", "last_dow": "Last day of week"}

for ds, P in [("retail", "R"), ("cdnow", "C")]:
    res = J(ds, "results.json")
    rob = J(ds, "robustness.json")
    par = J(ds, "paired.json")
    fwd = J(ds, "forward.json")
    det = J(ds, "detect.json")
    dec = pd.read_csv(os.path.join(HERE, f"out_{ds}", "detect_decisions.csv"))
    dec["bug"] = dec.bug.fillna("None").astype(str)
    st = json.load(open(os.path.join(HERE, f"out_{ds}", "stats.json")))
    m = macros
    m[P + "Inv"] = num(st["orders"]); m[P + "Cust"] = num(st["customers"])
    m[P + "Rows"] = num(st["rows"]); m[P + "TrainRows"] = num(st["train_rows"])
    m[P + "TestRows"] = num(st["test_rows"]); m[P + "BaseRate"] = f3(st["base_rate"])
    m[P + "TestCust"] = num(rob["n_test_customers"])
    b = rob["cells"]["baseline"]["production_auc"]
    m[P + "BaseAUC"] = f3(b[0]); m[P + "BaseCI"] = f"{f3(b[1])}--{f3(b[2])}"
    for key, mac in [("rotate_incremental_monthly", "IncM"), ("rotate_incremental_weekly", "IncW"),
                     ("shift_-100", "Shift"), ("observed_rotate_incr+shift_-100", "Obs"),
                     ("observed+unshifted_cutoff", "ObsUnal")]:
        v = par[key]["production_auc_drop_vs_baseline"]
        m[P + "Drop" + mac] = f3(v[0]); m[P + "Drop" + mac + "CI"] = f"{f3(v[1])}--{f3(v[2])}"
        m[P + "Drop" + mac + "Sig"] = "yes" if v[1] > 0 else "no"
    ou = rob["cells"]["observed+unshifted_cutoff"]
    m[P + "ObsUnalOff"] = f4(ou["offline_auc"][0]); m[P + "ObsUnalProd"] = f3(ou["production_auc"][0])
    m[P + "ObsUnalProdCI"] = f"{f3(ou['production_auc'][1])}--{f3(ou['production_auc'][2])}"
    m[P + "IncMatch"] = pct(res["cells"]["rotate_incremental"]["features"]["frequency"]["match_rate"])
    u = rob["cells"]["shift_-100_unshifted_cutoff"]
    m[P + "OffUnal"] = f4(u["offline_auc"][0]); m[P + "ProdUnal"] = f3(u["production_auc"][0])
    m[P + "ProdUnalCI"] = f"{f3(u['production_auc'][1])}--{f3(u['production_auc'][2])}"
    m[P + "LeakUnal"] = pct(u["leak_rate"][0])
    ss = rob["shift_sweep"]
    m[P + "LeakOne"] = pct(ss["-1"]["unshifted"]["leak_rate"], 1)
    m[P + "LeakSeven"] = pct(ss["-7"]["unshifted"]["leak_rate"])
    m[P + "LeakYear"] = pct(ss["-365"]["unshifted"]["leak_rate"])
    m[P + "StaleHundred"] = pct(fwd["+100_unshifted"]["rows_missing_past_events"])
    m[P + "ProdFwdHundred"] = f3(fwd["+100_unshifted"]["production_auc"])
    m[P + "OffFwdHundred"] = f3(fwd["+100_unshifted"]["offline_auc"])
    ep = rob["epoch_sweep"]
    for e in ["weekly", "monthly", "quarterly", "yearly"]:
        m[P + "Epoch" + e.capitalize() + "Match"] = pct(ep[e]["frequency_match"])
        m[P + "Epoch" + e.capitalize() + "AUC"] = f3(ep[e]["production_auc"])
    # matched comparison threshold sensitivity and footprint
    for mon, M in [("tfdv", "Tfdv"), ("evidently", "Evid")]:
        x = dec[dec.monitor == mon]
        for tau, T in [(0.05, "Lo"), (0.1, "Mid"), (0.2, "Hi")]:
            g = x.assign(a=x.score_matched > tau).groupby(["sanitization", "bug"])
            caught = g.apply(lambda z: bool((z.a & z.truth).any()), include_groups=False)
            fa = g.apply(lambda z: bool(z.a.any()), include_groups=False)
            m[P + M + "Matched" + T] = str(int(caught[caught.index.get_level_values(1) != "None"].sum()))
            m[P + M + "MatchedFA" + T] = str(int(fa[fa.index.get_level_values(1) == "None"].sum()))
        # ceiling: bugs this monitor detects with no sanitization, counted once
        # per sanitization setting; matched detections are compared against it
        none = x[x.sanitization == "none"]
        det_none = {b for b, z in none.groupby("bug") if b != "None" and bool((z.plain & z.truth).any())}
        m[P + M + "Ceiling"] = str(4 * len(det_none))
        g = x.groupby(["sanitization", "bug"])
        mdet = g.apply(lambda z: bool((z.matched & z.truth).any()), include_groups=False)
        m[P + M + "MatchedDet"] = str(int(mdet[mdet.index.get_level_values(1) != "None"].sum()))
        lost = [(s, b) for (s, b), v in mdet.items() if b in det_none and not v]
        m[P + M + "MatchedLost"] = ", ".join(sorted({f"{b} under {s}" for s, b in lost})) or "none"
        fp = x[(x.sanitization == "rotate_incremental") & (x.bug == "window60") & (x.feature == "freq_90d")]
        m[P + M + "Footprint"] = f3(float(fp.score_matched.iloc[0]))
        nb = x[(x.sanitization == "none") & (x.bug == "tz_shift")]
        m[P + M + "TzNone"] = "detects" if bool((nb.plain & nb.truth).any()) else "misses"
        bf = x[(x.bug == "None") & x.plain].groupby("sanitization").size()
        m[P + M + "AlertsMin"] = str(int(bf.min())); m[P + M + "AlertsMax"] = str(int(bf.max()))
        r = x[(x.bug == "None") & (x.sanitization == "rotate_incremental") & (x.feature == "freq_90d")]
        m[P + M + "ResObs"] = f3(float(r.score_obs.iloc[0])); m[P + M + "ResExp"] = f3(float(r.score_exp.iloc[0]))
    # tables -------------------------------------------------------------
    cells = ["rotate_full", "rotate_incremental", "shift_-98", "shift_-100",
             "shift_-100_unshifted_cutoff", "HEADLINE_rotate_incr+shift_-100"]
    rows = []
    for f in TFDV_FEATS:
        vals = " & ".join(f"{res['cells'][c]['features'][f]['match_rate']:.2f}" for c in cells)
        rows.append((NAMES[f], vals))
    match_rows[ds] = rows
    s = det["summary"]
    lines = []
    for mon, label in [("tfdv", "TFDV"), ("evidently", "Evidently")]:
        for i, (var, vl) in enumerate([("plain", "Plain"), ("rules", "Rules"),
                                       ("replay", "Residual"), ("matched", "Matched")]):
            d = s[f"{mon}/{var}"]
            lead = f"\\multirow{{4}}{{*}}{{{label}}}" if i == 0 else ""
            lines.append((lead, vl, d["bug_free_scenarios_alerting"],
                          d["bugs_distinguishable"], d["observed_case_distinguishable"]))
    open(os.path.join(DST, f"detect_{ds}.json"), "w").write(json.dumps(lines))

open(os.path.join(DST, "tab_match_both.tex"), "w").write("\n".join(
    f"{a[0]} & {a[1]} && {b[1]}\\\\" for a, b in zip(match_rows["retail"], match_rows["cdnow"])) + "\n")

# combined detection table: both datasets side by side
dr = json.load(open(os.path.join(DST, "detect_retail.json")))
dc = json.load(open(os.path.join(DST, "detect_cdnow.json")))
out = []
for i, (a, c) in enumerate(zip(dr, dc)):
    out.append(f"{a[0]} & {a[1]} & {a[2]} & {a[3]} & {a[4]} & {c[2]} & {c[3]} & {c[4]}\\\\")
    if i == 3:
        out.append("\\midrule")
open(os.path.join(DST, "tab_detect.tex"), "w").write("\n".join(out) + "\n")

# paired AUC table, both datasets
rows = []
for key, label in [("rotate_incremental_weekly", "Incr.\\ rot., weekly keys"),
                   ("observed+unshifted_cutoff", "Incr.\\ rot.\\ + $-100$, unaligned"),
                   ("rotate_incremental_monthly", "Incr.\\ rot., monthly keys"),
                   ("observed_rotate_incr+shift_-100", "Incr.\\ rot.\\ + $-100$, aligned"),
                   ("shift_-100", "Shift $-100$, aligned")]:
    cells = []
    for ds in ["retail", "cdnow"]:
        v = J(ds, "paired.json")[key]["production_auc_drop_vs_baseline"]
        sg = lambda x: f"$-${-x:.3f}" if x < 0 else f"{x:.3f}"
        cells.append(f"{sg(v[0])} & {sg(v[1])}--{sg(v[2])}")
    rows.append(f"{label} & {cells[0]} & {cells[1]}\\\\")
open(os.path.join(DST, "tab_auc.tex"), "w").write("\n".join(rows) + "\n")

# epoch sweep table
rows = []
for e in ["weekly", "monthly", "quarterly", "yearly"]:
    r, c = J("retail", "robustness.json")["epoch_sweep"][e], J("cdnow", "robustness.json")["epoch_sweep"][e]
    rows.append(f"{e.capitalize()} & {100*r['frequency_match']:.0f}\\% & {r['production_auc']:.3f} & "
                f"{100*c['frequency_match']:.0f}\\% & {c['production_auc']:.3f}\\\\")
open(os.path.join(DST, "tab_epoch.tex"), "w").write("\n".join(rows) + "\n")

# collisions table
rows = []
cr, cc = J("retail", "collide.json"), J("cdnow", "collide.json")
macros["RCollN"] = num(cr["n_entities"]); macros["CCollN"] = num(cc["n_entities"])
for k in ["12_bits", "16_bits", "20_bits", "24_bits", "32_bits", "64_bits"]:
    a, b = cr["cells"][k], cc["cells"][k]
    rows.append(f"{k.split('_')[0]} & {100*a['entities_sharing_id']:.2f}\\% & {100*a['birthday_expected']:.2f}\\% & "
                f"{a['frequency_match']:.3f} & {a['production_auc']:.3f} && "
                f"{100*b['entities_sharing_id']:.2f}\\% & {100*b['birthday_expected']:.2f}\\% & "
                f"{b['frequency_match']:.3f} & {b['production_auc']:.3f}\\\\")
open(os.path.join(DST, "tab_collide_both.tex"), "w").write("\n".join(rows) + "\n")

# cross-table
x = J("retail", "xtable.json")
macros["XDom"] = pct(x["share_domestic_online"])
for sc, M in [("consistent", "Cons"), ("format_mismatch", "Fmt"), ("stale_quarterly", "Stale")]:
    macros["X" + M + "Match"] = pct(x[sc]["is_domestic_match"])
    macros["X" + M + "Off"] = f3(x[sc]["offline_auc"]); macros["X" + M + "Prod"] = f3(x[sc]["production_auc"])

# cost table
rows = []
for ds, P in [("retail", "R"), ("cdnow", "C")]:
    c = J(ds, "cost.json")
    macros[P + "FullBackfillS"] = f"{c['full_backfill_seconds']:.2f}"
    s = c["summary"]
    macros[P + "EvidCostMin"] = f"{min(s[f'evidently@{p}']['detected_mean_of_16'] for p in [0.05, 0.1, 0.2, 0.5, 1.0]):.1f}"
    macros[P + "EvidCostMax"] = f"{max(s[f'evidently@{p}']['detected_mean_of_16'] for p in [0.05, 0.1, 0.2, 0.5, 1.0]):.1f}"
    macros[P + "EvidCostLoS"] = f"{s['evidently@0.05']['median_seconds_per_check']:.2f}"
    macros[P + "TfdvCostLoS"] = f"{s['tfdv@0.05']['median_seconds_per_check']:.2f}"
for p in [0.05, 0.1, 0.2, 0.5, 1.0]:
    cells = []
    for ds in ["retail", "cdnow"]:
        s = J(ds, "cost.json")["summary"]
        t, e = s[f"tfdv@{p}"], s[f"evidently@{p}"]
        cells.append(f"{t['detected_mean_of_16']:.1f} & {e['detected_mean_of_16']:.1f} & "
                     f"{max(t['false_alarms_max'], e['false_alarms_max'])} & "
                     f"{e['median_seconds_per_check']:.2f}")
    rows.append(f"{int(p*100)}\\% & {cells[0]} & {cells[1]}\\\\")
open(os.path.join(DST, "tab_cost.tex"), "w").write("\n".join(rows) + "\n")

# manifest-only predictions
for ds, P in [("retail", "R"), ("cdnow", "C")]:
    p = J(ds, "predict.json")
    assert p["fp"] == 0 and p["fn"] == 0, f"prediction errors on {ds}: {p['errors']}"
    macros[P + "PredDecisions"] = str(p["decisions"])

# raw vs matched testing, both datasets; also verify the text's claims
rows = []
for ds, title in [("retail", "Online Retail II"), ("cdnow", "CDNOW")]:
    d2 = J(ds, "detect2.json")
    dec2 = pd.read_csv(os.path.join(HERE, f"out_{ds}", "detect2_decisions.csv"))
    rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{title}}}}}\\\\")
    for kind, kl in [("code", "Code"), ("sanitization-dependent", "San.-dep.")]:
        for mon, ml in [("evidently", "Evidently"), ("tfdv", "TFDV")]:
            v = d2[f"{kind}/{mon}"]
            rows.append(f"{kl} & {ml} & {v['harmful_settings']} & {v['plain']} & {v['raw_miss']} & "
                        f"{v['matched']} & {v['matched_miss']}\\\\")
            if kind == "code":
                assert v["raw_miss"] >= v["matched_miss"], (ds, mon, "raw < matched on code bugs")
            else:
                assert v["raw_miss"] == 0, (ds, mon, "raw caught a sanitization-dependent bug")
                assert v["matched_miss"] == v["harmful_settings"], (ds, mon, "matched+m missed one")
    harmless = dec2[~dec2.harmful]
    alerts = harmless.raw_false.sum() + harmless.matched_false.sum()
    assert alerts == 0, (ds, "alerts on harmless settings")
open(os.path.join(DST, "tab_detect2.tex"), "w").write("\n".join(rows) + "\n")

with open(os.path.join(DST, "numbers.tex"), "w") as fh:
    for k, v in sorted(macros.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(macros), "macros written")
