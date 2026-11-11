"""Run 3: can a skew monitor find genuine bugs under sanitization?

Monitoring window: the 4 test cutoffs. "Training" side = backfilled
features (possibly sanitized, possibly buggy); "serving" side = online
features. Monitors:
  TFDV       skew comparator: approx. Jensen-Shannon divergence (numeric),
             L-infinity (categorical last_dow), threshold 0.1 each
  Evidently  DataDriftPreset defaults (Wasserstein normed / Jensen-Shannon
             distance, threshold 0.1)
Variants:
  plain     alert if score > threshold
  rules     manifest without raw access: suppress features the recorded
            sanitization is known to affect (H1, H5)
  replay    manifest with a secure-zone replay: run the SAME production
            backfill code on a 20% customer sample twice (raw history vs
            history sanitized per the manifest); expected score = score
            between those two; alert if observed - expected > threshold
Ground truth: a feature is genuinely skewed when the bug alone (no
sanitization) changes it on >= 1% of rows.
"""
import json, os, sys, contextlib, io
import numpy as np, pandas as pd
import skew
from skew import FEATURES, TEST_CUTOFFS, BUGS, load_invoices, online_table, backfill_table

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow_data_validation as tfdv
from evidently import Report
from evidently.presets import DataDriftPreset

TAU = 0.1
SANITIZATIONS = {
    "none": ("none", 0),
    "rotate_incremental": ("rotate_incremental", 0),
    "shift_-100": ("none", -100),
    "observed_rotate+shift": ("rotate_incremental", -100),
}
HISTORY = ["frequency", "monetary", "tenure_days", "freq_90d", "spend_90d",
           "weekend_share", "q4_share"]


def rule_affected(key_mode, shift):
    a = set()
    if key_mode == "rotate_incremental":
        a |= set(HISTORY)
    if shift != 0:
        a.add("q4_share")
        if shift % 7:
            a |= {"weekend_share", "last_dow"}
    return a


def prep(df):
    d = df[FEATURES].copy()
    d["last_dow"] = d.last_dow.map(lambda v: "missing" if pd.isna(v) else str(int(v)))
    return d.reset_index(drop=True)


def tfdv_scores(train, serve):
    st, ss = (tfdv.generate_statistics_from_dataframe(prep(x)) for x in (train, serve))
    sch = tfdv.infer_schema(st)
    for f in FEATURES:
        c = tfdv.get_feature(sch, f).skew_comparator
        if f == "last_dow":
            c.infinity_norm.threshold = TAU
        else:
            c.jensen_shannon_divergence.threshold = TAU
    an = tfdv.validate_statistics(statistics=st, schema=sch, serving_statistics=ss)
    out = {f: 0.0 for f in FEATURES}
    for d in an.drift_skew_info:
        for m in d.skew_measurements:
            out[d.path.step[0]] = max(out[d.path.step[0]], float(m.value))
    return out


def evidently_scores(train, serve):
    with contextlib.redirect_stdout(io.StringIO()):
        s = Report([DataDriftPreset()]).run(current_data=prep(train),
                                            reference_data=prep(serve)).dict()
    out = {}
    for m in s["metrics"]:
        if m["config"].get("type", "").endswith("ValueDrift"):
            out[m["config"]["column"]] = float(m["value"])
    return out


def window(df, custs=None):
    d = df[df.cutoff.isin(TEST_CUTOFFS)]
    return d if custs is None else d[d.cust.isin(custs)]


def prf(rows):
    tp = sum(r["alert"] and r["truth"] for r in rows)
    fp = sum(r["alert"] and not r["truth"] for r in rows)
    fn = sum((not r["alert"]) and r["truth"] for r in rows)
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3),
            "recall": round(r, 3)}


if __name__ == "__main__":
    inv = load_invoices()
    online = online_table(inv)
    on_w = window(online)
    rng = np.random.default_rng(7)
    allc = online.cust.unique()
    sample = set(rng.choice(allc, size=int(0.2 * len(allc)), replace=False))

    # ground truth per bug
    truth = {}
    for b in BUGS:
        bb = window(backfill_table(inv, online, "none", 0, "shifted", bug=b,
                                   cutoffs=TEST_CUTOFFS))
        m = on_w.merge(bb, on=["cust", "cutoff"], suffixes=("_on", "_bf"))
        truth[b] = {f: bool((~np.isclose(m[f + "_on"], m[f + "_bf"],
                                         rtol=1e-9, atol=1e-6)).mean() >= 0.01)
                    for f in FEATURES}
        print("truth", b, [f for f, v in truth[b].items() if v], flush=True)

    monitors = {"tfdv": tfdv_scores, "evidently": evidently_scores}
    rows = []
    for sname, (km, sd) in SANITIZATIONS.items():
        aff = rule_affected(km, sd)
        for b in BUGS:
            obs = window(backfill_table(inv, online, km, sd, "shifted", bug=b,
                                        cutoffs=TEST_CUTOFFS))
            raw_s = window(backfill_table(inv, online, "none", 0, "shifted", bug=b,
                                          cutoffs=TEST_CUTOFFS), sample)
            san_s = window(backfill_table(inv, online, km, sd, "shifted", bug=b,
                                          cutoffs=TEST_CUTOFFS), sample)
            # matched: sanitize BOTH sides per the manifest; serving-side
            # (reference, bug-free) feature code on the sanitized sample
            ref_s = window(backfill_table(inv, online, km, sd, "shifted", bug=None,
                                          cutoffs=TEST_CUTOFFS), sample)
            for mon, fn in monitors.items():
                so = fn(obs, on_w)
                se = fn(san_s, raw_s)
                sm = fn(san_s, ref_s)
                for f in FEATURES:
                    t = truth[b][f]
                    rows.append({"monitor": mon, "sanitization": sname,
                                 "bug": str(b), "feature": f, "truth": t,
                                 "score_obs": round(so[f], 4),
                                 "score_exp": round(se[f], 4),
                                 "plain": so[f] > TAU,
                                 "rules": so[f] > TAU and f not in aff,
                                 "score_matched": round(sm[f], 4),
                                 "replay": so[f] - se[f] > TAU,
                                 "matched": sm[f] > TAU})
            print("done", sname, b, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(skew.out("detect_decisions.csv"), index=False)
    summary = {}
    for mon in monitors:
        for var in ["plain", "rules", "replay", "matched"]:
            d = df[df.monitor == mon]
            r = [{"alert": a, "truth": t} for a, t in zip(d[var], d.truth)]
            s = prf(r)
            # scenario level: bug caught if any truly-skewed feature alerts;
            # false alarm if a bug-free scenario raises any alert
            g = d.groupby(["sanitization", "bug"])
            caught = g.apply(lambda x: bool((x[var] & x.truth).any()), include_groups=False)
            bugged = caught[caught.index.get_level_values("bug") != "None"]
            fa = g.apply(lambda x: bool(x[var].any()), include_groups=False)
            fa = fa[fa.index.get_level_values("bug") == "None"]
            s["bugs_caught"] = f"{int(bugged.sum())}/{len(bugged)}"
            # distinguishable: bug changes the alert set vs the same
            # sanitization without a bug (what an operator can actually see)
            alerts = g.apply(lambda x: frozenset(x.feature[x[var]]), include_groups=False)
            dist = [alerts[(sn, b)] != alerts[(sn, "None")] and
                    bool(set(alerts[(sn, b)]) - set(alerts[(sn, "None")]) &
                         {f for f, v in truth[None if b == "None" else b].items() if v})
                    for sn, b in alerts.index if b != "None"]
            s["bugs_distinguishable"] = f"{sum(dist)}/{len(dist)}"
            s["observed_case_distinguishable"] = "{}/4".format(sum(
                d for (sn, b), d in zip([i for i in alerts.index if i[1] != "None"], dist)
                if sn == "observed_rotate+shift"))
            s["bug_free_scenarios_alerting"] = f"{int(fa.sum())}/{len(fa)}"
            summary[f"{mon}/{var}"] = s
            print(mon, var, s, flush=True)
    json.dump({"tau": TAU, "truth": truth, "summary": summary},
              open(skew.out("detect.json"), "w"), indent=1)
