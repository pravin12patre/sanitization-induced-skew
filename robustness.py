"""Run 2: uncertainty and sweeps.

1. Headline cells: 5 model seeds x customer-cluster bootstrap (B=200) for
   offline AUC, production AUC, their gap, and the leak rate.
2. Shift-offset sweep (shifted cutoffs): which calendar features break.
3. Shift-offset sweep (unshifted cutoffs): leak rate and AUC gap vs offset.
4. Key-epoch sweep for incremental rotation: weekly/monthly/quarterly/yearly.
"""
import json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import skew
from skew import (FEATURES, TRAIN_CUTOFFS, TEST_CUTOFFS, load_invoices,
                  online_table, backfill_table, divergence)

B, SEEDS = 200, [0, 1, 2, 3, 4]
rng = np.random.default_rng(12345)


def fit_predict(online, back, seed):
    lab = online[["cust", "cutoff", "label"]]
    back = back.drop(columns=["label"], errors="ignore")
    tr = back[back.cutoff.isin(TRAIN_CUTOFFS)].merge(lab, on=["cust", "cutoff"])
    te_bf = back[back.cutoff.isin(TEST_CUTOFFS)].merge(lab, on=["cust", "cutoff"])
    te_on = online[online.cutoff.isin(TEST_CUTOFFS)]
    clf = HistGradientBoostingClassifier(random_state=seed).fit(tr[FEATURES], tr.label)
    pb = te_bf.assign(p=clf.predict_proba(te_bf[FEATURES])[:, 1])[["cust", "label", "p"]]
    po = te_on.assign(p=clf.predict_proba(te_on[FEATURES])[:, 1])[["cust", "label", "p"]]
    return pb, po


def boot_auc(pb, po, custs, idx_sets):
    """Customer-cluster bootstrap: resample customers, keep all their rows."""
    out = []
    gb = {c: i for i, c in enumerate(custs)}
    ib, io = pb.cust.map(gb).values, po.cust.map(gb).values
    for s in idx_sets:
        w = np.bincount(s, minlength=len(custs))
        wb, wo = w[ib], w[io]
        ob = roc_auc_score(pb.label, pb.p, sample_weight=wb)
        oo = roc_auc_score(po.label, po.p, sample_weight=wo)
        out.append((ob, oo, ob - oo))
    return np.array(out)


def ci(x):
    return [round(float(np.mean(x)), 4), round(float(np.percentile(x, 2.5)), 4),
            round(float(np.percentile(x, 97.5)), 4)]


if __name__ == "__main__":
    inv = load_invoices()
    online = online_table(inv)
    custs = np.array(sorted(online[online.cutoff.isin(TEST_CUTOFFS)].cust.unique()))
    idx_sets = [rng.integers(0, len(custs), len(custs)) for _ in range(B)]
    res = {"n_test_customers": int(len(custs)), "bootstrap_B": B, "seeds": SEEDS}

    # 1. headline cells with uncertainty
    cells = {
        "baseline": None,
        "rotate_incremental": ("rotate_incremental", 0, "shifted"),
        "shift_-100": ("none", -100, "shifted"),
        "shift_-100_unshifted_cutoff": ("none", -100, "unshifted"),
        "observed_rotate_incr+shift_-100": ("rotate_incremental", -100, "shifted"),
        "observed+unshifted_cutoff": ("rotate_incremental", -100, "unshifted"),
    }
    res["cells"] = {}
    for name, cfg in cells.items():
        back = online if cfg is None else backfill_table(inv, online, *cfg)
        draws = np.vstack([boot_auc(*fit_predict(online, back, s), custs,
                                    idx_sets[s::len(SEEDS)]) for s in SEEDS])
        leak = back.leak.values if "leak" in back else np.zeros(len(back))
        # leak CI by customer-cluster bootstrap
        lc = back.assign(leak=leak).groupby("cust").leak.agg(["sum", "count"])
        lw = [lc.sample(len(lc), replace=True, random_state=i) for i in range(B)]
        lr = [float(x["sum"].sum() / x["count"].sum()) for x in lw]
        res["cells"][name] = {"offline_auc": ci(draws[:, 0]),
                              "production_auc": ci(draws[:, 1]),
                              "gap": ci(draws[:, 2]), "leak_rate": ci(lr)}
        print(name, res["cells"][name], flush=True)

    # 2./3. shift sweep
    res["shift_sweep"] = {}
    for d in [-1, -7, -28, -30, -91, -100, -182, -364, -365]:
        row = {}
        for cm in ["shifted", "unshifted"]:
            back = backfill_table(inv, online, "none", d, cm)
            div, _ = divergence(online, back)
            pb, po = fit_predict(online, back, 0)
            row[cm] = {"leak_rate": round(float(back.leak.mean()), 4),
                       "offline_auc": round(roc_auc_score(pb.label, pb.p), 4),
                       "production_auc": round(roc_auc_score(po.label, po.p), 4),
                       "match": {c: div[c]["match_rate"] for c in FEATURES}}
        res["shift_sweep"][d] = row
        print("shift", d, {k: (v["leak_rate"], v["offline_auc"], v["production_auc"])
                           for k, v in row.items()}, flush=True)

    # 4. key-epoch sweep
    res["epoch_sweep"] = {}
    for freq, label in [("W", "weekly"), ("M", "monthly"), ("Q", "quarterly"),
                        ("Y", "yearly")]:
        skew.EPOCH_FREQ = freq
        back = backfill_table(inv, online, "rotate_incremental", 0, "shifted")
        div, _ = divergence(online, back)
        pb, po = fit_predict(online, back, 0)
        res["epoch_sweep"][label] = {
            "frequency_match": div["frequency"]["match_rate"],
            "tenure_match": div["tenure_days"]["match_rate"],
            "offline_auc": round(roc_auc_score(pb.label, pb.p), 4),
            "production_auc": round(roc_auc_score(po.label, po.p), 4)}
        print("epoch", label, res["epoch_sweep"][label], flush=True)
    skew.EPOCH_FREQ = "M"
    json.dump(res, open(skew.out("robustness.json"), "w"), indent=1)
