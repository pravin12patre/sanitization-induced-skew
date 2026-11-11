"""Run 4: forward (positive) global shift with unshifted cutoffs -> staleness, not leakage."""
import json, numpy as np
import skew
from sklearn.metrics import roc_auc_score
from skew import load_invoices, online_table, backfill_table, divergence, FEATURES
from robustness import fit_predict
inv = load_invoices(); online = online_table(inv)
out = {}
for d in [7, 30, 100]:
    for cm in ["shifted", "unshifted"]:
        b = backfill_table(inv, online, "none", d, cm)
        div, m = divergence(online, b)
        pb, po = fit_predict(online, b, 0)
        # staleness: share of rows whose backfill misses >=1 true past event
        miss = float((m.frequency_bf.fillna(0) < m.frequency_on).mean())
        out[f"+{d}_{cm}"] = {"leak_rate": round(float(b.leak.mean()), 4),
            "rows_missing_past_events": round(miss, 4),
            "offline_auc": round(roc_auc_score(pb.label, pb.p), 4),
            "production_auc": round(roc_auc_score(po.label, po.p), 4)}
        print(f"+{d} {cm}", out[f"+{d}_{cm}"], flush=True)
json.dump(out, open(skew.out("forward.json"), "w"), indent=1)
