"""Paired production-AUC drop vs baseline (same seeds, same bootstrap draws)."""
import json, numpy as np
import skew
from skew import load_invoices, online_table, backfill_table, TEST_CUTOFFS
from robustness import fit_predict, boot_auc, ci, B, SEEDS
rng = np.random.default_rng(12345)
inv = load_invoices(); online = online_table(inv)
custs = np.array(sorted(online[online.cutoff.isin(TEST_CUTOFFS)].cust.unique()))
idx = [rng.integers(0, len(custs), len(custs)) for _ in range(B)]
def draws(back):
    return np.vstack([boot_auc(*fit_predict(online, back, s), custs, idx[s::len(SEEDS)]) for s in SEEDS])
base = draws(online)
out = {}
cfgs = {"rotate_incremental_monthly": ("rotate_incremental", 0, "shifted", "M"),
        "rotate_incremental_weekly": ("rotate_incremental", 0, "shifted", "W"),
        "shift_-100": ("none", -100, "shifted", "M"),
        "observed_rotate_incr+shift_-100": ("rotate_incremental", -100, "shifted", "M"),
        "observed+unshifted_cutoff": ("rotate_incremental", -100, "unshifted", "M")}
for k, (km, sd, cm, f) in cfgs.items():
    skew.EPOCH_FREQ = f
    d = draws(backfill_table(inv, online, km, sd, cm))
    out[k] = {"production_auc_drop_vs_baseline": ci(base[:, 1] - d[:, 1])}
    print(k, out[k], flush=True)
json.dump(out, open(skew.out("paired.json"), "w"), indent=1)
