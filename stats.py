"""Dataset statistics used in the paper."""
import json
import skew
from skew import load_invoices, online_table, TRAIN_CUTOFFS, TEST_CUTOFFS
inv = load_invoices(); on = online_table(inv)
s = {"orders": int(len(inv)), "customers": int(inv.cust.nunique()),
     "rows": int(len(on)), "train_rows": int(on.cutoff.isin(TRAIN_CUTOFFS).sum()),
     "test_rows": int(on.cutoff.isin(TEST_CUTOFFS).sum()),
     "base_rate": round(float(on.label.mean()), 4),
     "first": str(inv.ts.min().date()), "last": str(inv.ts.max().date()),
     "train_cutoffs": [str(TRAIN_CUTOFFS[0].date()), str(TRAIN_CUTOFFS[-1].date())],
     "test_cutoffs": [str(TEST_CUTOFFS[0].date()), str(TEST_CUTOFFS[-1].date())]}
json.dump(s, open(skew.out("stats.json"), "w"), indent=1); print(s)
