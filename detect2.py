"""Run 8: which bugs does each detector catch?

Detectors, all scoring backfill-code output against reference (serving) code
(the raw and matched detectors are also run with an exact missing-value-rate
check added, since both sides share inputs):
  plain    sanitized backfill vs online features (counted only if the bug
           adds an alert beyond the same setting without a bug)
  raw      differential test on a raw 20% sample: backfill code vs reference
           code, no sanitization
  matched  the same test on the 20% sample after replaying the sanitization

Ground truth is per setting: a bug is harmful in a sanitization setting if it
changes the backfilled training features (vs. the bug-free backfill in the
same setting) on >= 1% of rows for some feature. Bugs:
  code bugs                 window60, wrong_column, retention365, tz_shift
  sanitization-dependent    hardcoded_start, test_account_prefix (no-ops on
                            raw data, see skew.SAN_BUGS)
"""
import json
import numpy as np, pandas as pd
import skew
from skew import FEATURES, TEST_CUTOFFS, BUGS, SAN_BUGS, load_invoices, online_table, backfill_table
from detect import SANITIZATIONS, TAU, tfdv_scores, evidently_scores, window

inv = load_invoices()
online = online_table(inv)
on_w = window(online)
rng = np.random.default_rng(7)
allc = online.cust.unique()
sample = set(rng.choice(allc, size=int(0.2 * len(allc)), replace=False))
inv_s, on_s = inv[inv.cust.isin(sample)], online[online.cust.isin(sample)]
monitors = {"tfdv": tfdv_scores, "evidently": evidently_scores}
bugs = [b for b in BUGS if b is not None] + list(SAN_BUGS)


def bf(data, onl, km, sd, bug):
    return window(backfill_table(data, onl, km, sd, "shifted", bug=bug, cutoffs=TEST_CUTOFFS))


def changed(a, b):
    m = a.merge(b, on=["cust", "cutoff"], suffixes=("_a", "_b"))
    out = set()
    for f in FEATURES:
        x, y = m[f + "_a"], m[f + "_b"]
        same = np.isclose(x, y, rtol=1e-9, atol=1e-6) | (x.isna() & y.isna())
        if (~same).mean() >= 0.01:
            out.add(f)
    return out


def missing_shift(a, b, tol=0.01):
    """Features whose missing-value rate differs by more than tol. Matched
    comparison feeds identical inputs to both sides, so any difference is
    caused by code."""
    return {f for f in FEATURES if abs(a[f].isna().mean() - b[f].isna().mean()) > tol}


rows = []
raw_ref = bf(inv_s, on_s, "none", 0, None)
for sname, (km, sd) in SANITIZATIONS.items():
    clean_full = bf(inv, online, km, sd, None)
    san_ref = bf(inv_s, on_s, km, sd, None)
    plain_clean = {mon: {f for f, v in fn(clean_full, on_w).items() if v > TAU}
                   for mon, fn in monitors.items()}
    for b in bugs:
        full = bf(inv, online, km, sd, b)
        truth = changed(full, clean_full)
        raw_b = bf(inv_s, on_s, "none", 0, b)
        san_b = bf(inv_s, on_s, km, sd, b)
        miss_raw = missing_shift(raw_b, raw_ref)
        miss_mat = missing_shift(san_b, san_ref)
        for mon, fn in monitors.items():
            plain = {f for f, v in fn(full, on_w).items() if v > TAU}
            raw = {f for f, v in fn(raw_b, raw_ref).items() if v > TAU}
            mat = {f for f, v in fn(san_b, san_ref).items() if v > TAU}
            rows.append({"sanitization": sname, "bug": b, "monitor": mon,
                         "kind": "sanitization-dependent" if b in SAN_BUGS else "code",
                         "harmful": bool(truth), "truth": sorted(truth),
                         "plain": bool((plain - plain_clean[mon]) & truth),
                         "raw": bool(raw & truth), "matched": bool(mat & truth),
                         "raw_miss": bool((raw | miss_raw) & truth),
                         "matched_miss": bool((mat | miss_mat) & truth),
                         "raw_false": bool(raw - truth) if truth else bool(raw),
                         "matched_false": bool(mat - truth) if truth else bool(mat)})
        print("done", sname, b, sorted(truth), flush=True)

df = pd.DataFrame(rows)
df.to_csv(skew.out("detect2_decisions.csv"), index=False)
summ = {}
for (kind, mon), g in df.groupby(["kind", "monitor"]):
    h = g[g.harmful]
    summ[f"{kind}/{mon}"] = {"harmful_settings": int(len(h)),
                             "plain": int(h.plain.sum()), "raw": int(h.raw.sum()),
                             "matched": int(h.matched.sum()),
                             "raw_miss": int(h.raw_miss.sum()),
                             "matched_miss": int(h.matched_miss.sum()),
                             "harmless_settings": int((~g.harmful).sum()),
                             "matched_alerts_on_harmless": int(g[~g.harmful].matched_false.sum()),
                             "raw_alerts_on_harmless": int(g[~g.harmful].raw_false.sum())}
    print(kind, mon, summ[f"{kind}/{mon}"], flush=True)
json.dump(summ, open(skew.out("detect2.json"), "w"), indent=1)
