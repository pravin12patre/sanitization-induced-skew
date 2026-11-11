"""Run 7: cost and sensitivity of matched comparison vs sample fraction.

For each sample fraction p and sampling seed, run matched comparison on the
sampled customers only (sanitize sample, run reference and backfill code on
it, score with the monitor) and record bugs detected, false alarms, and
wall-clock seconds. The full sanitized backfill over the monitoring window is
timed once for scale.
"""
import json, time
import numpy as np, pandas as pd
import skew
from skew import (FEATURES, TEST_CUTOFFS, BUGS, load_invoices, online_table,
                  backfill_table)
from detect import (SANITIZATIONS, TAU, tfdv_scores, evidently_scores, window)

inv = load_invoices()
online = online_table(inv)
on_w = window(online)
allc = online.cust.unique()

# ground truth per bug (same rule as detect.py)
truth = {}
for b in BUGS:
    bb = window(backfill_table(inv, online, "none", 0, "shifted", bug=b, cutoffs=TEST_CUTOFFS))
    m = on_w.merge(bb, on=["cust", "cutoff"], suffixes=("_on", "_bf"))
    truth[b] = {f for f in FEATURES if (~np.isclose(m[f + "_on"], m[f + "_bf"],
                rtol=1e-9, atol=1e-6)).mean() >= 0.01}

t0 = time.perf_counter()
backfill_table(inv, online, "rotate_incremental", -100, "shifted", cutoffs=TEST_CUTOFFS)
full_backfill_s = time.perf_counter() - t0
print("full backfill seconds:", round(full_backfill_s, 2), flush=True)

monitors = {"tfdv": tfdv_scores, "evidently": evidently_scores}
rows = []
for p in [0.05, 0.1, 0.2, 0.5, 1.0]:
    for seed in ([0] if p == 1.0 else [0, 1, 2]):
        rng = np.random.default_rng(100 + seed)
        samp = set(rng.choice(allc, size=max(1, int(p * len(allc))), replace=False))
        inv_s = inv[inv.cust.isin(samp)]
        on_s = online[online.cust.isin(samp)]
        for sname, (km, sd) in SANITIZATIONS.items():
            for b in BUGS:
                for mon, fn in monitors.items():
                    # full cost of one check: sanitize the sample, run the
                    # reference and backfill code on it, score with the monitor
                    t0 = time.perf_counter()
                    ref_s = window(backfill_table(inv_s, on_s, km, sd, "shifted",
                                                  bug=None, cutoffs=TEST_CUTOFFS))
                    bf_s = window(backfill_table(inv_s, on_s, km, sd, "shifted",
                                                 bug=b, cutoffs=TEST_CUTOFFS))
                    sc = fn(bf_s, ref_s)
                    secs = time.perf_counter() - t0
                    alerts = {f for f in FEATURES if sc[f] > TAU}
                    rows.append({"fraction": p, "seed": seed, "sanitization": sname,
                                 "bug": str(b), "monitor": mon, "seconds": secs,
                                 "false_alarm": b is None and bool(alerts),
                                 "detected": b is not None and bool(alerts & truth[b])})
        print("done fraction", p, "seed", seed, flush=True)

df = pd.DataFrame(rows)
df.to_csv(skew.out("cost_runs.csv"), index=False)
summ = {}
for (mon, p), g in df.groupby(["monitor", "fraction"]):
    per_seed = g.groupby("seed").agg(det=("detected", "sum"), fa=("false_alarm", "sum"))
    summ[f"{mon}@{p}"] = {
        "detected_mean_of_16": round(float(per_seed.det.mean()), 2),
        "detected_min": int(per_seed.det.min()),
        "false_alarms_max": int(per_seed.fa.max()),
        "median_seconds_per_check": round(float(g.seconds.median()), 3),
        "relative_to_full_backfill": round(float(g.seconds.median()) / full_backfill_s, 3)}
    print(mon, p, summ[f"{mon}@{p}"], flush=True)
json.dump({"full_backfill_seconds": round(full_backfill_s, 3),
           "n_customers": int(len(allc)), "summary": summ},
          open(skew.out("cost.json"), "w"), indent=1)
