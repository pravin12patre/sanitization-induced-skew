"""Paper figures, read directly from the result files.
Usage: python3 make_figures.py <output_dir>"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DST = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "paper")


def points(ds):
    rob = json.load(open(os.path.join(HERE, f"out_{ds}", "robustness.json")))
    fwd = json.load(open(os.path.join(HERE, f"out_{ds}", "forward.json")))
    base = json.load(open(os.path.join(HERE, f"out_{ds}", "results.json")))["baseline_auc"]
    pts = []
    for d, v in rob["shift_sweep"].items():
        u = v["unshifted"]
        pts.append((int(d), u["offline_auc"], u["production_auc"], u["leak_rate"], 0.0))
    for k, v in fwd.items():
        if k.endswith("_unshifted"):
            pts.append((int(k.split("_")[0]), v["offline_auc"], v["production_auc"],
                        v["leak_rate"], v["rows_missing_past_events"]))
    return sorted(pts), base


plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(7.1, 3.3), sharex=True,
                         gridspec_kw={"height_ratios": [1.6, 1]})
for col, (ds, title) in enumerate([("retail", "Online Retail II"), ("cdnow", "CDNOW")]):
    pts, base = points(ds)
    xs = range(len(pts))
    a1, a2 = axes[0][col], axes[1][col]
    a1.set_title(title, fontsize=9)
    a1.plot(xs, [p[1] for p in pts], "o-", color="0.6", ms=3, lw=1.2, label="Offline AUC (backfill)")
    a1.plot(xs, [p[2] for p in pts], "o-", color="#1f5fbf", ms=3, lw=1.5, label="Production AUC")
    a1.axhline(base, ls="--", lw=0.8, color="0.3")
    a1.text(-0.3, base + 0.008, "unsanitized baseline", ha="left", fontsize=7, color="0.3")
    a1.set_ylim(0.45, 1.03)
    a2.bar(xs, [100 * p[3] for p in pts], color="#c0392b", alpha=0.75, label="rows with future events (leakage)")
    a2.bar(xs, [100 * p[4] for p in pts], color="0.55", label="rows missing past events (staleness)")
    a2.set_xticks(list(xs))
    a2.set_xticklabels([f"{p[0]:+d}" for p in pts], rotation=45)
    a2.set_ylim(0, 100)
    a2.set_yticks([0, 20, 40, 60, 80])
    if col == 0:
        a1.set_ylabel("AUC")
        a2.set_ylabel("% of rows")
        a1.legend(loc="lower right", frameon=False, fontsize=7)
        a2.legend(loc="upper center", frameon=False, fontsize=6.5)
fig.supxlabel("Global date offset $\\delta$ (days), cutoffs not shifted", fontsize=8)
fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=1.0)
fig.savefig(os.path.join(DST, "fig_cutoff.pdf"))
fig.savefig(os.path.join(DST, "fig_cutoff.png"), dpi=200)
print("ok")
