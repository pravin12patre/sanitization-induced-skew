"""Run 9: can the manifest alone predict which features sanitization skews?

Predictions come only from the propositions and the manifest's recorded
operations (no data access); observations are the exact-match rates of
Table 'match'. A feature counts as observed-skewed if it matches on < 99% of
rows.
"""
import json
import skew

REL = {"recency_days", "frequency", "monetary", "freq_90d", "spend_90d", "tenure_days"}
CAL = {"weekend_share", "q4_share", "last_dow"}
HISTORY = {"frequency", "monetary", "freq_90d", "spend_90d", "tenure_days",
           "weekend_share", "q4_share"}          # Prop. 4: all but recency, last_dow
CELLS = {  # cell -> (scope, shift_days, aligned)
    "rotate_full": ("full", 0, True),
    "rotate_incremental": ("incremental", 0, True),
    "shift_-98": ("none", -98, True),
    "shift_-100": ("none", -100, True),
    "shift_-100_unshifted_cutoff": ("none", -100, False),
    "HEADLINE_rotate_incr+shift_-100": ("incremental", -100, True),
}


def predict(scope, d, aligned):
    p = set()
    if scope == "incremental":
        p |= HISTORY                                  # Prop. 4
    if d != 0 and not aligned:
        p |= REL | CAL                                # Prop. 3
    if d != 0:
        p.add("q4_share")                             # Prop. 2, month membership
        if d % 7:
            p |= {"weekend_share", "last_dow"}        # Prop. 2, P = 7 days
    return p


if __name__ == "__main__":
    res = json.load(open(skew.out("results.json")))["cells"]
    tp = fp = fn = tn = 0
    errors = []
    for cell, cfg in CELLS.items():
        pred = predict(*cfg)
        for f in skew.FEATURES:
            obs = res[cell]["features"][f]["match_rate"] < 0.99
            p = f in pred
            tp += p and obs; fp += p and not obs; fn += obs and not p; tn += not p and not obs
            if p != obs:
                errors.append({"cell": cell, "feature": f, "predicted": p,
                               "match_rate": res[cell]["features"][f]["match_rate"]})
    out = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "cells": len(CELLS),
           "decisions": tp + fp + fn + tn, "errors": errors}
    print(json.dumps(out, indent=1))
    json.dump(out, open(skew.out("predict.json"), "w"), indent=1)
