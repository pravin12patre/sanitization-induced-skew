"""Run 5: collisions from truncated masked identifiers (stable key, no shift).

For m = 16^h possible masked values and n entities, the probability that a
given entity shares its masked value with at least one other is
1 - (1 - 1/m)^(n-1) (birthday bound). Merged entities inflate counts and sums.
"""
import json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import skew
from skew import load_invoices, online_table, backfill_table, divergence, mask
from robustness import fit_predict

inv = load_invoices()
online = online_table(inv)
custs = inv.cust.unique()
n = len(custs)
res = {"n_entities": int(n), "cells": {}}
for h in [3, 4, 5, 6, 8, 16]:
    skew.MASK_HEX = h
    m = 16 ** h
    ids = pd.Series([mask(c, "k-stable") for c in custs])
    shared = float(ids.duplicated(keep=False).mean())
    expected = 1 - (1 - 1 / m) ** (n - 1)
    back = backfill_table(inv, online, "stable", 0, "shifted")
    div, _ = divergence(online, back)
    pb, po = fit_predict(online, back, 0)
    res["cells"][f"{4 * h}_bits"] = {
        "entities_sharing_id": round(shared, 5),
        "birthday_expected": round(expected, 5),
        "frequency_match": div["frequency"]["match_rate"],
        "monetary_match": div["monetary"]["match_rate"],
        "offline_auc": round(roc_auc_score(pb.label, pb.p), 4),
        "production_auc": round(roc_auc_score(po.label, po.p), 4)}
    print(4 * h, "bits", res["cells"][f"{4 * h}_bits"], flush=True)
skew.MASK_HEX = 16
json.dump(res, open(skew.out("collide.json"), "w"), indent=1)
