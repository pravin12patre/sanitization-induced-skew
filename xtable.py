"""Run 6 (retail only): cross-table masking mismatch.

A customer dimension (country) is joined to facts by masked customer id to
build is_domestic (country == United Kingdom). Facts use full key rotation per
monthly refresh, which Proposition 1 shows is skew-free on its own.
Scenarios for the dimension table:
  consistent     same key epoch and same identifier canonicalization
  format_mismatch dimension canonicalizes ids as floats ('12345.0') while
                 facts use integers ('12345'): every join misses
  stale_quarterly dimension refreshed quarterly, facts monthly: keys agree
                 only in quarter-start months
Join misses are filled with the default 0 ('unknown'), as pipelines commonly do.
"""
import json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import skew
from skew import (FEATURES, TRAIN_CUTOFFS, TEST_CUTOFFS, CUTOFFS, load_invoices,
                  online_table, features_at, sanitize, mask)

assert skew.DATASET == "retail"
F2 = FEATURES + ["is_domestic"]
inv = load_invoices()
raw = pd.read_parquet(skew.os.path.join(skew.HERE, "data", "raw.parquet"))
raw = raw[raw.CustomerID.notna()]
dim = (raw.groupby("CustomerID").Country.agg(lambda s: s.mode().iat[0]).reset_index())
dim["cust"] = dim.CustomerID.astype(int).astype(str)
dim["dom"] = (dim.Country == "United Kingdom").astype(float)

online = online_table(inv)
online["is_domestic"] = online.cust.map(dim.set_index("cust").dom)


def dim_key(t, scenario):
    if scenario == "stale_quarterly":
        q = pd.Timestamp(t).to_period("Q").start_time
        return f"k-{q.date()}"
    return f"k-{pd.Timestamp(t).date()}"


def backfill(scenario):
    rows = []
    for t in CUTOFFS:
        s = sanitize(inv, "rotate_full", 0, refresh_epoch=str(t.date()))
        f = features_at(s, t, "eid")
        k = dim_key(t, scenario)
        canon = (lambda c: str(float(c))) if scenario == "format_mismatch" else str
        d = pd.Series(dim.dom.values,
                      index=[mask(canon(c), k) for c in dim.cust])
        f["is_domestic"] = d.reindex(f.index).fillna(0.0).values
        vis = s[s.ts < t].sort_values("ts").groupby("cust").eid.last()
        spine = online[online.cutoff == t][["cust"]].copy()
        spine["eid"] = spine.cust.map(vis)
        b = spine.merge(f, left_on="eid", right_index=True, how="left")
        b["cutoff"] = t
        rows.append(b.drop(columns=["_leaky"], errors="ignore"))
    return pd.concat(rows, ignore_index=True)


def evaluate(back, seed=0):
    lab = online[["cust", "cutoff", "label"]]
    tr = back[back.cutoff.isin(TRAIN_CUTOFFS)].merge(lab, on=["cust", "cutoff"])
    te = back[back.cutoff.isin(TEST_CUTOFFS)].merge(lab, on=["cust", "cutoff"])
    on_te = online[online.cutoff.isin(TEST_CUTOFFS)]
    clf = HistGradientBoostingClassifier(random_state=seed).fit(tr[F2], tr.label)
    return (round(roc_auc_score(te.label, clf.predict_proba(te[F2])[:, 1]), 4),
            round(roc_auc_score(on_te.label, clf.predict_proba(on_te[F2])[:, 1]), 4))


res = {"share_domestic_online": round(float(online.is_domestic.mean()), 4)}
for sc in ["consistent", "format_mismatch", "stale_quarterly"]:
    b = backfill(sc)
    m = online.merge(b, on=["cust", "cutoff"], suffixes=("_on", "_bf"))
    match = float((m.is_domestic_on == m.is_domestic_bf).mean())
    joined = float(np.mean([t.month in (1, 4, 7, 10) for t in m.cutoff])) \
        if sc == "stale_quarterly" else (0.0 if sc == "format_mismatch" else 1.0)
    off, prod = evaluate(b)
    res[sc] = {"is_domestic_match": round(match, 4),
               "rows_in_matching_key_epoch": round(joined, 4),
               "offline_auc": off, "production_auc": prod}
    print(sc, res[sc], flush=True)
json.dump(res, open(skew.out("xtable.json"), "w"), indent=1)
