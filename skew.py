"""Sanitization-induced skew: baseline pipeline + headline experiment.

Data: UCI Online Retail II (Chen, 2012, doi:10.24432/C5CG6D, CC BY 4.0).

Setup
-----
* "Online" features: point-in-time features computed from the true history at
  each monthly cutoff, i.e. what a model was trained on and served.
* "Backfill" features: the same feature code run over a sanitized copy.
* Sanitization knobs:
    key_mode   none | stable | rotate_full | rotate_incremental
               rotate_full        = every refresh re-masks the whole history
                                    with a new key (copy is self-consistent)
               rotate_incremental = each refresh masks only the new delta with
                                    that refresh's key and appends it
    shift_days global date offset (0, -98, -100)
    cutoff_mode shifted   = backfill cutoffs shifted by the same offset
                unshifted = cutoffs come from an unshifted source (e.g. a
                            label/prediction log kept outside the copy)
"""
import hashlib, hmac, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

FEATURES = ["recency_days", "frequency", "monetary", "freq_90d", "spend_90d",
            "tenure_days", "weekend_share", "q4_share", "last_dow"]
# Dataset selection: DATASET=retail (Online Retail II, default) or cdnow
DATASET = os.environ.get("DATASET", "retail")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_" + DATASET)
os.makedirs(OUT, exist_ok=True)
if DATASET == "retail":
    TRAIN_CUTOFFS = pd.date_range("2010-06-01", "2011-05-01", freq="MS")
    TEST_CUTOFFS = pd.date_range("2011-06-01", "2011-09-01", freq="MS")
    DATA_START = pd.Timestamp("2009-12-01")
elif DATASET == "cdnow":
    TRAIN_CUTOFFS = pd.date_range("1997-04-01", "1997-12-01", freq="MS")
    TEST_CUTOFFS = pd.date_range("1998-01-01", "1998-04-01", freq="MS")
    DATA_START = pd.Timestamp("1997-01-01")
else:
    raise ValueError(DATASET)
CUTOFFS = TRAIN_CUTOFFS.append(TEST_CUTOFFS)
LABEL_DAYS = 90
EPOCH_FREQ = "M"  # key-epoch period for rotate_incremental (refresh cadence)
MASK_HEX = 16     # hex characters kept from the HMAC (16 = 64 bits)


def out(name):
    return os.path.join(OUT, name)


def load_invoices():
    return load_retail() if DATASET == "retail" else load_cdnow()


def load_cdnow(path=os.path.join(HERE, "data", "CDNOW_master.txt")):
    """CDNOW master file -> one row per customer-day order. Dates have day
    resolution; 'price_sum' holds the CD count (the wrong-column bug's column)."""
    df = pd.read_csv(path, sep=r"\s+", header=None, skiprows=1,
                     names=["cust", "date", "cds", "dollars"], dtype={"cust": str})
    df = df[(df.dollars > 0) & (df.cds > 0)]
    df["ts"] = pd.to_datetime(df.date.astype(str), format="%Y%m%d")
    inv = (df.groupby(["cust", "ts"])
             .agg(amount=("dollars", "sum"), price_sum=("cds", "sum"))
             .reset_index())
    inv["Invoice"] = inv.cust + "-" + inv.ts.dt.strftime("%Y%m%d")
    return inv[["Invoice", "cust", "ts", "amount", "price_sum"]]


def load_retail(path=os.path.join(HERE, "data", "raw.parquet")):
    """Clean lines -> one row per invoice (customer, true timestamp, amount)."""
    df = pd.read_parquet(path)
    df = df[df.CustomerID.notna() & ~df.Invoice.str.startswith("C")
            & (df.Quantity > 0) & (df.Price > 0)].copy()
    df["amount"] = df.Quantity * df.Price
    inv = (df.groupby("Invoice")
             .agg(cust=("CustomerID", "first"), ts=("InvoiceDate", "min"),
                  amount=("amount", "sum"), price_sum=("Price", "sum"))
             .reset_index())
    inv["cust"] = inv.cust.astype(int).astype(str)
    inv["ts"] = pd.to_datetime(inv.ts)
    return inv


def mask(cust, key):
    return hmac.new(key.encode(), cust.encode(), hashlib.sha256).hexdigest()[:MASK_HEX]


def sanitize(inv, key_mode, shift_days, refresh_epoch):
    """Return a sanitized copy as of a refresh: columns eid, ts (shifted),
    true_ts, amount, cust (kept ONLY for the experimenter's join)."""
    s = inv.copy()
    if key_mode == "none":
        s["eid"] = s.cust
    elif key_mode == "stable":
        s["eid"] = [mask(c, "k-stable") for c in s.cust]
    elif key_mode == "rotate_full":
        # whole history re-masked with the key of the current refresh
        s["eid"] = [mask(c, f"k-{refresh_epoch}") for c in s.cust]
    elif key_mode == "rotate_incremental":
        # each event was masked when its month's delta was loaded
        ep = s.ts.dt.to_period(EPOCH_FREQ).astype(str)
        s["eid"] = [mask(c, f"k-{e}") for c, e in zip(s.cust, ep)]
    else:
        raise ValueError(key_mode)
    s["true_ts"] = s.ts
    s["ts"] = s.ts + pd.Timedelta(days=shift_days)
    return s


BUGS = {  # genuine pipeline bugs injected into backfill code only
    None: "no bug",
    "window60": "90-day window implemented as 60 days",
    "wrong_column": "monetary sums unit price instead of line amount",
    "retention365": "source table keeps only the last 365 days",
    "tz_shift": "timestamps converted with a +12 h timezone error",
}


# Sanitization-dependent bugs: backfill-only code that is a no-op on raw data
# and only changes features once the data is sanitized.
SAN_BUGS = {
    "hardcoded_start": "drops events before the system's first recorded date "
                       "(no-op on raw data; drops shifted events)",
    "test_account_prefix": "drops 'internal test accounts' whose identifier "
                           "starts with '9' (no raw identifier does; ~1/16 of "
                           "masked identifiers do)",
}


def features_at(events, t, id_col, bug=None):
    """Point-in-time features from events strictly before t, per id_col."""
    e = events
    if bug == "hardcoded_start":
        e = e[e.ts >= DATA_START]
    if bug == "test_account_prefix":
        e = e[~e[id_col].astype(str).str.startswith("9")]
    if bug == "tz_shift":
        e = e.assign(ts=e.ts + pd.Timedelta(hours=12))
    e = e[e.ts < t]
    if bug == "retention365":
        e = e[e.ts >= t - pd.Timedelta(days=365)]
    e = e.assign(_wk=(e.ts.dt.dayofweek >= 5).astype(float),
                 _q4=e.ts.dt.month.isin([11, 12]).astype(float),
                 _amt=e.price_sum if bug == "wrong_column" else e.amount)
    g = e.groupby(id_col)
    last = g.ts.max()
    f = pd.DataFrame({
        "recency_days": (t - last).dt.total_seconds() / 86400,
        "frequency": g.size().astype(float),
        "monetary": g._amt.sum(),
        "tenure_days": (t - g.ts.min()).dt.total_seconds() / 86400,
        "weekend_share": g._wk.mean(),
        "q4_share": g._q4.mean(),
        "last_dow": last.dt.dayofweek.astype(float),
    })
    win = 60 if bug == "window60" else 90
    e = e.assign(amount=e._amt)
    w = e[e.ts >= t - pd.Timedelta(days=win)].groupby(id_col)
    f["freq_90d"] = w.size().astype(float)
    f["spend_90d"] = w.amount.sum()
    f[["freq_90d", "spend_90d"]] = f[["freq_90d", "spend_90d"]].fillna(0.0)
    # leakage bookkeeping: did any included event truly happen at/after t?
    if "true_ts" in e:
        f["_leaky"] = g.true_ts.max()
    return f


def online_table(inv):
    rows = []
    for t in CUTOFFS:
        f = features_at(inv, t, "cust")
        fut = inv[(inv.ts >= t) & (inv.ts < t + pd.Timedelta(days=LABEL_DAYS))]
        f["label"] = f.index.isin(fut.cust).astype(int)
        f["cutoff"] = t
        rows.append(f.reset_index())
    return pd.concat(rows, ignore_index=True)


def backfill_table(inv, online, key_mode, shift_days, cutoff_mode, bug=None,
                   cutoffs=CUTOFFS):
    rows = []
    for t in cutoffs:
        s = sanitize(inv, key_mode, shift_days, refresh_epoch=str(t.date()))
        tq = t + pd.Timedelta(days=shift_days) if cutoff_mode == "shifted" else t
        f = features_at(s, tq, "eid", bug=bug)
        # The spine is the customer's most recent entity id visible at tq
        # (what the warehouse would key the row on). Experimenter-side join.
        vis = s[s.ts < tq].sort_values("ts").groupby("cust").eid.last()
        spine = online[online.cutoff == t][["cust"]].copy()
        spine["eid"] = spine.cust.map(vis)
        b = spine.merge(f, left_on="eid", right_index=True, how="left")
        b["leak"] = (b["_leaky"] >= t).astype(int)
        b["cutoff"] = t
        rows.append(b.drop(columns=["_leaky"]))
    return pd.concat(rows, ignore_index=True)


def psi(a, b, bins=10):
    a, b = a.dropna(), b.dropna()
    if len(b) == 0:
        return float("nan")
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0 if np.allclose(a.mean(), b.mean()) else float("inf")
    pa = np.histogram(np.clip(a, edges[0], edges[-1]), edges)[0] / len(a)
    pb = np.histogram(np.clip(b, edges[0], edges[-1]), edges)[0] / len(b)
    pa, pb = np.clip(pa, 1e-6, None), np.clip(pb, 1e-6, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def divergence(online, back):
    m = online.merge(back, on=["cust", "cutoff"], suffixes=("_on", "_bf"))
    out = {}
    for c in FEATURES:
        a, b = m[f"{c}_on"], m[f"{c}_bf"]
        match = np.isclose(a, b, rtol=1e-9, atol=1e-6) | (a.isna() & b.isna())
        out[c] = {"match_rate": round(float(match.mean()), 4),
                  "psi": round(psi(a, b), 4),
                  "ks": round(float(ks_2samp(a.dropna(), b.dropna()).statistic), 4)
                  if b.notna().any() else None}
    return out, m


def auc_gap(online, back, seed=0):
    """Train on backfilled features; report offline AUC (backfilled test)
    vs production AUC (online-served test)."""
    tr = back[back.cutoff.isin(TRAIN_CUTOFFS)].merge(
        online[["cust", "cutoff", "label"]], on=["cust", "cutoff"])
    te_bf = back[back.cutoff.isin(TEST_CUTOFFS)].merge(
        online[["cust", "cutoff", "label"]], on=["cust", "cutoff"])
    te_on = online[online.cutoff.isin(TEST_CUTOFFS)]
    clf = HistGradientBoostingClassifier(random_state=seed).fit(tr[FEATURES], tr.label)
    off = roc_auc_score(te_bf.label, clf.predict_proba(te_bf[FEATURES])[:, 1])
    prod = roc_auc_score(te_on.label, clf.predict_proba(te_on[FEATURES])[:, 1])
    return round(off, 4), round(prod, 4)


CELLS = [  # (name, key_mode, shift_days, cutoff_mode)
    ("control", "none", 0, "shifted"),
    ("stable_key", "stable", 0, "shifted"),
    ("rotate_full", "rotate_full", 0, "shifted"),
    ("rotate_incremental", "rotate_incremental", 0, "shifted"),
    ("shift_-98", "none", -98, "shifted"),
    ("shift_-100", "none", -100, "shifted"),
    ("shift_-100_unshifted_cutoff", "none", -100, "unshifted"),
    ("HEADLINE_rotate_incr+shift_-100", "rotate_incremental", -100, "shifted"),
    ("HEADLINE_rotate_incr+shift_-100_unshifted_cutoff",
     "rotate_incremental", -100, "unshifted"),
]

if __name__ == "__main__":
    inv = load_invoices()
    online = online_table(inv)
    print(f"invoices={len(inv)} customers={inv.cust.nunique()} "
          f"rows={len(online)} base_rate={online.label.mean():.3f}")
    clf = HistGradientBoostingClassifier(random_state=0).fit(
        online[online.cutoff.isin(TRAIN_CUTOFFS)][FEATURES],
        online[online.cutoff.isin(TRAIN_CUTOFFS)].label)
    te = online[online.cutoff.isin(TEST_CUTOFFS)]
    base_auc = round(roc_auc_score(te.label, clf.predict_proba(te[FEATURES])[:, 1]), 4)
    print("baseline AUC (online train, online test):", base_auc)
    results = {"baseline_auc": base_auc, "cells": {}}
    for name, km, sd, cm in CELLS:
        back = backfill_table(inv, online, km, sd, cm)
        div, _ = divergence(online, back)
        off, prod = auc_gap(online, back)
        results["cells"][name] = {
            "key_mode": km, "shift_days": sd, "cutoff_mode": cm,
            "leak_rate": round(float(back.leak.mean()), 4),
            "offline_auc": off, "production_auc": prod,
            "offline_minus_production": round(off - prod, 4),
            "features": div}
        print(f"\n== {name}: leak={back.leak.mean():.3f} "
              f"offlineAUC={off} prodAUC={prod}")
        for c, d in div.items():
            print(f"   {c:14s} match={d['match_rate']:.3f} psi={d['psi']} ks={d['ks']}")
    json.dump(results, open(out("results.json"), "w"), indent=1, default=str)
