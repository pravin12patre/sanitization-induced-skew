# Sanitization-Induced Skew

Code, data preparation, and results for the paper
**"Sanitization-Induced Skew: When Privacy Compliance Breaks Feature Reproducibility"**
by Pravin Barapatre (submitted to IEEE ICDE 2027).

The experiments show how deterministic identifier masking, key rotation, hash
truncation, cross-table masking mismatches, and global date shifting change
point-in-time ML features that are backfilled from sanitized history, and how
skew monitors (TensorFlow Data Validation and Evidently) behave under
sanitization. They also evaluate the paper's proposed *matched comparison*.

## Data

| Dataset | Source | License | File |
| --- | --- | --- | --- |
| Online Retail II | D. Chen, UCI Machine Learning Repository, [doi:10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D) | CC BY 4.0 | `data/raw.parquet` (converted from the `onlineretail2` R package copy) |
| CDNOW master | Fader & Hardie, distributed in the MIT-licensed [`lifetimes`](https://github.com/CamDavidsonPilon/lifetimes) package | see source | `data/CDNOW_master.txt` |

`data/` is included so results reproduce without network access.

## Reproduce

Python 3.13 with the pinned versions in `requirements.txt`:

```bash
pip install -r requirements.txt
DATASET=retail ./run_all.sh    # ~10 min on 2 CPU cores
DATASET=cdnow  ./run_all.sh
```

Results are written to `out_retail/` and `out_cdnow/` (the versions used in
the paper are committed). All randomness is seeded, so reruns reproduce every
result file exactly, except the wall-clock timings in `cost.json` and
`cost_runs.csv`.

| Script | Paper content |
| --- | --- |
| `skew.py` | Feature pipeline, sanitizers, per-feature match rates (Table: match rates) |
| `robustness.py` | Bootstrap CIs, shift-offset sweep, key-epoch sweep |
| `paired.py` | Paired production-AUC drops vs. unsanitized baseline |
| `forward.py` | Forward (positive) shift with unaligned cutoffs: staleness |
| `detect.py` | Skew-monitor detection experiment (plain, rules, residual, matched) |
| `detect2.py` | Raw-data differential testing vs. matched comparison, incl. sanitization-dependent bugs |
| `predict.py` | Manifest-only predictions of skewed features vs. observed match rates |
| `collide.py` | Hash truncation collisions vs. the birthday bound |
| `xtable.py` | Cross-table masking mismatch (retail only) |
| `cost.py` | Matched-comparison cost vs. sample fraction |
| `make_figures.py` | Paper figures from the result files |

## Notes

- The experimenter keeps true customer identifiers only to align rows for
  measurement; the simulated pipelines never use them.
- Labels always come from true history, isolating the effect on features.
- Bugs in the detection experiment are injected, not observed in production.

## License

Code: MIT. Data: see the licenses of the original sources above.
