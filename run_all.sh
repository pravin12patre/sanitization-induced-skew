#!/usr/bin/env bash
# Reproduce every experiment for one dataset: DATASET=retail|cdnow ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
python3 stats.py
python3 skew.py
python3 predict.py
python3 robustness.py
python3 paired.py
python3 forward.py
python3 detect.py
python3 detect2.py
python3 collide.py
[ "${DATASET:-retail}" = "retail" ] && python3 xtable.py
python3 cost.py
