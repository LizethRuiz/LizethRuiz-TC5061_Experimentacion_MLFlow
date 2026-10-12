#!/usr/bin/env bash
# Reproduces the 9 runs of experiment "TC5061-attrition-logreg-hiperparametros".
set -euo pipefail
cd "$(dirname "$0")"
python train.py                                                   # 1. baseline (defaults)
python train.py --run-name "C=1.0-l2-cw=none-thr=0.5-repeticion"  # 2. same args -> same result
python train.py --C 0.1                                           # 3. more regularization
python train.py --C 0.01                                          # 4. strong regularization
python train.py --class-weight balanced                           # 5. class weighting
python train.py --class-weight balanced --C 0.1                   # 6. weighting + regularization
python train.py --penalty l1 --C 0.1 --class-weight balanced      # 7. L1 (sparse coefficients)
python train.py --threshold 0.3                                   # 8. lower decision threshold
python train.py --class-weight balanced --C 0.1 --seed 7          # 9. same config as 6, other split
