#!/usr/bin/env bash
# Reproduces every table and figure in the article (about 15 minutes on a 2-core laptop).
set -e
cd "$(dirname "$0")"
python src/run_experiments.py   # baseline experiments  -> results/*.csv
python src/run_extensions.py    # extensions (needs the baseline tuning CSVs) -> results/ext_*.csv
python src/make_figures.py      # figures -> figures/*.png
echo "Done. See results/ and figures/"
