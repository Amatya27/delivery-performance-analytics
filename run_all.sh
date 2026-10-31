#!/usr/bin/env bash
# runs the whole thing from scratch: download -> ETL -> model -> stats -> exports -> charts
set -e
cd "$(dirname "$0")"

python src/get_data.py
python src/etl.py
python src/model.py
python src/stats_tests.py
python src/export_tableau.py
python src/plot_trends.py
