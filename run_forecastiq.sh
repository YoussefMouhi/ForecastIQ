#!/bin/bash
set -e

cd "$(dirname "$0")"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONFAULTHANDLER=1

exec .venv/bin/python -m streamlit run app.py --server.fileWatcherType none --server.fileWatcherType none
