#!/usr/bin/env bash
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python src/paper_inr_mm.py
