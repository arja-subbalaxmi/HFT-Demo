#!/usr/bin/env bash
# replay 1-ms binance test-net dump (optional)
wget -q -O data/btcinr_1ms.csv https://github.com/coder-hft/binance-paper-inr/releases/download/v1.0/btcinr_2024_06_01.csv
python -m pytest tests/test_strategy.py -v
