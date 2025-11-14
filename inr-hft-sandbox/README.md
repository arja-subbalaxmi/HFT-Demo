# INR-HFT-Sandbox
Zero-cost, micro-second HFT simulator for Binance Test-Net BTC-INR with ₹1 000 play money.

## Quick start
1. `python3 -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt`
3. Add **Binance Test-Net** API key in `config.yaml`
4. `bash scripts/run.sh`

## Metrics shown
- Tick-to-place latency (µs)  
- Fake fills & running P&L  
- Inventory guard-rails (max 0.0005 BTC)

## Scale to live
When 30-day paper Sharpe > 1.5 and latency < 500 µs, flip `sandbox: false` and deposit **₹10 000** real INR.
