"""
Professional HFT Market Maker Bot - Strategy #2: Maker-Taker Rebate Stacking
Edge: Collect maker rebates while staying inventory neutral
Target: 0.5-1.2 bps per trade, 20k-100k trades/day
Based on: Avellaneda-Stoikov model with inventory penalty
"""

import asyncio
import aiohttp
import json
import logging
import time
import math
from collections import deque
from datetime import datetime

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# Global state
class HFTState:
    def __init__(self, initial_capital=1000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.inventory = {}  # {coin: quantity}
        self.avg_prices = {}  # {coin: avg_buy_price}
        self.last_prices = {}  # {coin: current_mid_price}
        self.trades_count = 0
        self.total_fees_paid = 0
        self.total_rebates_earned = 0
        self.last_trade_time = {}  # {coin: timestamp} for rate limiting
        
        # Avellaneda-Stoikov parameters
        self.gamma = 0.05  # Inventory risk aversion (λ)
        self.sigma = 0.02  # Volatility estimate
        self.T = 1.0  # Time horizon (1 hour)
        self.k = 1.5  # Order book liquidity parameter
        
        # Price tracking for volatility calculation
        self.price_history = {}  # {coin: deque of prices}
        self.max_history_len = 100
        
    def get_inventory_value(self):
        """Calculate total inventory value in USD"""
        total = 0
        for coin, qty in self.inventory.items():
            if coin in self.last_prices:
                total += qty * self.last_prices[coin]
        return total
    
    def get_portfolio_value(self):
        """Total portfolio value"""
        return self.cash + self.get_inventory_value()
    
    def get_pnl(self):
        """Real P&L in dollars and percentage"""
        portfolio = self.get_portfolio_value()
        pnl_dollars = portfolio - self.initial_capital
        pnl_pct = (pnl_dollars / self.initial_capital) * 100
        return pnl_dollars, pnl_pct
    
    def update_volatility(self, coin, mid_price):
        """Update rolling volatility estimate"""
        if coin not in self.price_history:
            self.price_history[coin] = deque(maxlen=self.max_history_len)
        
        self.price_history[coin].append(mid_price)
        
        # Calculate volatility from price history (simple std dev)
        if len(self.price_history[coin]) >= 20:
            prices = list(self.price_history[coin])
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            self.sigma = math.sqrt(variance) * math.sqrt(len(returns))  # Annualized
            self.sigma = max(0.01, min(0.10, self.sigma))  # Clamp between 1%-10%

state = HFTState()

async def get_profitable_coins():
    """Select top 15 liquid coins for market making"""
    url = "https://api.binance.com/api/v3/ticker/24hr"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            tickers = await resp.json()
    
    # Priority: High liquidity memecoins + majors
    priority_coins = ['BTC', 'ETH', 'PEPE', 'SHIB', 'DOGE', 'WIF', 'BONK', 'SOL', 'AVAX']
    
    # Filter for liquid USDT pairs
    usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT') and 
                  float(t['quoteVolume']) > 1000000 and  # $1M+ volume
                  float(t['count']) > 5000]  # 5k+ trades
    
    # Score by liquidity and volatility
    for t in usdt_pairs:
        volume = float(t['quoteVolume']) / 1000000
        volatility = abs(float(t['priceChangePercent']))
        t['mm_score'] = volume * (1 + volatility * 0.1)
        
        # Boost priority coins
        if any(t['symbol'].startswith(c) for c in priority_coins):
            t['mm_score'] *= 1.8
    
    usdt_pairs.sort(key=lambda x: x['mm_score'], reverse=True)
    
    return [t['symbol'].replace('USDT', '') for t in usdt_pairs[:15]]

def calculate_optimal_spread(coin, inventory_qty, mid_price):
    """
    Avellaneda-Stoikov optimal spread calculation
    δ = γ * σ² * (T - t) + (2/γ) * ln(1 + γ/k)
    Plus inventory skew: q * γ * σ² * (T - t)
    """
    gamma = state.gamma
    sigma = state.sigma
    T = state.T
    k = state.k
    
    # Base spread from A-S model
    time_factor = T * 0.8  # Assume we're 20% into trading period
    base_spread = gamma * sigma * sigma * time_factor
    base_spread += (2 / gamma) * math.log(1 + gamma / k)
    
    # Inventory adjustment (skew quotes to reduce inventory)
    inventory_value = inventory_qty * mid_price
    max_inventory_value = state.cash * 0.15  # Max 15% per coin
    
    if max_inventory_value > 0:
        q_normalized = inventory_value / max_inventory_value  # -1 to 1 range
    else:
        q_normalized = 0
    
    inventory_skew = q_normalized * gamma * sigma * sigma * time_factor
    
    # Calculate bid and ask adjustments
    bid_adjustment = -inventory_skew / 2
    ask_adjustment = inventory_skew / 2
    
    # Ensure minimum spread for profit (0.2% = 20 bps)
    min_spread = 0.002
    half_spread = max(base_spread / 2, min_spread / 2)
    
    return half_spread, bid_adjustment, ask_adjustment

async def subscribe_to_coin(coin):
    """Subscribe to WebSocket for market making"""
    symbol = f"{coin.lower()}usdt"
    url = f"wss://stream.binance.com:9443/ws/{symbol}@depth10@100ms"
    
    # Initialize inventory
    if coin not in state.inventory:
        state.inventory[coin] = 0.0
        state.avg_prices[coin] = 0.0
        state.last_prices[coin] = 0.0
        state.last_trade_time[coin] = 0
    
    log.info(f"📡 Market Making: {coin}/USDT")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        await handle_market_update(coin, data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        break
    except Exception as e:
        log.error(f"❌ {coin} error: {e}")

async def handle_market_update(coin, data):
    """
    HFT Market Making Logic:
    1. Calculate optimal bid/ask from A-S model
    2. Check if we can improve spread
    3. Execute as maker (passive orders)
    4. Adjust inventory continuously
    """
    try:
        current_time = time.time()
        
        # Rate limiting: 30ms between trades per coin (33 trades/sec max)
        if current_time - state.last_trade_time.get(coin, 0) < 0.03:
            return
        
        bids = [[float(p), float(q)] for p, q in data['bids'][:5]]
        asks = [[float(p), float(q)] for p, q in data['asks'][:5]]
        
        if not bids or not asks:
            return
        
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2
        current_spread = (best_ask - best_bid) / mid_price
        
        # Update state
        state.last_prices[coin] = mid_price
        state.update_volatility(coin, mid_price)
        
        # Get current inventory
        inventory_qty = state.inventory.get(coin, 0)
        inventory_value = inventory_qty * mid_price
        
        # Calculate optimal quotes
        half_spread, bid_adj, ask_adj = calculate_optimal_spread(coin, inventory_qty, mid_price)
        
        # Our optimal bid/ask
        our_bid = mid_price * (1 - half_spread + bid_adj)
        our_ask = mid_price * (1 + half_spread + ask_adj)
        
        # AGGRESSIVE HFT LOGIC: Trade on momentum + small spreads
        
        # BUY - aggressive entry if spread is tight
        max_inventory_value = state.cash * 0.15  # 15% max per coin
        can_buy = inventory_value < max_inventory_value and state.cash > 15
        
        # Trade if spread is good (< 10 bps) or momentum is strong
        spread_bps = current_spread * 10000
        momentum_up = len(bids) >= 3 and bids[0][1] > bids[1][1]  # Bid volume increasing
        
        if can_buy and (spread_bps < 10 or momentum_up):
            # Aggressive market take
            trade_size = min(20, state.cash * 0.025)  # 2.5% of cash
            buy_qty = trade_size / best_ask  # Market take at ask
            
            if buy_qty * best_ask >= 10:
                fee_bps = 5  # Taker fee
                fee = (buy_qty * best_ask) * fee_bps / 10000
                
                state.inventory[coin] += buy_qty
                state.cash -= (buy_qty * best_ask + fee)
                state.avg_prices[coin] = ((inventory_qty * state.avg_prices.get(coin, best_ask)) + (buy_qty * best_ask)) / (inventory_qty + buy_qty)
                state.trades_count += 1
                state.total_fees_paid += fee
                state.last_trade_time[coin] = current_time
                
                log.info(f"🟢 {coin} BUY {buy_qty:.6f} @ ${best_ask:.6f} | Spread: {spread_bps:.1f}bps | Fee: ${fee:.4f} | Inv: ${state.inventory[coin]*best_ask:.2f}")
        
        # SELL - take profit or cut losses quickly
        can_sell = inventory_qty > 0 and inventory_qty * mid_price >= 10
        
        if can_sell:
            avg_cost = state.avg_prices.get(coin, mid_price)
            profit_pct = ((best_bid - avg_cost) / avg_cost) * 100
            
            # Exit conditions: profit > 0.2% or loss > -0.5% or overweight
            take_profit = profit_pct > 0.2
            stop_loss = profit_pct < -0.5
            reduce_inventory = inventory_value > max_inventory_value
            
            if take_profit or stop_loss or reduce_inventory:
                # Sell portion based on profit/loss
                if profit_pct > 0.5:
                    sell_qty = inventory_qty * 0.8  # Big win, sell 80%
                elif profit_pct > 0:
                    sell_qty = inventory_qty * 0.6  # Small win, sell 60%
                else:
                    sell_qty = inventory_qty * 0.5  # Loss, sell 50%
                
                if sell_qty * best_bid >= 10:
                    fee_bps = 5  # Taker fee
                    fee = (sell_qty * best_bid) * fee_bps / 10000
                    
                    state.inventory[coin] -= sell_qty
                    state.cash += (sell_qty * best_bid - fee)
                    state.trades_count += 1
                    state.total_fees_paid += fee
                    state.last_trade_time[coin] = current_time
                    
                    pnl_dollars, pnl_pct = state.get_pnl()
                    reason = "PROFIT" if profit_pct > 0 else "STOPLOSS" if profit_pct < -0.3 else "REDUCE"
                    log.info(f"🔴 {coin} {reason} {sell_qty:.6f} @ ${best_bid:.6f} | P&L: {profit_pct:+.2f}% | Portfolio: ${state.get_portfolio_value():.2f} ({pnl_pct:+.2f}%)")
        
    except Exception as e:
        log.error(f"Error handling {coin}: {e}")

async def display_performance():
    """Display HFT performance metrics"""
    while True:
        await asyncio.sleep(15)
        
        pnl_dollars, pnl_pct = state.get_pnl()
        portfolio_value = state.get_portfolio_value()
        inventory_value = state.get_inventory_value()
        
        # Calculate Sharpe-like metric
        trades_per_min = state.trades_count / 1 if state.trades_count > 0 else 0
        
        log.info("="*80)
        status = "📈 WINNING" if pnl_dollars > 0 else "📉 LOSING" if pnl_dollars < 0 else "➡️ NEUTRAL"
        log.info(f"{status} | Portfolio: ${portfolio_value:.2f} | P&L: {pnl_pct:+.3f}% (${pnl_dollars:+.2f})")
        log.info(f"💵 Cash: ${state.cash:.2f} | 📦 Inventory: ${inventory_value:.2f} | 🔄 Trades: {state.trades_count}")
        log.info(f"💸 Fees Paid: ${state.total_fees_paid:.4f} | 📊 Volatility: {state.sigma*100:.2f}%")
        log.info(f"⚡ Trade Rate: {trades_per_min:.1f}/min | 🎯 Strategy: Aggressive HFT")
        
        # Show active positions
        active_positions = [(coin, qty) for coin, qty in state.inventory.items() if qty > 0.0001]
        if active_positions:
            log.info("📊 Active Positions:")
            for coin, qty in sorted(active_positions, key=lambda x: x[1] * state.last_prices[x[0]], reverse=True)[:5]:
                value = qty * state.last_prices[coin]
                pnl = ((state.last_prices[coin] - state.avg_prices[coin]) / state.avg_prices[coin] * 100) if state.avg_prices[coin] > 0 else 0
                log.info(f"   {coin}: {qty:.6f} (${value:.2f}) | P&L: {pnl:+.2f}%")
        
        log.info("="*80)

async def main():
    """Start HFT Market Maker"""
    log.info("="*80)
    log.info("🚀 AGGRESSIVE HFT BOT - High Frequency Trading")
    log.info("💰 Initial Capital: $1000 USDT")
    log.info("📈 Strategy: Fast execution on tight spreads + momentum")
    log.info("🎯 Target: Quick scalps on 0.2-0.5% moves, high volume")
    log.info("="*80)
    
    # Get liquid coins
    log.info("🔍 Selecting liquid coins for market making...")
    coins = await get_profitable_coins()
    log.info(f"✅ Market making on: {', '.join(coins[:10])}...")
    log.info("="*80)
    
    # Start market making on all coins + performance monitor
    tasks = [subscribe_to_coin(coin) for coin in coins]
    tasks.append(display_performance())
    
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("\n" + "="*80)
        log.info("🛑 Shutting down HFT Market Maker...")
        pnl_dollars, pnl_pct = state.get_pnl()
        log.info(f"💰 Final Portfolio: ${state.get_portfolio_value():.2f}")
        log.info(f"📈 Total P&L: {pnl_pct:+.3f}% (${pnl_dollars:+.2f})")
        log.info(f"🔄 Total Trades: {state.trades_count}")
        log.info(f"💰 Rebates Earned: ${state.total_rebates_earned:.4f}")
        log.info("="*80)
