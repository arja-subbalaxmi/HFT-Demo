#!/usr/bin/env python3
"""
Multi-Coin HFT Bot - Trades ALL profitable coins on Binance automatically
Scans for volatile coins and makes profits from price movements
"""

import asyncio
import json
import time
import aiohttp
from collections import defaultdict
import yaml
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# Global state
coin_data = {}
total_cash = 1000.0
total_trades = 0
initial_capital = 1000.0  # Track starting capital for real P&L
trade_cooldowns = {}  # Prevent repeated trades on same coin
active_coins = []

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

cfg = load_config()

async def get_top_coins():
    """Get top volatile USDT pairs INCLUDING memecoins from Binance"""
    url = "https://api.binance.com/api/v3/ticker/24hr"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            tickers = await resp.json()
    
    # PRIORITY: Memecoins (high volatility = profit potential)
    memecoins = ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'DOGS', 'NEIRO', 'MEME', 'TURBO']
    
    # Filter USDT pairs: Lower thresholds for more opportunities
    usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT') and 
                  float(t['priceChangePercent']) != 0 and
                  float(t['quoteVolume']) > 500000 and  # $500K+ volume (was 800K)
                  float(t['count']) > 3000]  # 3000+ trades (was 5000)
    
    # Prioritize memecoins, then sort by volatility
    memecoin_pairs = [t for t in usdt_pairs if any(t['symbol'].startswith(m) for m in memecoins)]
    other_pairs = [t for t in usdt_pairs if not any(t['symbol'].startswith(m) for m in memecoins)]
    
    memecoin_pairs.sort(key=lambda x: abs(float(x['priceChangePercent'])), reverse=True)
    other_pairs.sort(key=lambda x: abs(float(x['priceChangePercent'])), reverse=True)
    
    # Return top 10 memecoins + top 10 others
    selected = memecoin_pairs[:10] + other_pairs[:10]
    return [t['symbol'].replace('USDT', '') for t in selected]

async def subscribe_to_coin(coin):
    """Subscribe to WebSocket for a specific coin"""
    global total_cash, total_trades, coin_data
    
    symbol = f"{coin.lower()}usdt"
    url = f"wss://stream.binance.com:9443/ws/{symbol}@depth10@100ms"
    
    # Initialize coin state
    if coin not in coin_data:
        coin_data[coin] = {
            'inventory': 0.0,
            'trades': 0,
            'pnl_pct': 0.0,
            'last_price': 0.0,
            'avg_buy_price': 0.0
        }
    
    log.info(f"📡 Subscribing to {coin}/USDT")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        await handle_coin_update(coin, data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        break
    except Exception as e:
        log.error(f"❌ {coin} connection error: {e}")

async def handle_coin_update(coin, data):
    """SMART TRADING: No repeated trades, compound wins, cut losses fast"""
    global total_cash, total_trades, coin_data, trade_cooldowns, initial_capital
    
    try:
        # Skip if coin is on cooldown (prevent churning)
        import time
        current_time = time.time()
        if coin in trade_cooldowns and current_time < trade_cooldowns[coin]:
            return
        bids = [[float(p), float(q)] for p, q in data['bids'][:5]]
        asks = [[float(p), float(q)] for p, q in data['asks'][:5]]
        
        if not bids or not asks:
            return
        
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2
        spread_pct = ((best_ask - best_bid) / mid_price) * 100
        
        # Track price movement
        last_price = coin_data[coin]['last_price']
        coin_data[coin]['last_price'] = mid_price
        
        if last_price == 0:
            last_price = mid_price
        
        price_change_pct = ((mid_price - last_price) / last_price) * 100 if last_price > 0 else 0
        volatility = abs(price_change_pct)
        
        position = coin_data[coin]['inventory']
        position_value = position * mid_price
        
        # SMART position sizing: Adjust based on overall portfolio performance
        total_portfolio_value = total_cash
        for c in coin_data:
            total_portfolio_value += coin_data[c]['inventory'] * coin_data[c]['last_price']
        
        # Dynamic allocation based on performance
        if total_portfolio_value > initial_capital * 1.02:  # Winning
            base_allocation = 0.08  # Be aggressive
        elif total_portfolio_value < initial_capital * 0.98:  # Losing
            base_allocation = 0.05  # Be conservative
        else:
            base_allocation = 0.07  # Neutral
            
        max_position_value = total_cash * base_allocation
        
        # SELECTIVE ENTRY: Only trade on STRONG signals (avoid churning)
        buy_signal = (
            (price_change_pct < -0.15 and volatility > 0.1) or  # Significant dip only
            (spread_pct > 0.4 and volatility > 0.15) or  # Wide spread + volatility
            (volatility > 0.3)  # Extreme volatility
        )
        
        if buy_signal and position_value < max_position_value and total_cash > 15:
            # Conservative sizing to avoid overtrading
            trade_size = min(18, total_cash * 0.018)
            buy_qty = trade_size / mid_price
            
            if buy_qty * mid_price >= 12:
                coin_data[coin]['inventory'] += buy_qty
                coin_data[coin]['avg_buy_price'] = ((position * coin_data[coin].get('avg_buy_price', mid_price)) + (buy_qty * mid_price)) / (position + buy_qty)
                total_cash -= buy_qty * mid_price
                coin_data[coin]['trades'] += 1
                total_trades += 1
                
                # Set cooldown: 20 seconds before buying same coin again
                trade_cooldowns[coin] = current_time + 20
                
                log.info(f"🚀 {coin} BUY {buy_qty:.6f} @ ${mid_price:.6f} | Δ{price_change_pct:+.2f}% V:{volatility:.2f}% | Cash: ${total_cash:.2f}")
        
        # SMART EXITS: Take profits, cut losses
        elif position > 0:
            avg_buy = coin_data[coin].get('avg_buy_price', mid_price)
            profit_pct = ((mid_price - avg_buy) / avg_buy) * 100
            
            # Exit conditions
            sell_signal = (
                (profit_pct >= 0.25) or  # Quality profit taking
                (profit_pct < -0.8) or  # Hard stop loss
                (price_change_pct > 0.4 and profit_pct > 0.15)  # Big momentum spike
            )
            
            if sell_signal:
                # Exit strategy based on outcome
                if profit_pct > 0.25:
                    sell_pct = 0.80  # Sell 80% on good wins
                elif profit_pct < -0.5:
                    sell_pct = 1.0  # Exit 100% on losses
                else:
                    sell_pct = 0.85  # Default 85%
                    
                sell_qty = position * sell_pct
                
                if sell_qty * mid_price >= 12:
                    coin_data[coin]['inventory'] -= sell_qty
                    total_cash += sell_qty * mid_price
                    coin_data[coin]['trades'] += 1
                    total_trades += 1
                    coin_data[coin]['pnl_pct'] = profit_pct
                    
                    # Set cooldown: 30 seconds before selling again
                    trade_cooldowns[coin] = current_time + 30
                    
                    # Enhanced logging
                    if profit_pct >= 0.25:
                        reason = "🎯 PROFIT"
                        emoji = "💰"
                    elif profit_pct < -0.5:
                        reason = "🛑 STOPLOSS"
                        emoji = "⚠️"
                    else:
                        reason = "🚀 MOMENTUM"
                        emoji = "💰"
                    
                    # Calculate real portfolio value
                    portfolio_value = total_cash
                    for c in coin_data:
                        portfolio_value += coin_data[c]['inventory'] * coin_data[c]['last_price']
                    real_pnl = ((portfolio_value - initial_capital) / initial_capital) * 100
                    
                    log.info(f"{emoji} {coin} SELL {sell_qty:.6f} @ ${mid_price:.6f} | {reason} {profit_pct:+.2f}% | Portfolio: ${portfolio_value:.2f} ({real_pnl:+.2f}%)")
            last_price = mid_price
        
        price_change_pct = ((mid_price - last_price) / last_price) * 100 if last_price > 0 else 0
        
        position = coin_data[coin]['inventory']
        position_value = position * mid_price
        max_position_value = total_cash * 0.08  # 8% per coin
        
        # REALISTIC TRADING: Only trade on SIGNIFICANT price moves or opportunities
        
        # BUY SIGNAL: Price dropped >0.1% OR wide spread >0.5%
        if (price_change_pct < -0.1 or spread_pct > 0.5) and position_value < max_position_value and total_cash > 15:
            buy_qty = min(15 / mid_price, (max_position_value - position_value) / mid_price)
            if buy_qty * mid_price >= 8:
                coin_data[coin]['inventory'] += buy_qty
                coin_data[coin]['avg_buy_price'] = ((position * coin_data[coin].get('avg_buy_price', mid_price)) + (buy_qty * mid_price)) / (position + buy_qty)
                total_cash -= buy_qty * mid_price
                coin_data[coin]['trades'] += 1
                total_trades += 1
                log.info(f"📈 {coin} BUY {buy_qty:.6f} @ ${mid_price:.6f} | Change: {price_change_pct:+.2f}% | Cash: ${total_cash:.2f}")
        
        # SELL SIGNAL: Price rose >0.15% above buy price AND we have profit
        elif position > 0:
            avg_buy = coin_data[coin].get('avg_buy_price', mid_price)
            profit_pct = ((mid_price - avg_buy) / avg_buy) * 100
            
            # Sell if profitable OR price jumped >0.2%
            if profit_pct > 0.15 or price_change_pct > 0.2:
                sell_qty = position * 0.6  # Sell 60%
                if sell_qty * mid_price >= 8:
                    coin_data[coin]['inventory'] -= sell_qty
                    total_cash += sell_qty * mid_price
                    coin_data[coin]['trades'] += 1
                    total_trades += 1
                    
                    coin_data[coin]['pnl_pct'] = profit_pct
                    log.info(f"💵 {coin} SELL {sell_qty:.6f} @ ${mid_price:.6f} | Profit: {profit_pct:+.2f}% | Cash: ${total_cash:.2f}")
        
    except Exception as e:
        log.error(f"Error handling {coin}: {e}")

async def display_summary():
    """Display portfolio summary every 10 seconds with REAL P&L"""
    global total_cash, total_trades, coin_data, initial_capital
    
    while True:
        await asyncio.sleep(10)
        
        # Calculate total portfolio value (cash + all holdings)
        total_value = total_cash
        for coin, data in coin_data.items():
            total_value += data['inventory'] * data['last_price']
        
        # REAL P&L calculation
        pnl_dollars = total_value - initial_capital
        pnl_pct = (pnl_dollars / initial_capital) * 100
        active_coins = len([c for c in coin_data if coin_data[c]['trades'] > 0])
        
        log.info("="*70)
        status_emoji = "📈" if pnl_dollars > 0 else "📉" if pnl_dollars < 0 else "➡️"
        log.info(f"{status_emoji} PORTFOLIO: Cash: ${total_cash:.2f} | Holdings: ${total_value - total_cash:.2f} | Total: ${total_value:.2f}")
        log.info(f"💵 P&L: {pnl_pct:+.2f}% (${pnl_dollars:+.2f}) | Trades: {total_trades} | Active: {active_coins} coins")
        
        # Show top performers
        performers = [(coin, data['pnl_pct']) for coin, data in coin_data.items() if data['trades'] > 0]
        performers.sort(key=lambda x: x[1], reverse=True)
        
        if performers:
            log.info("🏆 Top Performers:")
            for coin, pnl in performers[:5]:
                trades = coin_data[coin]['trades']
                log.info(f"   {coin}: {pnl:+.2f}% ({trades} trades)")
        log.info("=" * 70)

async def main():
    log.info("=" * 70)
    log.info("🚀 MULTI-COIN HFT BOT - Auto-trading ALL profitable coins")
    log.info("💰 Starting capital: $1000 USDT")
    log.info("📈 Strategy: High-frequency market making across top 20 coins")
    log.info("=" * 70)
    
    # Get top coins
    log.info("🔍 Scanning Binance for top volatile coins...")
    global active_coins
    active_coins = await get_top_coins()
    
    log.info(f"✅ Found {len(active_coins)} profitable coins to trade:")
    log.info(f"   {', '.join(active_coins[:10])}...")
    log.info("=" * 70)
    
    # Start WebSocket for each coin + summary display
    tasks = [subscribe_to_coin(coin) for coin in active_coins]
    tasks.append(display_summary())
    
    await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        total_value = total_cash
        for coin, data in coin_data.items():
            total_value += data['inventory'] * data['last_price']
        
        pnl_pct = ((total_value - 1000) / 1000) * 100
        
        log.info("\n" + "=" * 70)
        log.info("🛑 Shutting down multi-coin bot...")
        log.info(f"💰 Final Capital: ${total_cash:.2f}")
        log.info(f"📊 Total Portfolio Value: ${total_value:.2f}")
        log.info(f"📈 Overall P&L: {pnl_pct:+.2f}% (${total_value - 1000:+.2f})")
        log.info(f"⚡ Total Trades: {total_trades}")
        log.info("=" * 70)
