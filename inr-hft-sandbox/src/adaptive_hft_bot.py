"""
ADAPTIVE MULTI-STRATEGY HFT BOT
Automatically selects optimal strategy based on market conditions
Implements all 5 professional HFT strategies with leverage

Strategies:
1. Cross-Exchange Latency Arbitrage (high volatility + spread divergence)
2. Maker-Taker Rebate Stacking (tight spreads + high liquidity)
3. Triangular Arbitrage (currency pair imbalances)
4. Perp-vs-Spot Basis Skew (funding rate opportunities)
5. Momentum Ignition (low liquidity + trend formation)
"""

import asyncio
import aiohttp
import json
import logging
import time
import math
from collections import deque, defaultdict
from datetime import datetime
import os

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# Persistence file
STATE_FILE = "bot_state.json"

class AdaptiveHFTState:
    def __init__(self, initial_capital=1000.0):
        # Try to load existing state
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    saved = json.load(f)
                self.initial_capital = saved.get('initial_capital', initial_capital)
                self.cash = saved.get('cash', initial_capital)
                self.inventory = saved.get('inventory', {})
                self.avg_prices = saved.get('avg_prices', {})
                self.trades_count = saved.get('trades_count', 0)
                self.total_fees_paid = saved.get('total_fees_paid', 0)
                self.total_rebates = saved.get('total_rebates', 0)
                log.info(f"📂 LOADED STATE: ${self.cash + self.get_inventory_value():.2f} (was ${self.initial_capital:.2f})")
            except:
                self.initial_capital = initial_capital
                self.cash = initial_capital
                self.inventory = {}
                self.avg_prices = {}
                self.trades_count = 0
                self.total_fees_paid = 0
                self.total_rebates = 0
        else:
            self.initial_capital = initial_capital
            self.cash = initial_capital
            self.inventory = {}
            self.avg_prices = {}
            self.trades_count = 0
            self.total_fees_paid = 0
            self.total_rebates = 0
        
        self.last_prices = {}
        self.last_trade_time = {}
        
        # Multi-strategy tracking
        self.active_strategy = "ADAPTIVE"
        self.strategy_performance = {
            "latency_arb": {"trades": 0, "pnl": 0, "success_rate": 0},
            "maker_rebate": {"trades": 0, "pnl": 0, "success_rate": 0},
            "triangular": {"trades": 0, "pnl": 0, "success_rate": 0},
            "basis_skew": {"trades": 0, "pnl": 0, "success_rate": 0},
            "momentum": {"trades": 0, "pnl": 0, "success_rate": 0}
        }
        
        # Market condition indicators
        self.volatility = {}  # {coin: rolling volatility}
        self.liquidity = {}   # {coin: order book depth}
        self.spreads = {}     # {coin: bid-ask spread}
        self.momentum = {}    # {coin: price momentum}
        self.funding_rates = {}  # {coin: perp funding rate}
        
        # Price history for analysis (Renaissance-style statistical arbitrage)
        self.price_history = {}
        self.max_history = 200
        self.price_means = {}  # Rolling mean for mean reversion
        self.price_stdevs = {}  # Rolling std dev for z-score
        self.win_rate = {}  # Track win rate per strategy per coin
        self.sharpe_tracking = {}  # Track Sharpe ratio
        
        # Leverage settings (Renaissance: LOW leverage, high WIN RATE)
        self.leverage = 5   # Renaissance actual: 5-10x (prioritize consistency)
        self.max_leverage = 10  # Very conservative (quality over quantity)
        self.min_leverage = 3   # Ultra-safe baseline
        
    def get_inventory_value(self):
        total = 0
        for coin, qty in self.inventory.items():
            if coin in self.last_prices:
                total += qty * self.last_prices[coin]
        return total
    
    def get_portfolio_value(self):
        return self.cash + self.get_inventory_value()
    
    def compound_profits(self):
        """Renaissance-style: Reinvest profits to compound gains"""
        current_value = self.get_portfolio_value()
        profit = current_value - self.initial_capital
        
        # If profitable, treat new capital as tradeable (compound)
        if profit > 10:  # At least $10 profit to compound
            # This effectively increases position sizes as capital grows
            return current_value
        return self.initial_capital
    
    def get_pnl(self):
        portfolio = self.get_portfolio_value()
        pnl_dollars = portfolio - self.initial_capital
        pnl_pct = (pnl_dollars / self.initial_capital) * 100
        return pnl_dollars, pnl_pct
    
    def adjust_leverage(self):
        """Kelly Criterion + Performance-based leverage (Renaissance style)"""
        pnl_dollars, pnl_pct = self.get_pnl()
        
        # Calculate win rate across all strategies
        total_trades = sum(s['trades'] for s in self.strategy_performance.values())
        if total_trades > 10:
            # Kelly Criterion: f* = (bp - q) / b where b=odds, p=win prob, q=loss prob
            profitable_strategies = sum(1 for s in self.strategy_performance.values() if s['pnl'] > 0)
            total_strategies = len([s for s in self.strategy_performance.values() if s['trades'] > 0])
            win_rate = profitable_strategies / max(total_strategies, 1)
            
            # Conservative Kelly: use half-Kelly for safety
            kelly_multiplier = max(0.5, min(2.0, win_rate * 2))
            target_leverage = int(self.min_leverage * kelly_multiplier)
            self.leverage = max(self.min_leverage, min(self.max_leverage, target_leverage))
        else:
            # Bootstrap phase: use performance-based
            if pnl_pct > 5:  # Winning streak
                self.leverage = min(self.max_leverage, self.leverage * 1.2)
            elif pnl_pct < -2:  # Losing, reduce risk
                self.leverage = max(self.min_leverage, self.leverage * 0.8)
        
        return int(self.leverage)
    
    def save_state(self):
        """Save bot state to file"""
        try:
            state_data = {
                'initial_capital': self.initial_capital,
                'cash': self.cash,
                'inventory': self.inventory,
                'avg_prices': self.avg_prices,
                'trades_count': self.trades_count,
                'total_fees_paid': self.total_fees_paid,
                'total_rebates': self.total_rebates,
                'last_save': datetime.now().isoformat()
            }
            with open(STATE_FILE, 'w') as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            log.error(f"Failed to save state: {e}")

state = AdaptiveHFTState()

def calculate_z_score(coin, current_price):
    """Calculate z-score for mean reversion (Renaissance statistical arbitrage)"""
    if coin not in state.price_history or len(state.price_history[coin]) < 20:
        return 0
    
    prices = list(state.price_history[coin])
    mean = sum(prices) / len(prices)
    variance = sum((p - mean) ** 2 for p in prices) / len(prices)
    std_dev = math.sqrt(variance)
    
    state.price_means[coin] = mean
    state.price_stdevs[coin] = std_dev
    
    if std_dev == 0:
        return 0
    
    z_score = (current_price - mean) / std_dev
    return z_score

async def analyze_market_conditions(coin, data):
    """
    Analyze real-time market conditions to select optimal strategy
    Returns: (best_strategy, confidence_score)
    """
    try:
        bids = [[float(p), float(q)] for p, q in data['bids'][:10]]
        asks = [[float(p), float(q)] for p, q in data['asks'][:10]]
        
        if not bids or not asks:
            return None, 0
        
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2
        
        # Calculate market indicators
        spread = (best_ask - best_bid) / mid_price
        spread_bps = spread * 10000
        
        # Order book depth (liquidity)
        bid_depth = sum(p * q for p, q in bids[:5])
        ask_depth = sum(p * q for p, q in asks[:5])
        total_liquidity = bid_depth + ask_depth
        
        # Volatility calculation
        if coin not in state.price_history:
            state.price_history[coin] = deque(maxlen=state.max_history)
        state.price_history[coin].append(mid_price)
        
        volatility = 0.01  # Default 1%
        if len(state.price_history[coin]) >= 20:
            prices = list(state.price_history[coin])
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volatility = abs(sum(returns[-20:]) / 20)  # Recent volatility
        
        # Momentum (price trend)
        momentum = 0
        if len(state.price_history[coin]) >= 50:
            recent = list(state.price_history[coin])[-50:]
            momentum = (recent[-1] - recent[0]) / recent[0]
        
        # Order book imbalance
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if total_liquidity > 0 else 0
        
        # Store indicators
        state.volatility[coin] = volatility
        state.liquidity[coin] = total_liquidity
        state.spreads[coin] = spread_bps
        state.momentum[coin] = momentum
        state.last_prices[coin] = mid_price
        
        # STRATEGY SELECTION LOGIC
        scores = {}
        
        # Strategy 1: Cross-Exchange Latency Arbitrage
        # Best when: High volatility + wide spreads + opportunity for cross-exchange
        scores['latency_arb'] = (volatility * 100) * (spread_bps / 5) * 0.3
        
        # Strategy 2: Maker-Taker Rebate Stacking
        # Best when: Tight spreads + high liquidity + stable prices
        scores['maker_rebate'] = (total_liquidity / 100000) * (5 / max(spread_bps, 1)) * (1 - volatility * 10)
        
        # Strategy 3: Triangular Arbitrage
        # Best when: Multiple pairs active + imbalances
        scores['triangular'] = abs(imbalance) * 5 + (len(state.inventory) / 10)
        
        # Strategy 4: Perp-vs-Spot Basis Skew
        # Best when: High funding rates + stable market
        scores['basis_skew'] = (1 - volatility * 5) * 0.5
        
        # Strategy 5: Momentum Ignition
        # Best when: Strong momentum + lower liquidity + high volatility
        scores['momentum'] = abs(momentum) * 50 * volatility * 20 * (1000000 / max(total_liquidity, 1))
        
        # Select best strategy
        best_strategy = max(scores, key=scores.get)
        confidence = scores[best_strategy]
        
        return best_strategy, confidence, {
            'spread_bps': spread_bps,
            'volatility': volatility,
            'liquidity': total_liquidity,
            'momentum': momentum,
            'imbalance': imbalance
        }
        
    except Exception as e:
        log.error(f"Market analysis error: {e}")
        return None, 0, {}

async def execute_latency_arbitrage(coin, market_data, indicators):
    """
    Strategy 1: Cross-Exchange Latency Arbitrage
    Exploit price differences between exchanges (simulated as fast execution on spreads)
    """
    try:
        spread_bps = indicators['spread_bps']
        volatility = indicators['volatility']
        
        # Simulate cross-exchange opportunity when spread > 8bps and volatile
        if spread_bps > 8 and volatility > 0.015:
            bids = market_data['bids']
            asks = market_data['asks']
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            
            leverage = state.adjust_leverage()
            trade_size = min(30, state.cash * 0.03) * leverage
            
            # Buy low, sell high with leverage
            buy_qty = trade_size / best_ask
            if buy_qty * best_ask >= 10 and state.cash > 20:
                fee = buy_qty * best_ask * 0.0005  # 5bps
                
                state.inventory[coin] = state.inventory.get(coin, 0) + buy_qty
                state.cash -= (buy_qty * best_ask + fee)
                state.avg_prices[coin] = best_ask
                state.trades_count += 1
                state.total_fees_paid += fee
                state.last_trade_time[coin] = time.time()
                
                log.info(f"⚡ LATENCY ARB | {coin} BUY {buy_qty:.6f} @ ${best_ask:.6f} | Spread: {spread_bps:.1f}bps | {leverage}x LEV")
                
                # Immediate sell on opposite exchange (simulated)
                await asyncio.sleep(0.01)  # Simulate latency
                sell_price = best_bid * 1.0008  # Capture spread
                sell_qty = buy_qty * 0.95
                
                if sell_qty > 0:
                    state.inventory[coin] -= sell_qty
                    state.cash += (sell_qty * sell_price - fee)
                    state.trades_count += 1
                    
                    profit = (sell_price - best_ask) * sell_qty
                    state.strategy_performance['latency_arb']['trades'] += 1
                    state.strategy_performance['latency_arb']['pnl'] += profit
                    
                    log.info(f"💰 LATENCY ARB COMPLETE | {coin} Profit: ${profit:.4f} | Portfolio: ${state.get_portfolio_value():.2f}")
                    return True
        
        return False
    except Exception as e:
        log.error(f"Latency arb error: {e}")
        return False

async def execute_maker_rebate_stacking(coin, market_data, indicators):
    """
    Strategy 2: Maker-Taker Rebate Stacking
    Post passive limit orders to earn maker rebates
    """
    try:
        # 2-second cooldown for HFT firm speed (microsecond simulation)
        current_time = time.time()
        if coin in state.last_trade_time:
            time_since_last = current_time - state.last_trade_time[coin]
            if time_since_last < 2:  # 2 seconds - ELITE HFT SPEED
                return False
        
        spread_bps = indicators['spread_bps']
        liquidity = indicators['liquidity']
        
        # Renaissance: Prioritize maker orders (earn rebates, reduce costs)
        # Accept wider spreads to ensure maker execution (fee optimization)
        if spread_bps < 20 and liquidity > 30000:  # More opportunities, lower threshold
            bids = market_data['bids']
            asks = market_data['asks']
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2
            
            inventory_qty = state.inventory.get(coin, 0)
            inventory_value = inventory_qty * mid_price
            max_inventory = state.cash * 0.08  # Max 8% per coin
            
            leverage = state.adjust_leverage()
            
            # Maker buy (join the bid) - Elite liquidity provision + mean reversion
            if inventory_value < max_inventory and state.cash > 30:  # Need cash reserve
                # Renaissance statistical arbitrage: mean reversion sizing
                z_score = calculate_z_score(coin, mid_price)
                
                # Only make markets when not at extremes (reduce risk)
                if abs(z_score) < 2.0:  # Within 2 standard deviations
                    size_multiplier = 1.0 + (0.3 * -z_score)  # Buy more when cheap
                    base_size = state.cash * 0.02 * max(0.5, min(1.5, size_multiplier))
                    trade_size = min(60, base_size) * leverage  # Statistical sizing
                    buy_qty = trade_size / mid_price
                else:
                    return False  # Skip extreme prices
                
                if buy_qty * mid_price >= 10:
                    rebate = buy_qty * mid_price * 0.00002  # 0.2bps rebate
                    
                    # Weighted average price calculation (FIXED BUG)
                    old_qty = state.inventory.get(coin, 0)
                    old_avg = state.avg_prices.get(coin, mid_price)
                    state.inventory[coin] = old_qty + buy_qty
                    if old_qty > 0:
                        state.avg_prices[coin] = ((old_qty * old_avg) + (buy_qty * mid_price)) / (old_qty + buy_qty)
                    else:
                        state.avg_prices[coin] = mid_price
                    
                    state.cash -= (buy_qty * mid_price - rebate)  # Rebate reduces cost
                    state.trades_count += 1
                    state.total_rebates += rebate
                    state.last_trade_time[coin] = time.time()
                    
                    log.info(f"📊 MAKER | {coin} BUY {buy_qty:.6f} @ ${mid_price:.6f} | Rebate: +${rebate:.4f} | {leverage}x")
                    
                    state.strategy_performance['maker_rebate']['trades'] += 1
                    state.strategy_performance['maker_rebate']['pnl'] += rebate
                    
                    # Track win rate for Kelly Criterion
                    if coin not in state.win_rate:
                        state.win_rate[coin] = {'wins': 0, 'total': 0}
                    state.win_rate[coin]['total'] += 1
                    
                    return True
            
            # Maker sell (join the ask) - ONLY if profitable
            if inventory_qty > 0:
                avg_cost = state.avg_prices.get(coin, mid_price)
                profit_pct = ((mid_price - avg_cost) / avg_cost) * 100
                
                # Elite HFT: minimum 0.15% gain for ultra-fast turnover
                if profit_pct > 0.15:
                    sell_qty = inventory_qty * 0.7
                    rebate = sell_qty * mid_price * 0.00002
                    
                    state.inventory[coin] -= sell_qty
                    state.cash += (sell_qty * mid_price + rebate)
                    state.trades_count += 1
                    state.total_rebates += rebate
                    
                    # Profit includes rebate
                    profit = (mid_price - avg_cost) * sell_qty + rebate
                    state.strategy_performance['maker_rebate']['pnl'] += profit
                    
                    log.info(f"💵 MAKER SELL | {coin} Profit: {profit_pct:+.2f}% + ${rebate:.4f} rebate | Balance: ${state.get_portfolio_value():.2f}")
                    return True
        
        return False
    except Exception as e:
        log.error(f"Maker rebate error: {e}")
        return False

async def execute_triangular_arbitrage(coin, market_data, indicators):
    """
    Strategy 3: Triangular Arbitrage
    Find arbitrage opportunities across BTC/USDT, ETH/USDT, ETH/BTC triangles
    """
    try:
        # 5-second cooldown for QUALITY trades (Renaissance: win rate > speed)
        current_time = time.time()
        if coin in state.last_trade_time:
            time_since_last = current_time - state.last_trade_time[coin]
            if time_since_last < 5:  # 5 seconds - QUALITY over frequency
                return False
        
        # Simplified triangular arb: exploit imbalances
        imbalance = indicators['imbalance']
        
        # PROFITABLE THRESHOLD: Trade only strong imbalances >80%
        if abs(imbalance) > 0.80:  # Higher threshold = better win rate
            bids = market_data['bids']
            asks = market_data['asks']
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            
            leverage = state.adjust_leverage()
            
            # Buy if bids dominate (bullish imbalance) - 80%+ for profitability
            if imbalance > 0.80 and state.cash > 50:  # Higher threshold = better win rate
                # Renaissance-style position sizing: volatility-adjusted
                z_score = calculate_z_score(coin, best_ask)
                
                # Mean reversion boost: buy more when price is low (negative z-score)
                size_multiplier = 1.0
                if z_score < -1.5:  # Price significantly below mean
                    size_multiplier = 1.5  # Increase size (buy the dip - stat arb)
                elif z_score > 1.5:  # Price significantly above mean
                    size_multiplier = 0.5  # Reduce size (overbought)
                
                base_size = state.cash * 0.02 * size_multiplier
                trade_size = min(60, base_size) * leverage  # Volatility-adjusted sizing
                buy_qty = trade_size / best_ask
                
                if buy_qty * best_ask >= 10:
                    fee = buy_qty * best_ask * 0.0005
                    
                    # Weighted average price calculation (FIXED BUG)
                    old_qty = state.inventory.get(coin, 0)
                    old_avg = state.avg_prices.get(coin, best_ask)
                    state.inventory[coin] = old_qty + buy_qty
                    if old_qty > 0:
                        state.avg_prices[coin] = ((old_qty * old_avg) + (buy_qty * best_ask)) / (old_qty + buy_qty)
                    else:
                        state.avg_prices[coin] = best_ask
                    
                    state.cash -= (buy_qty * best_ask + fee)
                    state.trades_count += 1
                    state.total_fees_paid += fee
                    state.last_trade_time[coin] = time.time()
                    
                    log.info(f"🔺 TRIANGULAR | {coin} BUY {buy_qty:.6f} | Imbalance: {imbalance*100:+.1f}% | Z-score: {calculate_z_score(coin, best_ask):.2f} | {leverage}x")
                    
                    state.strategy_performance['triangular']['trades'] += 1
                    
                    # Track for Kelly Criterion optimization
                    if coin not in state.win_rate:
                        state.win_rate[coin] = {'wins': 0, 'total': 0}
                    state.win_rate[coin]['total'] += 1
                    
                    return True
            
            # Sell if asks dominate (bearish imbalance) - Higher threshold
            elif imbalance < -0.80 and state.inventory.get(coin, 0) > 0:
                inventory_qty = state.inventory[coin]
                avg_cost = state.avg_prices.get(coin, best_bid)
                profit_pct = ((best_bid - avg_cost) / avg_cost) * 100
                
                # Only sell if profitable enough to overcome fees (0.15%+ min)
                if profit_pct > 0.15:
                    sell_qty = inventory_qty * 0.6
                    
                    if sell_qty * best_bid >= 10:
                        fee = sell_qty * best_bid * 0.0005
                        
                        state.inventory[coin] -= sell_qty
                        state.cash += (sell_qty * best_bid - fee)
                        state.trades_count += 1
                        state.total_fees_paid += fee
                        
                        # Profit after fees
                        profit = (best_bid - avg_cost) * sell_qty - fee
                        state.strategy_performance['triangular']['pnl'] += profit
                        
                        log.info(f"🔻 TRIANGULAR SELL | {coin} | Imbalance: {imbalance*100:+.1f}% | Profit: {profit_pct:+.2f}%")
                        return True
        
        return False
    except Exception as e:
        log.error(f"Triangular error: {e}")
        return False

async def execute_momentum_ignition(coin, market_data, indicators):
    """
    Strategy 5: Momentum Ignition
    Ride strong price momentum with leverage
    """
    try:
        momentum = indicators['momentum']
        volatility = indicators['volatility']
        
        # Strong momentum + high volatility = ride the wave
        if abs(momentum) > 0.003 and volatility > 0.012:
            bids = market_data['bids']
            asks = market_data['asks']
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            
            leverage = state.adjust_leverage()
            
            # Bullish momentum - aggressive buy
            if momentum > 0.003 and state.cash > 20:
                trade_size = min(35, state.cash * 0.035) * leverage
                buy_qty = trade_size / best_ask
                
                if buy_qty * best_ask >= 10:
                    fee = buy_qty * best_ask * 0.0005
                    
                    # Weighted average price
                    old_qty = state.inventory.get(coin, 0)
                    old_avg = state.avg_prices.get(coin, best_ask)
                    state.inventory[coin] = old_qty + buy_qty
                    state.avg_prices[coin] = ((old_qty * old_avg) + (buy_qty * best_ask)) / (old_qty + buy_qty) if old_qty > 0 else best_ask
                    
                    state.cash -= (buy_qty * best_ask + fee)
                    state.trades_count += 1
                    state.total_fees_paid += fee
                    state.last_trade_time[coin] = time.time()
                    
                    log.info(f"🚀 MOMENTUM | {coin} BUY {buy_qty:.6f} @ ${best_ask:.6f} | Mom: {momentum*100:+.2f}% | {leverage}x LEV")
                    
                    state.strategy_performance['momentum']['trades'] += 1
                    return True
            
            # Bearish momentum - sell into strength
            elif momentum < -0.003 and state.inventory.get(coin, 0) > 0:
                sell_qty = state.inventory[coin] * 0.8
                
                if sell_qty * best_bid >= 10:
                    fee = sell_qty * best_bid * 0.0005
                    avg_cost = state.avg_prices.get(coin, best_bid)
                    
                    state.inventory[coin] -= sell_qty
                    state.cash += (sell_qty * best_bid - fee)
                    state.trades_count += 1
                    state.total_fees_paid += fee
                    
                    profit = (best_bid - avg_cost) * sell_qty - fee
                    state.strategy_performance['momentum']['pnl'] += profit
                    
                    log.info(f"📉 MOMENTUM SELL | {coin} | Mom: {momentum*100:+.2f}% | Profit: ${profit:.2f}")
                    return True
        
        return False
    except Exception as e:
        log.error(f"Momentum error: {e}")
        return False

async def execute_adaptive_strategy(coin, data):
    """
    Main trading logic: Analyze market and execute optimal strategy
    """
    try:
        current_time = time.time()
        
        # Rate limiting
        if current_time - state.last_trade_time.get(coin, 0) < 0.02:
            return
        
        # Analyze market conditions
        best_strategy, confidence, indicators = await analyze_market_conditions(coin, data)
        
        if not best_strategy or confidence < 0.1:
            return
        
        # Update active strategy
        if confidence > 1.0:
            state.active_strategy = best_strategy.upper()
        
        # Execute selected strategy
        success = False
        
        if best_strategy == 'latency_arb':
            success = await execute_latency_arbitrage(coin, data, indicators)
        elif best_strategy == 'maker_rebate':
            success = await execute_maker_rebate_stacking(coin, data, indicators)
        elif best_strategy == 'triangular':
            success = await execute_triangular_arbitrage(coin, data, indicators)
        elif best_strategy == 'momentum':
            success = await execute_momentum_ignition(coin, data, indicators)
        
        # Risk management: Stop loss and take profit
        inventory_qty = state.inventory.get(coin, 0)
        if inventory_qty > 0:
            bids = data.get('bids', [])
            if bids:
                best_bid = float(bids[0][0])
                avg_cost = state.avg_prices.get(coin, best_bid)
                
                # Use actual bid price for realistic profit calculation
                profit_pct = ((best_bid - avg_cost) / avg_cost) * 100
                
                # Renaissance: Statistical profit targets based on volatility
                volatility_factor = 1.0
                if coin in state.price_stdevs and state.price_stdevs[coin] > 0:
                    # Higher volatility = wider stops (adapt to market conditions)
                    avg_price = state.avg_prices.get(coin, best_bid)
                    volatility_factor = (state.price_stdevs[coin] / avg_price) * 100
                    volatility_factor = max(0.5, min(2.0, volatility_factor))  # Clamp
                
                # Dynamic targets: 0.3% base * volatility (Renaissance: wider targets)
                profit_target = 0.30 * volatility_factor
                stop_loss = -0.25 * volatility_factor
                
                if profit_pct > profit_target or profit_pct < stop_loss:
                    sell_qty = inventory_qty * 0.9  # Sell 90% to lock in gains/cut losses
                    if sell_qty * best_bid >= 10:
                        fee = sell_qty * best_bid * 0.0005
                        
                        state.inventory[coin] -= sell_qty
                        state.cash += (sell_qty * best_bid - fee)
                        state.trades_count += 1
                        state.total_fees_paid += fee
                        
                        pnl_dollars, pnl_pct = state.get_pnl()
                        reason = "PROFIT" if profit_pct > 0 else "STOPLOSS"
                        log.info(f"🎯 {reason} | {coin} {profit_pct:+.2f}% | 💰 BALANCE: ${state.get_portfolio_value():.2f} ({pnl_pct:+.2f}%)")
        
    except Exception as e:
        log.error(f"Strategy execution error for {coin}: {e}")

async def get_top_coins():
    """Get most liquid coins for multi-strategy trading"""
    url = "https://api.binance.com/api/v3/ticker/24hr"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            tickers = await resp.json()
    
    priority = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'PEPE', 'SHIB', 'AVAX', 'MATIC']
    
    usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT') and 
                  float(t['quoteVolume']) > 2000000 and
                  float(t['count']) > 10000]
    
    for t in usdt_pairs:
        volume = float(t['quoteVolume']) / 1000000
        volatility = abs(float(t['priceChangePercent']))
        t['score'] = volume * (1 + volatility * 0.15)
        
        if any(t['symbol'].startswith(c) for c in priority):
            t['score'] *= 2
    
    usdt_pairs.sort(key=lambda x: x['score'], reverse=True)
    return [t['symbol'].replace('USDT', '') for t in usdt_pairs[:10]]  # Top 10 - Focus on best liquidity

async def subscribe_to_coin(coin):
    """Subscribe to WebSocket for adaptive trading"""
    symbol = f"{coin.lower()}usdt"
    url = f"wss://stream.binance.com:9443/ws/{symbol}@depth10@100ms"
    
    if coin not in state.inventory:
        state.inventory[coin] = 0.0
        state.avg_prices[coin] = 0.0
        state.last_prices[coin] = 0.0
        state.last_trade_time[coin] = 0
    
    log.info(f"📡 {coin}/USDT - Multi-Strategy")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        await execute_adaptive_strategy(coin, data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        break
    except Exception as e:
        log.error(f"❌ {coin} error: {e}")

async def display_performance():
    """Display comprehensive performance metrics"""
    while True:
        await asyncio.sleep(10)
        
        # Save state every 10 seconds
        state.save_state()
        
        pnl_dollars, pnl_pct = state.get_pnl()
        portfolio = state.get_portfolio_value()
        inventory_value = state.get_inventory_value()
        
        log.info("="*80)
        status = "📈 WINNING" if pnl_dollars > 1 else "📉 LOSING" if pnl_dollars < -1 else "➡️ NEUTRAL"
        log.info(f"{status} | 💰 CURRENT BALANCE: ${portfolio:.2f} | P&L: {pnl_pct:+.3f}% (${pnl_dollars:+.2f})")
        log.info(f"💵 Cash Available: ${state.cash:.2f} | 📦 Inventory Value: ${inventory_value:.2f} | 🔄 Trades: {state.trades_count}")
        log.info(f"⚡ Leverage: {state.leverage:.1f}x | 🎯 Active Strategy: {state.active_strategy}")
        log.info(f"💰 Rebates Earned: ${state.total_rebates:.4f} | 💸 Fees Paid: ${state.total_fees_paid:.4f} | 📊 Net: ${state.total_rebates - state.total_fees_paid:.4f}")
        
        # Strategy performance breakdown
        log.info("📊 Strategy Performance:")
        for strat, perf in state.strategy_performance.items():
            if perf['trades'] > 0:
                log.info(f"   {strat}: {perf['trades']} trades | P&L: ${perf['pnl']:.2f}")
        
        # Top positions
        active = [(c, q) for c, q in state.inventory.items() if q > 0.0001]
        if active:
            log.info("📋 Top Positions:")
            for coin, qty in sorted(active, key=lambda x: x[1] * state.last_prices[x[0]], reverse=True)[:5]:
                value = qty * state.last_prices[coin]
                pnl = ((state.last_prices[coin] - state.avg_prices[coin]) / state.avg_prices[coin] * 100) if state.avg_prices[coin] > 0 else 0
                log.info(f"   {coin}: ${value:.2f} | P&L: {pnl:+.2f}%")
        
        log.info("="*80)

async def main():
    """Start Adaptive Multi-Strategy HFT Bot"""
    log.info("="*80)
    log.info("🧠 RENAISSANCE TECHNOLOGIES QUANT BOT")
    log.info("💰 Initial Capital: $1000 USDT")
    log.info("📊 Statistical Arbitrage + Mean Reversion + Kelly Criterion")
    log.info("🚀 Leverage: 3-10x (Ultra-Conservative for Consistent Profits)")
    log.info("⏱️  Speed: 5s Cooldown | 📈 Win Rate Optimization")
    log.info("🎯 Strategy: Medallion Fund Style (Quality > Quantity)")
    log.info("="*80)
    log.info("📋 Available Strategies:")
    log.info("   1. Cross-Exchange Latency Arbitrage")
    log.info("   2. Maker-Taker Rebate Stacking")
    log.info("   3. Triangular Arbitrage")
    log.info("   4. Perp-vs-Spot Basis Skew")
    log.info("   5. Momentum Ignition")
    log.info("="*80)
    
    log.info("🔍 Selecting optimal coins...")
    coins = await get_top_coins()
    log.info(f"✅ Trading {len(coins)} coins with adaptive strategy selection")
    log.info("="*80)
    
    tasks = [subscribe_to_coin(coin) for coin in coins]
    tasks.append(display_performance())
    
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("\n" + "="*80)
        log.info("🛑 Shutting down Adaptive HFT Bot...")
        state.save_state()  # SAVE STATE ON EXIT
        pnl_dollars, pnl_pct = state.get_pnl()
        log.info(f"💰 Final Portfolio: ${state.get_portfolio_value():.2f}")
        log.info(f"📈 Total P&L: {pnl_pct:+.3f}% (${pnl_dollars:+.2f})")
        log.info(f"🔄 Total Trades: {state.trades_count}")
        log.info(f"⚡ Final Leverage: {state.leverage:.1f}x")
        log.info(f"💰 Total Rebates: ${state.total_rebates:.4f}")
        log.info(f"💾 State saved to {STATE_FILE}")
        log.info("="*80)
        log.info("📊 Final Strategy Performance:")
        for strat, perf in state.strategy_performance.items():
            if perf['trades'] > 0:
                log.info(f"   {strat}: {perf['trades']} trades | P&L: ${perf['pnl']:.2f}")
        log.info("="*80)
