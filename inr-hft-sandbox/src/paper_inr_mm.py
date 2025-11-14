import asyncio, json, time, yaml, hmac, hashlib
from datetime import datetime
import aiohttp
import websockets
from src.strategy import micro_price
from src.risk import RiskGuard
from src.logger import get_logger

log = get_logger('mm')

with open('config.yaml') as f: cfg = yaml.safe_load(f)
rg = RiskGuard(cfg['risk'], cfg['strategy'])

# Trading state
inventory_btc = 0.0
cash = cfg['risk']['initial_cash']
active_orders = {'buy': None, 'sell': None}
last_update = time.time()
trade_count = 0
update_count = 0

# Binance API setup
USE_TESTNET = cfg['exchange'] == 'binance_test'
BASE_URL = 'https://testnet.binance.vision' if USE_TESTNET else 'https://api.binance.com'
# Always use production WebSocket for market data (more reliable)
WS_URL = 'wss://stream.binance.com:9443/ws'
API_KEY = cfg['credentials']['api_key']
API_SECRET = cfg['credentials']['secret']
SYMBOL = cfg['symbol'].replace('-', '').lower()

def sign_request(params):
    """Sign Binance API request"""
    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    params['signature'] = signature
    return params

async def place_order(session, side, price, quantity):
    """Place limit order on Binance"""
    try:
        params = {
            'symbol': cfg['symbol'].replace('-', ''),
            'side': side.upper(),
            'type': 'LIMIT',
            'timeInForce': 'GTC',
            'quantity': f"{quantity:.8f}",
            'price': f"{price:.2f}",
            'timestamp': int(time.time() * 1000)
        }
        params = sign_request(params)
        
        headers = {'X-MBX-APIKEY': API_KEY}
        async with session.post(f'{BASE_URL}/api/v3/order', params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                result = await resp.json()
                return result.get('orderId')
            else:
                error = await resp.text()
                if 'code' in error:
                    log.warning(f"❌ Order {side} failed: {error[:100]}")
                return None
    except asyncio.TimeoutError:
        log.warning(f"⏱️ Order {side} timeout")
        return None
    except Exception as e:
        log.error(f"❌ Order {side} error: {str(e)[:100]}")
        return None

async def cancel_order(session, order_id):
    """Cancel order on Binance"""
    try:
        params = {
            'symbol': cfg['symbol'].replace('-', ''),
            'orderId': order_id,
            'timestamp': int(time.time() * 1000)
        }
        params = sign_request(params)
        
        headers = {'X-MBX-APIKEY': API_KEY}
        async with session.delete(f'{BASE_URL}/api/v3/order', params=params, headers=headers) as resp:
            return resp.status == 200
    except Exception as e:
        log.error(f"Cancel error: {e}")
        return False

async def handle_orderbook(data, session):
    """Process order book data and place orders"""
    global inventory_btc, cash, active_orders, last_update, trade_count, update_count
    
    update_count += 1
    
    # Throttle to refresh_ms
    now = time.time()
    if (now - last_update) * 1000 < cfg['strategy']['refresh_ms']:
        return
    last_update = now
    
    try:
        # Parse order book
        bids = [[float(p), float(q)] for p, q in data['bids'][:10]]
        asks = [[float(p), float(q)] for p, q in data['asks'][:10]]
        
        if not bids or not asks:
            return
        
        book = {'bid': bids, 'ask': asks}
        quote = micro_price(book, None)
        
        spread_bps = cfg['strategy']['spread_bps'] / 10_000
        my_bid = round(quote.bid * (1 - spread_bps / 2), 2)
        my_ask = round(quote.ask * (1 + spread_bps / 2), 2)
        
        # Display status
        pnl = rg.pnl_pct(quote.mid, cash)
        log.info(f"📊 Mid: ${quote.mid:.2f} | Bid: ${bids[0][0]:.2f} Ask: ${asks[0][0]:.2f} | Inv: {inventory_btc:.5f} | P&L: {pnl:+.2f}% | Updates: {update_count}")
        
        # Cancel old orders
        if active_orders['buy']:
            await cancel_order(session, active_orders['buy'])
            active_orders['buy'] = None
            
        if active_orders['sell']:
            await cancel_order(session, active_orders['sell'])
            active_orders['sell'] = None
        
        # Place new buy order
        if rg.can_quote(inventory_btc, 'buy'):
            order_id = await place_order(session, 'buy', my_bid, cfg['strategy']['quote_size_btc'])
            if order_id:
                active_orders['buy'] = order_id
                trade_count += 1
                log.info(f"✅ PLACED BUY  {cfg['strategy']['quote_size_btc']:.5f} BTC @ ${my_bid} | Order #{order_id}")
        
        # Place new sell order  
        if rg.can_quote(inventory_btc, 'sell') and inventory_btc > 0:
            order_id = await place_order(session, 'sell', my_ask, cfg['strategy']['quote_size_btc'])
            if order_id:
                active_orders['sell'] = order_id
                trade_count += 1
                log.info(f"✅ PLACED SELL {cfg['strategy']['quote_size_btc']:.5f} BTC @ ${my_ask} | Order #{order_id}")
                
    except Exception as e:
        log.error(f"Error handling orderbook: {e}")

async def book_update(book, receipt):
    global inventory_btc, cash, active_orders, last_update, trade_count
    
    log.info(f"📥 Received order book update - Bid: {book['bid'][0][0]}, Ask: {book['ask'][0][0]}")
    
    # Throttle updates to refresh_ms
    now = time.time()
    if (now - last_update) * 1000 < cfg['strategy']['refresh_ms']:
        return
    last_update = now
    
    lat_us = (time.time_ns() - receipt.timestamp_ns) / 1_000
    quote = micro_price(book, None)
    spread_bps = cfg['strategy']['spread_bps'] / 10_000
    my_bid = round(quote.bid * (1 - spread_bps / 2), 2)
    my_ask = round(quote.ask * (1 + spread_bps / 2), 2)
    
    # Display current state
    pnl = rg.pnl_pct(quote.mid, cash)
    log.info(f"📊 Mid: ${quote.mid:.2f} | Inv: {inventory_btc:.5f} BTC | Cash: ${cash:.2f} | P&L: {pnl:.2f}% | Trades: {trade_count}")
    
    async with aiohttp.ClientSession() as session:
        # Cancel and replace orders if needed
        if active_orders['buy']:
            await cancel_order(session, active_orders['buy'])
            active_orders['buy'] = None
            
        if active_orders['sell']:
            await cancel_order(session, active_orders['sell'])
            active_orders['sell'] = None
        
        # Place new buy order
        if rg.can_quote(inventory_btc, 'buy'):
            order_id = await place_order(session, 'buy', my_bid, cfg['strategy']['quote_size_btc'])
            if order_id:
                active_orders['buy'] = order_id
                log.info(f"✅ BUY  {cfg['strategy']['quote_size_btc']} @ ${my_bid} | Order #{order_id} | lat {lat_us:.0f}µs")
                trade_count += 1
        
        # Place new sell order
        if rg.can_quote(inventory_btc, 'sell') and inventory_btc > 0:
            order_id = await place_order(session, 'sell', my_ask, cfg['strategy']['quote_size_btc'])
            if order_id:
                active_orders['sell'] = order_id
                log.info(f"✅ SELL {cfg['strategy']['quote_size_btc']} @ ${my_ask} | Order #{order_id} | lat {lat_us:.0f}µs")
                trade_count += 1

async def run_bot():
    """Main bot loop with WebSocket"""
    log.info("🔌 Connecting to Binance WebSocket...")
    
    async with aiohttp.ClientSession() as session:
        ws_endpoint = f"{WS_URL}/{SYMBOL}@depth10@100ms"
        log.info(f"📡 Subscribing to: {ws_endpoint}")
        
        try:
            async with websockets.connect(ws_endpoint, ping_interval=20, ping_timeout=10) as websocket:
                log.info("✅ WebSocket connected! Starting automated trading...")
                
                async for message in websocket:
                    data = json.loads(message)
                    await handle_orderbook(data, session)
                    
        except websockets.exceptions.ConnectionClosed:
            log.error("❌ WebSocket connection closed. Reconnecting...")
            await asyncio.sleep(5)
            await run_bot()  # Reconnect
        except Exception as e:
            log.error(f"❌ WebSocket error: {e}")
            await asyncio.sleep(5)
            await run_bot()  # Reconnect

def main():
    log.info("=" * 70)
    log.info(f"🚀 AUTOMATED HFT BOT - {cfg['symbol']} on {cfg['exchange'].upper()}")
    log.info(f"💰 Initial capital: {cfg['risk']['initial_cash']} {cfg['quote']}")
    log.info(f"📈 Order size: {cfg['strategy']['quote_size_btc']} BTC")
    log.info(f"📊 Spread: {cfg['strategy']['spread_bps']} bps | Refresh: {cfg['strategy']['refresh_ms']}ms")
    log.info(f"⚠️  {'TESTNET MODE - Safe to test!' if USE_TESTNET else '🔴 LIVE TRADING - REAL MONEY!'}")
    log.info("=" * 70)
    
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        log.info("\n🛑 Shutting down gracefully...")
        log.info(f"📊 Final P&L: {rg.pnl_pct(0, cash):.2f}% | Total orders placed: {trade_count}")

if __name__ == '__main__':
    main()
