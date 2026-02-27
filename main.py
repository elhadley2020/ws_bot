import os, asyncio, aiohttp, pandas as pd, numpy as np, json
from datetime import datetime
from dotenv import load_dotenv

# ================= CONFIG =================
load_dotenv()
API_KEY = os.getenv("OANDA_API_KEY")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
REST = "https://api-fxpractice.oanda.com/v3"
STREAM = f"https://stream-fxpractice.oanda.com/v3/accounts/{ACCOUNT_ID}/pricing/stream"

INSTR = ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD"]

# Indicators & risk
EMA_FAST, EMA_SLOW = 50, 200
RSI_PERIOD = 14
ATR_PERIOD = 14
RISK_PER_TRADE = 0.01
SL_MULTIPLIER = 1.5
TP_MULTIPLIER = 3
COOLDOWN = 4  # in candles
TIMEFRAME = "15min"  # main timeframe for trend
TIMEFRAME_LONG = "1H"  # Higher timeframe for trend confirmation

# ================= STATE =================
state = {
    "price": {p: pd.DataFrame(columns=["time", "open", "high", "low", "close"]) for p in INSTR},
    "price_1h": {p: pd.DataFrame(columns=["time", "open", "high", "low", "close"]) for p in INSTR},
    "last_signal": {p: 0 for p in INSTR},
    "pos": {p: 0 for p in INSTR},
    "entry": {p: 0.0 for p in INSTR},
    "last_trade_time": {p: 0 for p in INSTR},
    "locks": {p: asyncio.Lock() for p in INSTR}
}

# ================= INDICATORS =================
def add_indicators(df):
    # Fill NaN values with forward fill (or backward fill if preferred)
    df['high'] = df['high'].fillna(method='ffill')
    df['low'] = df['low'].fillna(method='ffill')
    df['prev_close'] = df['prev_close'].fillna(method='ffill')
    
    # EMA calculation
    df['ema50'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    
    # RSI calculation
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + avg_gain / avg_loss))

    # ATR calculation
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['prev_close']),
                                     abs(df['low'] - df['prev_close'])))
    df['atr'] = df['tr'].rolling(ATR_PERIOD).mean()

    # MACD calculation
    df['macd_line'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()

    return df

def is_trending(df):
    # Calculate the slope of EMA50 and EMA200
    ema50_slope = df['ema50'].diff().iloc[-1]  # Change in EMA50 over the last period
    ema200_slope = df['ema200'].diff().iloc[-1]  # Change in EMA200 over the last period
    
    # Check if both EMAs are flat (low slope), which typically indicates sideways movement
    if abs(ema50_slope) < 0.00001 and abs(ema200_slope) < 0.00001:
        return False  # Flat EMAs, avoid trade
    
    return True  # Trending market

def is_sideways(df, period=6):
    """Check if price has been moving in a narrow range for a set period"""
    price_range = df['high'].max() - df['low'].min()
    return price_range < 0.0005  # Adjust this threshold based on your asset's volatility

def get_signal(df_15min, df_1h):
    last_15min = df_15min.iloc[-1]
    last_1h = df_1h.iloc[-1]
    
    # Sideways check: Avoid trading if price has been range-bound for multiple candles
    if is_sideways(df_15min, period=6) or last_15min['atr'] < 0.0005 or not is_trending(df_15min):
        return 0  # No trade (sideways market or low ATR)
    
    # Long and short signals based on MACD, EMA, and RSI for trend confirmation
    if last_15min['macd_line'] > last_15min['macd_signal'] and last_15min['ema50'] > last_15min['ema200'] and last_15min['rsi'] > 50:
        if last_1h['macd_line'] > last_1h['macd_signal'] and last_1h['ema50'] > last_1h['ema200']:  # 1-hour confirmation
            return 1  # long
    
    elif last_15min['macd_line'] < last_15min['macd_signal'] and last_15min['ema50'] < last_15min['ema200'] and last_15min['rsi'] < 50:
        if last_1h['macd_line'] < last_1h['macd_signal'] and last_1h['ema50'] < last_1h['ema200']:  # 1-hour confirmation
            return -1  # short
    
    return 0  # no trade

# ================= RISK CALC =================
def calculate_units(nav, margin_avail, atr, risk=RISK_PER_TRADE):
    units = int((nav * risk) / (SL_MULTIPLIER * atr))
    units = min(units, int(margin_avail))
    return max(units, 0)

def fmt_price(price, pair):
    return f"{price:.3f}" if "JPY" in pair else f"{price:.5f}"

# ================= REST FUNCTIONS =================
async def get_account_info(session):
    async with session.get(f"{REST}/accounts/{ACCOUNT_ID}/summary") as r:
        data = await r.json()
        nav = float(data['account']['NAV'])
        margin_avail = float(data['account']['marginAvailable'])
        return nav, margin_avail

async def place_order(session, pair, price, atr, signal):
    nav, margin_avail = await get_account_info(session)
    units = calculate_units(nav, margin_avail, atr, RISK_PER_TRADE)
    if units == 0:
        print(f"{datetime.now()} | Skipping {pair}, insufficient margin")
        return

    stop_loss = price - SL_MULTIPLIER * atr if signal == 1 else price + SL_MULTIPLIER * atr
    take_profit = price + TP_MULTIPLIER * atr if signal == 1 else price - TP_MULTIPLIER * atr

    # Ensure minimum SL/TP distance
    min_dist = 0.01 if "JPY" in pair else 0.0001
    if abs(price - stop_loss) < min_dist: stop_loss = price + (min_dist if signal == -1 else -min_dist)
    if abs(price - take_profit) < min_dist: take_profit = price + (3 * min_dist if signal == 1 else -3 * min_dist)

    payload = {
        "order": {
            "instrument": pair,
            "units": str(units if signal == 1 else -units),
            "type": "MARKET",
            "timeInForce": "IOC",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": fmt_price(stop_loss, pair)},
            "takeProfitOnFill": {"price": fmt_price(take_profit, pair)}
        }
    }
    async with session.post(f"{REST}/accounts/{ACCOUNT_ID}/orders", json=payload) as r:
        res = await r.json()
        if "orderFillTransaction" in res:
            price_filled = float(res['orderFillTransaction']['price'])
            state['entry'][pair] = price_filled
            state['pos'][pair] = signal
            state['last_trade_time'][pair] = datetime.utcnow().timestamp()
            print(f"{datetime.now()} | Order filled: {pair} {units} units at {price_filled}")
        else:
            print(f"{datetime.now()} | Order rejected: {pair}, Response: {res}")

# ================= PROCESS TICK =================
async def process_tick(session, tick):
    pair = tick['instrument']
    async with state['locks'][pair]:
        bid = float(tick['bids'][0]['price'])
        ask = float(tick['asks'][0]['price'])
        price = (bid + ask) / 2

        # Update 15min data
        df_15min = state['price'][pair]
        now = pd.Timestamp.now(tz="UTC").floor("min")
        if not df_15min.empty and df_15min['time'].iloc[-1] == now:
            df_15min.at[df_15min.index[-1], 'high'] = max(df_15min['high'].iloc[-1], price)
            df_15min.at[df_15min.index[-1], 'low'] = min(df_15min['low'].iloc[-1], price)
            df_15min.at[df_15min.index[-1], 'close'] = price
        else:
            new_row = pd.DataFrame([[now, price, price, price, price]], columns=["time", "open", "high", "low", "close"])
            df_15min = pd.concat([df_15min, new_row], ignore_index=True)
        state['price'][pair] = df_15min.tail(300)

        # Update 1-hour data
        df_1h = state['price_1h'][pair]
        if not df_1h.empty and df_1h['time'].iloc[-1] == now:
            df_1h.at[df_1h.index[-1], 'high'] = max(df_1h['high'].iloc[-1], price)
            df_1h.at[df_1h.index[-1], 'low'] = min(df_1h['low'].iloc[-1], price)
            df_1h.at[df_1h.index[-1], 'close'] = price
        else:
            new_row = pd.DataFrame([[now, price, price, price, price]], columns=["time", "open", "high", "low", "close"])
            df_1h = pd.concat([df_1h, new_row], ignore_index=True)
        state['price_1h'][pair] = df_1h.tail(300)

        # Add indicators and get signals for both timeframes
        df_15min = add_indicators(df_15min)
        df_1h = add_indicators(df_1h)
        sig = get_signal(df_15min, df_1h)
        state['last_signal'][pair] = sig

        # Trade only if no position and cooldown passed
        if sig != 0 and state['pos'][pair] == 0 and (datetime.utcnow().timestamp() - state['last_trade_time'][pair]) > COOLDOWN * 60:
            await place_order(session, pair, price, df_15min['atr'].iloc[-1], sig)

# ================= STREAM PRICES =================
async def stream_prices():
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = {"instruments": ",".join(INSTR)}
    async with aiohttp.ClientSession(headers=headers) as session:
        while True:
            try:
                async with session.get(f"https://stream-fxpractice.oanda.com/v3/accounts/{ACCOUNT_ID}/pricing/stream",
                                       params=params) as r:
                    print("Streaming connected...")
                    async for line in r.content:
                        if not line: continue
                        data = json.loads(line.decode("utf-8"))
                        if "bids" in data:
                            asyncio.create_task(process_tick(session, data))
            except Exception as e:
                print("Stream error:", e)
                await asyncio.sleep(5)

# ================= MAIN =================
if __name__ == "__main__":
    asyncio.run(stream_prices())
