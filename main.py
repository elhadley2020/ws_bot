import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import json
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
MACD_SHORT = 12
MACD_LONG = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14
RISK_PER_TRADE = 0.01
SL_MULTIPLIER = 1.5
TP_MULTIPLIER = 3
COOLDOWN = 4  # in candles
TIMEFRAME = "15min"  # main timeframe for trend

# ================= STATE =================
state = {
    "price": {p: pd.DataFrame(columns=["time","open","high","low","close"]) for p in INSTR},
    "last_signal": {p: 0 for p in INSTR},
    "pos": {p: 0 for p in INSTR},
    "entry": {p: 0.0 for p in INSTR},
    "last_trade_time": {p: 0 for p in INSTR},
    "locks": {p: asyncio.Lock() for p in INSTR}
}

# ================= INDICATORS =================
def add_indicators(df):
    # EMA
    df['ema50'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + avg_gain / avg_loss))
    
    # MACD
    df['macd'] = df['close'].ewm(span=MACD_SHORT, adjust=False).mean() - df['close'].ewm(span=MACD_LONG, adjust=False).mean()
    df['macd_signal'] = df['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    
    # ATR (Average True Range)
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['prev_close']), abs(df['low'] - df['prev_close'])))
    df['atr'] = df['tr'].rolling(ATR_PERIOD).mean()

    return df

def get_signal(df):
    last = df.iloc[-1]
    
    # Check for MACD crossover (bullish or bearish)
    macd_crossover = last['macd'] > last['macd_signal']
    # Check for EMA alignment (bullish or bearish trend)
    ema_alignment = last['ema50'] > last['ema200']
    # Check for RSI conditions
    rsi_overbought = last['rsi'] > 70
    rsi_oversold = last['rsi'] < 30

    # Trading signals
    if macd_crossover and ema_alignment and not rsi_overbought:
        return 1  # Buy signal (long)
    elif not macd_crossover and not ema_alignment and not rsi_oversold:
        return -1  # Sell signal (short)
    
    return 0  # No signal

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

    stop_loss = price - SL_MULTIPLIER*atr if signal == 1 else price + SL_MULTIPLIER*atr
    take_profit = price + TP_MULTIPLIER*atr if signal == 1 else price - TP_MULTIPLIER*atr

    # Ensure minimum SL/TP distance
    min_dist = 0.01 if "JPY" in pair else 0.0001
    if abs(price - stop_loss) < min_dist: stop_loss = price + (min_dist if signal == -1 else -min_dist)
    if abs(price - take_profit) < min_dist: take_profit = price + (3*min_dist if signal == 1 else -3*min_dist)

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

        df = state['price'][pair]
        now = pd.Timestamp.now(tz="UTC").floor("min")
        if not df.empty and df['time'].iloc[-1] == now:
            df.at[df.index[-1], 'high'] = max(df['high'].iloc[-1], price)
            df.at[df.index[-1], 'low'] = min(df['low'].iloc[-1], price)
            df.at[df.index[-1], 'close'] = price
        else:
            new_row = pd.DataFrame([[now, price, price, price, price]], columns=["time", "open", "high", "low", "close"])
            df = pd.concat([df, new_row], ignore_index=True)
        state['price'][pair] = df.tail(300)

        if len(df) < ATR_PERIOD + 10:
            return
        df = add_indicators(df)
        sig = get_signal(df)
        state['last_signal'][pair] = sig

        # Trade only if no position and cooldown passed
        if sig != 0 and state['pos'][pair] == 0 and (datetime.utcnow().timestamp() - state['last_trade_time'][pair]) > COOLDOWN * 60:
            await place_order(session, pair, price, df['atr'].iloc[-1], sig)

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
                        if not line:
                            continue
                        data = json.loads(line.decode("utf-8"))
                        if "bids" in data:
                            asyncio.create_task(process_tick(session, data))
            except Exception as e:
                print("Stream error:", e)
                await asyncio.sleep(5)

# ================= MAIN =================
if __name__ == "__main__":
    asyncio.run(stream_prices())
