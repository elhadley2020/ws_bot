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

# Correct streaming URL for practice account
STREAM = f"https://stream-fxpractice.oanda.com/v3/accounts/{ACCOUNT_ID}/pricing/stream"
REST = "https://api-fxpractice.oanda.com/v3"

# Instruments
INSTR = ["EUR_USD", "GBP_USD", "USD_JPY", "EUR_JPY"]

# Indicators & risk
EMA_FAST, EMA_SLOW = 50, 200
RSI_PERIOD = 14
MACD_SHORT, MACD_LONG, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD = 14
RISK_PER_TRADE = 0.01
SL_MULTIPLIER = 1.5
TP_MULTIPLIER = 3
COOLDOWN = 1  # minutes
ENTRY_TIMEFRAME = "5min"
TREND_TIMEFRAME = "15min"  # higher timeframe for trend confirmation

# ================= STATE =================
state = {
    "price": {p: pd.DataFrame(columns=["time","open","high","low","close"]) for p in INSTR},
    "trend": {p: pd.DataFrame(columns=["time","open","high","low","close"]) for p in INSTR},
    "pos": {p: 0 for p in INSTR},
    "entry": {p: 0.0 for p in INSTR},
    "last_trade_time": {p: 0 for p in INSTR},
    "locks": {p: asyncio.Lock() for p in INSTR}
}

# ================= INDICATORS =================
def add_indicators(df):
    df['ema50'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + avg_gain / avg_loss))
    
    df['macd'] = df['close'].ewm(span=MACD_SHORT, adjust=False).mean() - df['close'].ewm(span=MACD_LONG, adjust=False).mean()
    df['macd_signal'] = df['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['prev_close']), abs(df['low'] - df['prev_close'])))
    df['atr'] = df['tr'].rolling(ATR_PERIOD).mean()
    return df

def get_signal(df):
    last = df.iloc[-1]
    macd_cross = last['macd'] > last['macd_signal']
    ema_trend = last['ema50'] > last['ema200']
    rsi_ok = last['rsi'] > 50
    if macd_cross and ema_trend and rsi_ok:
        return 1
    elif not macd_cross and not ema_trend and rsi_ok:
        return -1
    return 0

def get_trend_signal(df):
    if len(df) < EMA_SLOW:
        return 0
    last = df.iloc[-1]
    if last['ema50'] > last['ema200']:
        return 1
    elif last['ema50'] < last['ema200']:
        return -1
    return 0

# ================= RISK CALC =================
def calculate_units(nav, margin_avail, atr):
    units = int((nav * RISK_PER_TRADE) / (SL_MULTIPLIER * atr))
    units = min(units, int(margin_avail))
    return max(units, 0)

def fmt_price(price, pair):
    return f"{price:.3f}" if "JPY" in pair else f"{price:.5f}"

# ================= REST =================
async def get_account_info(session):
    async with session.get(f"{REST}/accounts/{ACCOUNT_ID}/summary") as r:
        data = await r.json()
        nav = float(data['account']['NAV'])
        margin_avail = float(data['account']['marginAvailable'])
        return nav, margin_avail

async def place_order(session, pair, price, atr, signal):
    nav, margin_avail = await get_account_info(session)
    units = calculate_units(nav, margin_avail, atr)
    if units == 0:
        print(f"{datetime.now()} | {pair}: insufficient margin")
        return

    sl = price - SL_MULTIPLIER*atr if signal == 1 else price + SL_MULTIPLIER*atr
    tp = price + TP_MULTIPLIER*atr if signal == 1 else price - TP_MULTIPLIER*atr
    payload = {
        "order": {
            "instrument": pair,
            "units": str(units if signal==1 else -units),
            "type": "MARKET",
            "timeInForce": "IOC",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": fmt_price(sl, pair)},
            "takeProfitOnFill": {"price": fmt_price(tp, pair)}
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
        price = (bid + ask)/2

        # Update 5-min candles
        df = state['price'][pair]
        now = pd.Timestamp.now(tz="UTC").floor("min")
        if not df.empty and df['time'].iloc[-1] == now:
            df.at[df.index[-1],'high'] = max(df['high'].iloc[-1], price)
            df.at[df.index[-1],'low'] = min(df['low'].iloc[-1], price)
            df.at[df.index[-1],'close'] = price
        else:
            new_row = pd.DataFrame([[now, price, price, price, price]], columns=["time","open","high","low","close"])
            df = pd.concat([df, new_row], ignore_index=True)
        df.name = pair
        state['price'][pair] = df.tail(300)

        # Update 15-min trend candles
        trend_df = state['trend'][pair]
        trend_now = pd.Timestamp.now(tz="UTC").floor("15min")
        if not trend_df.empty and trend_df['time'].iloc[-1] == trend_now:
            trend_df.at[trend_df.index[-1],'high'] = max(trend_df['high'].iloc[-1], price)
            trend_df.at[trend_df.index[-1],'low'] = min(trend_df['low'].iloc[-1], price)
            trend_df.at[trend_df.index[-1],'close'] = price
        else:
            new_trend_row = pd.DataFrame([[trend_now, price, price, price, price]], columns=["time","open","high","low","close"])
            trend_df = pd.concat([trend_df, new_trend_row], ignore_index=True)
        trend_df.name = pair
        state['trend'][pair] = trend_df.tail(300)

        # Indicators
        if len(df) < ATR_PERIOD+10 or len(trend_df) < EMA_SLOW:
            return
        df = add_indicators(df)
        trend_df = add_indicators(trend_df)
        sig = get_signal(df)
        trend_sig = get_trend_signal(trend_df)

        # Trade only if trend aligned
        if sig != 0 and sig == trend_sig and state['pos'][pair]==0 and (datetime.utcnow().timestamp() - state['last_trade_time'][pair]) > COOLDOWN*60:
            await place_order(session, pair, price, df['atr'].iloc[-1], sig)

# ================= STREAM PRICES =================
async def stream_prices():
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = {"instruments": ",".join(INSTR)}

    timeout = aiohttp.ClientTimeout(sock_connect=10, sock_read=None)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        while True:
            try:
                print("Connecting to OANDA stream...")
                async with session.get(STREAM, params=params) as resp:
                    if resp.status != 200:
                        print(f"Stream error: HTTP {resp.status}, retry in 10s")
                        await asyncio.sleep(10)
                        continue

                    print("Streaming connected...")
                    async for line in resp.content:
                        if not line:
                            continue
                        line_str = line.decode("utf-8").strip()
                        if not line_str:
                            continue
                        try:
                            data = json.loads(line_str)
                        except:
                            continue
                        if data.get("type") == "HEARTBEAT":
                            continue
                        if "bids" in data:
                            asyncio.create_task(process_tick(session, data))
            except Exception as e:
                print(f"Stream exception: {e}. Reconnecting in 10s...")
                await asyncio.sleep(10)

# ================= MAIN =================
if __name__ == "__main__":
    asyncio.run(stream_prices())
