import os
import json
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
import websocket
import threading

# ===== CONFIG =====
load_dotenv()

API_KEY = os.getenv("OANDA_API_KEY")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
BASE_URL = "https://api-fxpractice.oanda.com/v3"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY"]
EMA_PERIOD = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
RISK_PER_TRADE = 0.01
LOG_FILE = "bot_log.csv"

# ===== UTILITIES =====
def format_price(price, pair):
    precision = 3 if "JPY" in pair else 5
    return f"{price:.{precision}f}"

def add_indicators(df):
    df['ema'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -1*delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    df['rsi'] = 100 - (100 / (1 + avg_gain / avg_loss))
    df['tr'] = df['high'] - df['low']
    df['atr'] = df['tr'].rolling(ATR_PERIOD).mean()
    return df

def get_signal(df):
    last = df.iloc[-1]
    signal = 0
    if last['close'] > last['ema'] and last['rsi'] < 30:
        signal = 1
    elif last['close'] < last['ema'] and last['rsi'] > 70:
        signal = -1
    return signal

def place_order(pair, units, stop_loss=None, take_profit=None):
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/orders"
    order = {
        "order": {
            "instrument": pair,
            "units": str(units),
            "type": "MARKET",
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": stop_loss} if stop_loss else None,
            "takeProfitOnFill": {"price": take_profit} if take_profit else None
        }
    }
    r = requests.post(url, headers={"Authorization": f"Bearer {API_KEY}"}, json=order)
    r.raise_for_status()
    print(f"{datetime.now()} | Order placed: {pair} {units} units")
    return r.json()

def log_trade(pair, units, strategy, score, price):
    df = pd.DataFrame([[datetime.now(), pair, units, strategy, score, price]],
                      columns=["time", "pair", "units", "strategy", "score", "price"])
    df.to_csv(LOG_FILE, mode="a", header=not pd.io.common.file_exists(LOG_FILE), index=False)

# ===== STREAMING =====
price_data = {pair: pd.DataFrame(columns=["time", "open", "high", "low", "close"]) for pair in INSTRUMENTS}

def on_message(ws, message):
    data = json.loads(message)
    if 'bids' in data or 'asks' in data:
        process_tick(data)

def on_error(ws, error):
    print(f"Error: {error}")
    
def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed. Reconnecting...")
    time.sleep(3)
    start_streaming()

def on_open(ws):
    print("WebSocket opened!")

def start_streaming():
    url = f"wss://stream-fxpractice.oanda.com/v3/accounts/{ACCOUNT_ID}/pricing/stream"
    params = {"instruments": ",".join(INSTRUMENTS)}
    ws = websocket.WebSocketApp(url, header={"Authorization": f"Bearer {API_KEY}"}, on_message=on_message,
                                on_error=on_error, on_close=on_close)
    ws.run_forever()

# ===== PROCESS TICK =====
def process_tick(tick):
    global price_data
    pair = tick['instrument']
    bid = float(tick['bids'][0]['price'])
    ask = float(tick['asks'][0]['price'])
    price = (bid + ask) / 2

    df = price_data[pair]
    now = pd.Timestamp.now()

    # Convert tick to 1-min candle logic
    if not df.empty and (now - df['time'].iloc[-1]).seconds < 60:
        df.at[df.index[-1], 'high'] = max(df['high'].iloc[-1], price)
        df.at[df.index[-1], 'low'] = min(df['low'].iloc[-1], price)
        df.at[df.index[-1], 'close'] = price
    else:
        new_candle = pd.DataFrame([[now, price, price, price, price]], columns=["time", "open", "high", "low", "close"])
        df = pd.concat([df, new_candle], ignore_index=True)

    price_data[pair] = df.tail(EMA_PERIOD * 2)  # Keep only the most recent data

    if len(df) > ATR_PERIOD:
        df = add_indicators(df)
        signal = get_signal(df)
        if signal != 0:
            equity = 10000  # Example equity, replace with actual API call
            atr = df['atr'].iloc[-1]
            units = int((equity * RISK_PER_TRADE) / atr)
            last_price = df['close'].iloc[-1]
            stop_loss = last_price - 1.5 * atr if signal == 1 else last_price + 1.5 * atr
            take_profit = last_price + 3 * atr if signal == 1 else last_price - 3 * atr
            place_order(pair, units if signal == 1 else -units, format_price(stop_loss, pair), format_price(take_profit, pair))
            log_trade(pair, units, "RSI-EMA Strategy", signal, last_price)

# ===== MAIN =====
if __name__ == "__main__":
    print(f"Starting trading bot at {datetime.now()}")
    # Start WebSocket in a separate thread
    ws_thread = threading.Thread(target=start_streaming)
    ws_thread.daemon = True
    ws_thread.start()

    while True:
        time.sleep(60)  # Keep the main thread running to maintain WebSocket connection
