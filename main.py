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
STREAM_URL = f"wss://stream-fxpractice.oanda.com/v3/accounts/{ACCOUNT_ID}/pricing/stream"

HEADERS = {"Authorization": f"Bearer {API_KEY}"}

INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY"]

EMA_PERIOD = 50
RSI_PERIOD = 14
ATR_PERIOD = 14

RISK_PER_TRADE = 0.01
MAX_UNITS = 100000
SPREAD_FILTER_PIPS = 2
TRADE_COOLDOWN = 300  # seconds

LOG_FILE = "bot_log.csv"

last_trade_time = {pair: 0 for pair in INSTRUMENTS}
price_data = {pair: pd.DataFrame(columns=["time", "open", "high", "low", "close"]) for pair in INSTRUMENTS}

# ===== UTILITIES =====

def format_price(price, pair):
    precision = 3 if "JPY" in pair else 5
    return f"{price:.{precision}f}"

def get_account_equity():
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/summary"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return float(r.json()['account']['NAV'])

def get_open_positions(pair):
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/openPositions"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    positions = r.json()['positions']
    for pos in positions:
        if pos['instrument'] == pair:
            long_units = int(pos['long']['units'])
            short_units = int(pos['short']['units'])
            if long_units != 0 or short_units != 0:
                return True
    return False

# ===== INDICATORS =====

def add_indicators(df):
    df['ema'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()

    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()

    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )

    df['atr'] = df['tr'].rolling(ATR_PERIOD).mean()
    return df

def get_signal(df):
    last = df.iloc[-1]

    # Trend-following logic (better than original)
    if last['close'] > last['ema'] and last['rsi'] > 50:
        return 1
    elif last['close'] < last['ema'] and last['rsi'] < 50:
        return -1

    return 0

# ===== ORDER EXECUTION =====

def place_order(pair, units, stop_loss, take_profit):
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/orders"

    order_data = {
        "instrument": pair,
        "units": str(units),
        "type": "MARKET",
        "timeInForce": "FOK",
        "positionFill": "DEFAULT",
        "stopLossOnFill": {"price": stop_loss},
        "takeProfitOnFill": {"price": take_profit}
    }

    try:
        r = requests.post(url, headers=HEADERS, json={"order": order_data})
        r.raise_for_status()
        print(f"{datetime.now()} | Order placed: {pair} {units}")
        return r.json()
    except Exception as e:
        print("ORDER ERROR:", e)
        return None

def log_trade(pair, units, strategy, signal, price):
    df = pd.DataFrame([[datetime.now(), pair, units, strategy, signal, price]],
                      columns=["time", "pair", "units", "strategy", "signal", "price"])
    df.to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)

# ===== STREAM HANDLER =====

def process_tick(tick):
    global price_data

    pair = tick['instrument']
    bid = float(tick['bids'][0]['price'])
    ask = float(tick['asks'][0]['price'])

    pip = 0.01 if "JPY" in pair else 0.0001
    spread = (ask - bid) / pip

    if spread > SPREAD_FILTER_PIPS:
        return

    price = (bid + ask) / 2
    now = pd.Timestamp.now().floor("T")

    df = price_data[pair]

    if not df.empty and df['time'].iloc[-1] == now:
        df.at[df.index[-1], 'high'] = max(df['high'].iloc[-1], price)
        df.at[df.index[-1], 'low'] = min(df['low'].iloc[-1], price)
        df.at[df.index[-1], 'close'] = price
    else:
        new_row = pd.DataFrame([[now, price, price, price, price]],
                               columns=["time", "open", "high", "low", "close"])
        df = pd.concat([df, new_row], ignore_index=True)

    price_data[pair] = df.tail(200)

    if len(df) < ATR_PERIOD + 10:
        return

    df = add_indicators(df)
    signal = get_signal(df)

    if signal == 0:
        return

    # Cooldown check
    if time.time() - last_trade_time[pair] < TRADE_COOLDOWN:
        return

    # Position check
    if get_open_positions(pair):
        return

    equity = get_account_equity()
    atr = df['atr'].iloc[-1]
    last_price = df['close'].iloc[-1]

    stop_distance = 1.5 * atr
    risk_amount = equity * RISK_PER_TRADE
    units = int(min(risk_amount / stop_distance, MAX_UNITS))

    if units <= 0:
        return

    stop_loss = last_price - stop_distance if signal == 1 else last_price + stop_distance
    take_profit = last_price + (3 * atr) if signal == 1 else last_price - (3 * atr)

    result = place_order(
        pair,
        units if signal == 1 else -units,
        format_price(stop_loss, pair),
        format_price(take_profit, pair)
    )

    if result:
        last_trade_time[pair] = time.time()
        log_trade(pair, units, "EMA-RSI Trend", signal, last_price)

# ===== WEBSOCKET =====

def on_message(ws, message):
    data = json.loads(message)
    if 'bids' in data and 'asks' in data:
        process_tick(data)

def on_error(ws, error):
    print("WS ERROR:", error)

def on_close(ws, code, msg):
    print("WebSocket closed. Reconnecting...")
    time.sleep(5)
    start_streaming()

def start_streaming():
    params = "?instruments=" + ",".join(INSTRUMENTS)
    ws = websocket.WebSocketApp(
        STREAM_URL + params,
        header=[f"Authorization: Bearer {API_KEY}"],
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()

# ===== MAIN =====

if __name__ == "__main__":
    print(f"Starting trading bot at {datetime.now()}")

    ws_thread = threading.Thread(target=start_streaming)
    ws_thread.daemon = True
    ws_thread.start()

    while True:
        time.sleep(60)
