import os, asyncio, aiohttp, pandas as pd, numpy as np, json
from datetime import datetime
from dotenv import load_dotenv

# ================= CONFIG =================
load_dotenv()
API_KEY = os.getenv("OANDA_API_KEY")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
REST = "https://api-fxpractice.oanda.com/v3"
STREAM = f"https://stream-fxpractice.oanda.com/v3/accounts/{ACCOUNT_ID}/pricing/stream"

# Instruments to trade
INSTR = ["EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","USD_CAD"]

# Indicators and risk
EMA, RSI, ATR = 50, 14, 14
RISK_PER_TRADE = 0.01  # 1% of NAV
SL_MULTIPLIER = 1.5
TP_MULTIPLIER = 3
DASH_INTERVAL = 5  # seconds for dashboard refresh
COOLDOWN = 300  # seconds between trades per pair

# ================= STATE =================
state = {
    "price": {p: pd.DataFrame(columns=["time","open","high","low","close"]) for p in INSTR},
    "last_signal": {p: 0 for p in INSTR},
    "pos": {p: 0 for p in INSTR},
    "entry": {p: 0.0 for p in INSTR},
    "last_trade_time": {p: 0 for p in INSTR},
    "locks": {p: asyncio.Lock() for p in INSTR}
}

# ================= COLORS =================
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

# ================= UTILITIES =================
def add_indicators(df):
    df['ema'] = df['close'].ewm(span=EMA, adjust=False).mean()
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/RSI, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/RSI, adjust=False).mean()
    df['rsi'] = 100-(100/(1+avg_gain/avg_loss))
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(df['high']-df['low'],
                          np.maximum(abs(df['high']-df['prev_close']),
                                     abs(df['low']-df['prev_close'])))
    df['atr'] = df['tr'].rolling(ATR).mean()
    return df

def get_signal(df):
    last = df.iloc[-1]
    if last['close'] > last['ema'] and last['rsi'] > 55:
        return 1
    if last['close'] < last['ema'] and last['rsi'] < 45:
        return -1
    return 0

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
        print(f"{datetime.now()} | Skipping {pair}, not enough margin")
        return

    stop_loss = price - SL_MULTIPLIER*atr if signal==1 else price + SL_MULTIPLIER*atr
    take_profit = price + TP_MULTIPLIER*atr if signal==1 else price - TP_MULTIPLIER*atr

    payload = {
        "order": {
            "instrument": pair,
            "units": str(units if signal==1 else -units),
            "type": "MARKET",
            "timeInForce": "FOK",
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

# ================= DASHBOARD =================
async def dashboard():
    while True:
        os.system("clear")
        print(f"{'PAIR':6} | {'POS':3} | {'ENTRY':10} | {'LAST':10} | {'PNL':10} | {'SIGNAL':6}")
        print("-"*65)
        for pair in INSTR:
            pos = state['pos'][pair]
            entry = state['entry'][pair]
            last_price = state['price'][pair]['close'].iloc[-1] if not state['price'][pair].empty else 0
            pnl = (last_price - entry)*pos
            if "JPY" in pair: pnl *= 100
            pnl_color = GREEN if pnl>0 else RED if pnl<0 else YELLOW
            sig = state['last_signal'][pair]
            sig_color = BLUE if sig==1 else MAGENTA if sig==-1 else YELLOW
            print(f"{pair:6} | {pos:3} | {entry:10.5f} | {last_price:10.5f} | {pnl_color}{pnl:10.2f}{RESET} | {sig_color}{sig:6}{RESET}")
        await asyncio.sleep(DASH_INTERVAL)

# ================= STREAM PROCESS =================
async def process_tick(session, tick):
    pair = tick['instrument']
    async with state['locks'][pair]:
        bid = float(tick['bids'][0]['price'])
        ask = float(tick['asks'][0]['price'])
        price = (bid+ask)/2

        df = state['price'][pair]
        now = pd.Timestamp.now(tz="UTC").floor("min")
        if not df.empty and df['time'].iloc[-1]==now:
            df.at[df.index[-1],'high'] = max(df['high'].iloc[-1], price)
            df.at[df.index[-1],'low'] = min(df['low'].iloc[-1], price)
            df.at[df.index[-1],'close'] = price
        else:
            new_row = pd.DataFrame([[now, price, price, price, price]], columns=["time","open","high","low","close"])
            df = pd.concat([df, new_row], ignore_index=True)
        state['price'][pair] = df.tail(200)

        if len(df)<ATR+10: return
        df = add_indicators(df)
        sig = get_signal(df)
        state['last_signal'][pair] = sig

        # Only trade if signal exists, no current position, and cooldown passed
        if sig!=0 and state['pos'][pair]==0 and (datetime.utcnow().timestamp()-state['last_trade_time'][pair])>COOLDOWN:
            await place_order(session, pair, price, df['atr'].iloc[-1], sig)

# ================= STREAM PRICES =================
async def stream_prices():
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = {"instruments": ",".join(INSTR)}
    async with aiohttp.ClientSession(headers=headers) as session:
        asyncio.create_task(dashboard())
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
if __name__=="__main__":
    asyncio.run(stream_prices())
