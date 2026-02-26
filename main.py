import os, asyncio, aiohttp, pandas as pd, numpy as np, json
from datetime import datetime
from dotenv import load_dotenv

# ================= CONFIG =================
load_dotenv()
API_KEY = os.getenv("OANDA_API_KEY")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
REST = "https://api-fxpractice.oanda.com/v3"
STREAM = f"https://stream-fxpractice.oanda.com/v3/accounts/{ACCOUNT_ID}/pricing/stream"
INSTR = ["EUR_USD","GBP_USD","USD_JPY"]
EMA, RSI, ATR = 50, 14, 14
RISK, MAXU, COOLD = 0.01, 100000, 300
LOG="trades.csv"

# ================= STATE =================
state = {
    "price": {p: pd.DataFrame(columns=["time","open","high","low","close"]) for p in INSTR},
    "last": {p: 0 for p in INSTR},
    "pos": {p: 0 for p in INSTR},
    "locks": {p: asyncio.Lock() for p in INSTR}  # per-instrument lock
}

# ================= UTILITIES =================
def add_ind(df):
    df['ema'] = df['close'].ewm(span=EMA, adjust=False).mean()
    d = df['close'].diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/RSI, adjust=False).mean()
    al = l.ewm(alpha=1/RSI, adjust=False).mean()
    df['rsi'] = 100-(100/(1+ag/al))
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(df['high']-df['low'],
                          np.maximum(abs(df['high']-df['prev_close']),
                                     abs(df['low']-df['prev_close'])))
    df['atr'] = df['tr'].rolling(ATR).mean()
    return df

def signal(df):
    l = df.iloc[-1]
    if l['close'] > l['ema'] and l['rsi'] > 55: return 1
    if l['close'] < l['ema'] and l['rsi'] < 45: return -1
    return 0

def calc_units(nav, atr): return max(int(min(nav*RISK/(1.5*atr), MAXU)),0)
def fmt(price, pair): return f"{price:.3f}" if "JPY" in pair else f"{price:.5f}"
def log_trade(pair, units, sig, price):
    pd.DataFrame([[datetime.now(), pair, units, sig, price]],
                 columns=["time","pair","units","signal","price"])\
      .to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)

# ================= REST CALLS =================
async def get_nav(session):
    async with session.get(f"{REST}/accounts/{ACCOUNT_ID}/summary") as r:
        return float((await r.json())['account']['NAV'])

async def place_order(session, pair, units, stop, tp):
    if units == 0: return None
    precision = 3 if "JPY" in pair else 5
    stop = round(float(stop), precision)
    tp = round(float(tp), precision)
    payload = {
        "order": {
            "instrument": pair,
            "units": str(units),
            "type": "MARKET",
            "timeInForce": "GTC",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": str(stop)},
            "takeProfitOnFill": {"price": str(tp)}
        }
    }
    async with session.post(f"{REST}/accounts/{ACCOUNT_ID}/orders", json=payload) as r:
        res = await r.json()
        if "orderFillTransaction" in res:
            print(f"{datetime.now()} | Order filled: {pair} {units} units | Price: {res['orderFillTransaction']['price']}")
        else:
            print(f"{datetime.now()} | Order NOT filled! Response: {res}")
        return res

async def refresh_positions(session):
    async with session.get(f"{REST}/accounts/{ACCOUNT_ID}/openPositions") as r:
        data = await r.json()
        open_pos = {p:0 for p in INSTR}
        for pos in data.get('positions', []):
            instrument = pos['instrument']
            units = int(float(pos['long']['units']) - float(pos['short']['units']))
            if units != 0:
                open_pos[instrument] = 1 if units > 0 else -1
        state['pos'].update(open_pos)

# ================= PROCESS TICK (PER INSTRUMENT) =================
async def process_tick(session, tick):
    pair = tick['instrument']
    async with state['locks'][pair]:  # prevent race conditions per pair
        await refresh_positions(session)
        bid = float(tick['bids'][0]['price'])
        ask = float(tick['asks'][0]['price'])
        price = (bid + ask)/2

        df = state['price'][pair]
        now = pd.Timestamp.now(tz="UTC").floor("min")

        if not df.empty and df['time'].iloc[-1] == now:
            df.at[df.index[-1], 'high'] = max(df['high'].iloc[-1], price)
            df.at[df.index[-1], 'low'] = min(df['low'].iloc[-1], price)
            df.at[df.index[-1], 'close'] = price
        else:
            new_row = pd.DataFrame([[now, price, price, price, price]],
                                   columns=["time","open","high","low","close"])
            df = pd.concat([df, new_row], ignore_index=True)
        state['price'][pair] = df.tail(200)

        if len(df) < ATR + 10: return
        df = add_ind(df)
        sig = signal(df)
        if sig == 0 or state['pos'][pair] != 0 or (datetime.utcnow().timestamp()-state['last'][pair]) < COOLD: return

        nav = await get_nav(session)
        atr = df['atr'].iloc[-1]
        units = calc_units(nav, atr)
        if units == 0: return

        stop = price-1.5*atr if sig==1 else price+1.5*atr
        tp = price+3*atr if sig==1 else price-3*atr

        await place_order(session, pair, units if sig==1 else -units, stop, tp)
        state['last'][pair] = datetime.utcnow().timestamp()
        state['pos'][pair] = sig
        log_trade(pair, units, sig, price)

# ================= STREAM =================
async def stream_prices():
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = {"instruments": ",".join(INSTR)}
    while True:
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(STREAM, params=params) as r:
                    print("Streaming connected...")
                    async for line in r.content:
                        if not line: continue
                        data = json.loads(line.decode("utf-8"))
                        if "bids" in data:
                            # Schedule per-instrument tasks
                            asyncio.create_task(process_tick(session, data))
        except Exception as e:
            print("Stream error:", e)
            await asyncio.sleep(5)

# ================= MAIN =================
if __name__=="__main__":
    asyncio.run(stream_prices())
