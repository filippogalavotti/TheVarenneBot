#FdwJremArcLMJPtcaieK

from binance.client import Client
import tensorflow as tf
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
import numpy as np
import threading
import websocket
import json
import time
import queue
import requests
import os

class orderOCO:
    def __init__(self, price, abovePrice, belowPrice, quantity, side, timestamp):
        self.price = price
        self.abovePrice = abovePrice
        self.belowPrice = belowPrice
        self.quantity = quantity
        self.side = side
        self.timestamp = timestamp

    def __repr__(self):
        return (f"orderOCO(price={self.price}, abovePrice={self.abovePrice}, "
                f"belowPrice={self.belowPrice}, quantity={self.quantity})")
    
    def to_dict(self):
        return {
            'price': self.price,
            'abovePrice': self.abovePrice,
            'belowPrice': self.belowPrice,
            'quantity': self.quantity,
            'timestamp': self.timestamp
        }

stake = 0.05

startBudget = 0

cumulative_profit = 0

lastPrice_Lock = threading.Lock()
lastPrice = 0

ordersOCO_Lock = threading.Lock()
ordersOCO = []

price_queue = queue.Queue()

threadsRegister = {}

# Replace these with your Binance Testnet API credentials
api_key = ''
api_secret = ''

# Connect to Binance Testnet
BASE_URL = 'https://testnet.binance.vision/api'  # Spot testnet base URL
client = Client(api_key, api_secret)
client.API_URL = BASE_URL  # Override API URL

def send_telegram_message(text):
    BOT_TOKEN = ''
    CHAT_ID = ''
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    data = {'chat_id': CHAT_ID, 'text': text}
    requests.post(url, data=data)

def get_free_budget(asset):
    try:
        account_info = client.get_account()
        balances = account_info['balances']
        for balance in balances:
            if balance['asset'] == asset:
                return Decimal(balance['free']).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)
        return Decimal(0.0).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN) # If asset not found
    except Exception as e:
        message = f"Error fetching balance: {e}"
        print(message)
        send_telegram_message(message)
        return None
    
def get_price(symbol='BTCEUR'):

    try:

        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker["price"])
    
    except Exception as e:
        message = f"Error fetching ticker: {e}"
        print(message)
        send_telegram_message(message)
        return None

def format_quantity(qty_decimal):
    # Convert Decimal to plain decimal string
    s = format(qty_decimal, 'f')
    # Remove trailing zeros and trailing dot if any
    s = s.rstrip('0').rstrip('.') if '.' in s else s
    return s if s else '0'

def sellThreadRoutine():

    global cumulative_profit
    global startBudget
    global ordersOCO

    while True:

        try:

            websocket_price = price_queue.get()

            to_remove = []

            removedLong = False

            tpOrders = 0

            slOrders = 0

            expOrders = 0

            to_sell_qty = Decimal(0.0).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)

            weighted_to_sell_qty = Decimal(0.0).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)

            sold_qty = Decimal(0.0).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)

            weighted_sold_qty = Decimal(0.0).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)

            with ordersOCO_Lock:
            
                for order in ordersOCO:

                    if order.side == "LONG":

                        if websocket_price >= order.abovePrice: # Hit the take profit

                            try:


                                to_sell_qty += Decimal(order.quantity).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)

                                weighted_to_sell_qty += Decimal(float(order.quantity) * float(order.price)).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)

                                to_remove.append(order)

                                removedLong = True

                                tpOrders += 1

                            except Exception as e:
                                print(f'[sellThread] Unexpected error: {e}')

                        elif websocket_price <= order.belowPrice: # Hit the stop loss

                            try:

                                to_sell_qty += Decimal(order.quantity).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)

                                weighted_to_sell_qty += Decimal(float(order.quantity) * float(order.price)).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)

                                to_remove.append(order)

                                removedLong = True

                                slOrders += 1

                            except Exception as e:
                                print(f'[sellThread] Unexpected error: {e}')

                        elif (time.time() > (order.timestamp + 76800)): # The position expired

                            try:

                                to_sell_qty += Decimal(order.quantity).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)

                                weighted_to_sell_qty += Decimal(float(order.quantity) * float(order.price)).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)

                                to_remove.append(order)

                                removedLong = True

                                expOrders += 1

                            except Exception as e:
                                print(f'[sellThread] Unexpected error: {e}')


            # IF SOME ORDERS ARE TRIGGERED, THE SELL ROUTING IS STARTED
            if len(to_remove) > 0:

                # HANDLING LONG POSITIONS

                if removedLong:

                    avgToSellPrice = weighted_to_sell_qty / to_sell_qty

                    message = f"Selling {to_sell_qty} BTC\nBought @ {avgToSellPrice:.2f} (average)"

                    print(message)
                    send_telegram_message(message)

                    response = client.order_market_sell(
                        symbol="BTCEUR",
                        quantity=Decimal(to_sell_qty).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)
                    )
                            
                    fills = response['fills']

                    for fill in fills:

                        fill_qty = Decimal(fill['qty']).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)
                        fill_price = Decimal(fill['price']).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)

                        sold_qty += fill_qty
                        weighted_sold_qty += (fill_qty * fill_price)

                    avgSoldPrice = weighted_sold_qty / sold_qty

                    message = f"Sold {sold_qty} BTC @ {avgSoldPrice:.2f} (average)"

                    print(message)
                    send_telegram_message(message)

                    profit = (avgSoldPrice * sold_qty) - (avgToSellPrice * to_sell_qty)

                    cumulative_profit += profit

                    message = f"Result: {profit:.2f} EUR,\nProfit: {cumulative_profit:.2f} EUR ({((cumulative_profit / startBudget) * 100):.2f} %)"

                    print(message)
                    send_telegram_message(message)

                with ordersOCO_Lock:
                    # Remove all executed orders
                    for order in to_remove:
                        ordersOCO.remove(order)

                    message = f"🗒️Open Orders: {len(ordersOCO)}\nRemoved:\nTP: {tpOrders}\nSL: {slOrders}\nEXP: {expOrders}"
                    send_telegram_message(message)
                
        except Exception as e:
            message = f"[sellThread] Error processing message: {e}"
            print(message)
            send_telegram_message(message)

def on_message(ws, message):

    global lastPrice

    try:
        data = json.loads(message)
        websocket_price = float(data['c'])  # 'p' is the price field in trade stream

        price_queue.put(websocket_price)

        with lastPrice_Lock:
            lastPrice = websocket_price

    except Exception as e:
        message = f"[on_message] Error processing message: {e}"
        print(message)
        send_telegram_message(message)

def on_error(ws, error):
    message = f"WebSocket Error: {error}"
    print(message)
    send_telegram_message(message)

def on_close(ws, close_status_code, close_msg):
    message = f"[sellThread] WebSocket Closed: {close_msg}"
    print(message)
    send_telegram_message(message)

def on_open(ws):
    message = "[sellThread] WebSocket Connection Opened"
    print(message)
    send_telegram_message(message)

def priceThreadRoutine():
    url = "wss://stream.binance.com:9443/ws/btceur@ticker"

    while True:
        try:

            ws = websocket.WebSocketApp(
                url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.run_forever(ping_interval=120, ping_timeout=100)

        except Exception as e:
            message = f"WebSocket error: {e}"
            print(message)
            send_telegram_message(message)

        message = "[priceThread] WebSocket disconnected. Reconnecting in 10 seconds..."
        print(message)
        send_telegram_message(message)
        time.sleep(10)

def buyThreadRoutine():
    global stake
    global lastPrice

    # IMPORT MODELS

    HIGHmodel = tf.keras.models.load_model('/root/Data/Varenne/HIGHModel.keras') # Created by HIGHModel.py
    LOWmodel = tf.keras.models.load_model('/root/Data/Varenne/LOWModel.keras') # Created by LOWModel.py

    # Get 5-minute candles for BTCEUR
    candles = client.get_klines(
        symbol='BTCEUR',
        interval=Client.KLINE_INTERVAL_5MINUTE,
        limit=513
    )

    high_features = [float(candle[2]) for candle in candles]  # candle[2] is high
    low_features = [float(candle[3]) for candle in candles]   # candle[3] is low

    print(f"Length of high_features: {len(high_features)}")
    print(f"Length of low_features: {len(low_features)}")

    high_features.pop(512)
    low_features.pop(512)

    reference_timestamp = candles[511][0] # The last candle is the reference for future ones

    high_features = np.array(high_features)
    low_features = np.array(low_features)

    start_time = time.time()

    while True:

        time.sleep(0.5)

        try:

            # Get new candle
            candle = client.get_klines(
                symbol='BTCEUR',
                interval=Client.KLINE_INTERVAL_5MINUTE,
                limit=2
            )
        
        except Exception as e:
            print(f'\r[buyThread] Unexpected error: {str(e):<80}\n', end='', flush=True)
            start_time = time.time()
            continue

        candle = candle[-2]

        if(candle[0] != reference_timestamp):

            print(f'\r[buyThread] {datetime.fromtimestamp(candle[0] / 1000).replace(microsecond=0).isoformat()}-{datetime.fromtimestamp(candle[6] / 1000).replace(microsecond=0).isoformat():<80}', flush=True)
            
            # Adjusting reference timestamp to new open value
            reference_timestamp = candle[0]

            # Shifting features arrays
            high_features[:-1] = high_features[1:]
            high_features[-1] = float(candle[2])

            low_features[:-1] = low_features[1:]
            low_features[-1] = float(candle[3])

            # Normalising features arrays
            high_mean = high_features.mean()
            high_std = high_features.std()

            normalised_high = (high_features - high_mean) / high_std

            low_mean = low_features.mean()
            low_std = low_features.std()

            normalised_low = (high_features - high_mean) / high_std

            # Feeding models and denormalising predictions
            features = np.zeros((1, 512, 2), dtype=np.float32)

            features[0, :, 0] = normalised_high  # channel 0: normalized high_features
            features[0, :, 1] = normalised_low   # channel 1: normalized low_features

            high = HIGHmodel.predict(features)[0][0]

            high = ( high * high_std ) + high_mean

            low = LOWmodel.predict(features)[0][0]

            low = ( low * low_std ) + low_mean

            with lastPrice_Lock:
                price = lastPrice

            btcBudget = float(get_free_budget('BTC'))
            eurBudget = float(get_free_budget('EUR'))

            print(f"[buyThread] btcBudget: {btcBudget:.8f} BTC, eurBudget: {eurBudget:.2f} EUR, Price: {price:.2f}, High: {high:.2f}, Low: {low:.2f}")

            if ((high <= low) or (price > high) or (price < low)):
                start_time = time.time()
                continue

            high_delta = high - price
            low_delta = price - low

            if (high_delta > low_delta) and (eurBudget >= 50):  #Long position

                high = round(high, 2)
                low = round(low, 2)

                try:
                    
                    quoteOrderQty = Decimal(int(max(eurBudget*stake, 50))).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)

                    response = client.order_market_buy(
                        symbol="BTCEUR",
                        quoteOrderQty=quoteOrderQty
                    )

                    fills = response['fills']

                    btBoughtQty = Decimal(0.0).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)
                    btAverageBoughtQty = Decimal(0.0).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)

                    for fill in fills:

                        fill_qty = Decimal(fill['qty']).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)
                        fill_price = Decimal(fill['price']).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)

                        btBoughtQty += fill_qty
                        btAverageBoughtQty += (fill_qty * fill_price)

                    btAverageBoughtPrice = Decimal(btAverageBoughtQty / btBoughtQty).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)

                    # Print the rounded quantity for debugging
                    print(f"[buyThread] Bought: {btBoughtQty} BTC at {btAverageBoughtPrice:.2f}")

                    send_telegram_message(f"Bought {btBoughtQty} BTC @ {btAverageBoughtPrice:.2f},\nSL: {low:.2f}, TP: {high:.2f}")

                    placed_order = orderOCO(
                        price = btAverageBoughtPrice,
                        abovePrice = high,
                        belowPrice = low,
                        quantity = btBoughtQty,
                        side = "LONG",
                        timestamp = time.time()
                    )

                    with ordersOCO_Lock:
                        ordersOCO.append(placed_order)

                    print("[buyThread] Added order to register")

                except Exception as e:
                    print(f'[buyThread] Unexpected error: {e}')

            start_time = time.time()

        else:
            print(f'\r[buyThread] Waiting for new candle... {int(time.time() - start_time)}s', end='', flush=True)

def register_thread(name, target):
    thread = threading.Thread(target=target, name=name)
    thread.daemon = True
    thread.start()
    threadsRegister[name] = {'thread': thread, 'target': target}

# Diagnostics thread that restarts dead threads
def diagnosticsThreadRoutine(interval=5):
    print("[Diagnostics] Started")
    while True:
        for name, info in threadsRegister.items():
            thread = info['thread']
            if thread.is_alive():
                message = f" - {name}: 🟢 ALIVE"
                print(message)
                send_telegram_message(message)
            else:
                message = f" - {name}: 🔴 DEAD — restarting..."
                print(message)
                send_telegram_message(message)

                new_thread = threading.Thread(target=info['target'], name=name)
                new_thread.daemon = True
                new_thread.start()
                threadsRegister[name]['thread'] = new_thread

        btcBudget = float(get_free_budget('BTC'))
        eurBudget = float(get_free_budget('EUR'))

        message = f"🗒️Open Orders: {len(ordersOCO)}\neurBudget: {eurBudget:.2f}, btcBudget: {btcBudget:.8f} BTC"
        send_telegram_message(message)

        with ordersOCO_Lock:

            for order in ordersOCO:

                if order.side == "LONG":
                    message = f"LONG - {datetime.fromtimestamp(order.timestamp).strftime('%Y-%m-%d %H:%M:%S')}\nQTY: {Decimal(order.quantity).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)} BTC @ {order.price:.2f}\nTP: {order.abovePrice:.2f}\nSL: {order.belowPrice:.2f}"
                    send_telegram_message(message)

        time.sleep(3600)

if __name__ == "__main__":

    try:

        # Selling the entire BTC Budget for ETH (Ignore this part if your  EUR and BTC budget is already standardized)

        btcStartBudget = get_free_budget('BTC')

        if btcStartBudget > Decimal(0.0001):

            print("Selling BTC...")

            response = client.order_market_buy(
                symbol="ETHBTC",
                quoteOrderQty=btcStartBudget
            )

        eurStartBudget = get_free_budget("EUR")

        print(f"eurStartBudget: {eurStartBudget:.2f}")

        to_request = int(Decimal(1000) - eurStartBudget)

        print(f"to_request: {to_request}")

        if to_request > 10:

            print("Selling ETH to reach standard budget...")

            response = client.order_market_sell(
                symbol="ETHEUR",
                quoteOrderQty=to_request
            )

            print(response)

        elif to_request < -10:

            to_request = abs(to_request)

            print("Buying ETH to reach standard budget...")

            response = client.order_market_buy(
                symbol="ETHEUR",
                quoteOrderQty=to_request
            )

            print(response)

        startBudget = get_free_budget("EUR")

    except Exception as e:

        message = f'[main] Unexpected error: {e}'
        print(message)
        send_telegram_message(message)

    send_telegram_message("🐎 Svegliando Varenne...")

    eurStartBudget = get_free_budget("EUR")
    btcStartBudget = get_free_budget("BTC")

    message = f"startBudget: {startBudget:.2f} EUR\neurStartBudget: {eurStartBudget:.2f} EUR\nbtcStartBudget: {btcStartBudget} BTC"
    print(message)
    send_telegram_message(message)

    register_thread("buyThread", buyThreadRoutine)
    register_thread("sellThread", sellThreadRoutine)
    register_thread("priceThread", priceThreadRoutine)

    diagnosticsThread = threading.Thread(target=diagnosticsThreadRoutine)
    diagnosticsThread.daemon = True
    diagnosticsThread.start()

    # Example of keeping the main thread alive
    try:
        while True:
            pass  # Could also do other things here
    except KeyboardInterrupt:
        send_telegram_message("💤 Varenne va a letto...")
        print("\nExiting...")
