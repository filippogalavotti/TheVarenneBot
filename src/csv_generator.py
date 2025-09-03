from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta
import time
import requests
from requests.exceptions import ReadTimeout, ConnectionError

# Your Binance API credentials
api_key = ''
api_secret = ''
client = Client(api_key, api_secret)

# Parameters
symbol = 'BTCEUR'
interval = Client.KLINE_INTERVAL_5MINUTE
start_date = datetime(2020, 1, 1)
end_date = datetime(2025, 1, 25)
limit = 1000  # Max number of candles per request

# Function to handle retries
def fetch_klines_with_retry(symbol, interval, start_str, end_str, max_retries=5):
    for attempt in range(max_retries):
        try:
            return client.get_historical_klines(symbol, interval, start_str, end_str)
        except (ReadTimeout, ConnectionError, requests.exceptions.Timeout) as e:
            wait = 2 ** attempt
            print(f"⚠️ Timeout error on attempt {attempt + 1}: {e}")
            print(f"Retrying in {wait} seconds...")
            time.sleep(wait)
        except Exception as e:
            print(f"❌ Other error: {e}")
            break
    return []

# Container for all data
all_candles = []

# Loop through time in chunks
current_start = start_date

while current_start < end_date:
    current_start_str = current_start.strftime("%d %b %Y %H:%M:%S")
    current_end = current_start + timedelta(minutes=5 * limit)
    if current_end > end_date:
        current_end = end_date
    current_end_str = current_end.strftime("%d %b %Y %H:%M:%S")

    print(f"🔄 Fetching data from {current_start_str} to {current_end_str}...")

    candles = fetch_klines_with_retry(symbol, interval, current_start_str, current_end_str)
    if not candles:
        print("⚠️ No data returned. Skipping to next range...")
    else:
        all_candles.extend(candles)

    current_start = current_end
    time.sleep(0.5)  # To respect Binance API rate limits

# Create DataFrame
columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
           'Close Time', 'Quote Asset Volume', 'Number of Trades',
           'Taker Buy Base Volume', 'Taker Buy Quote Volume', 'Ignore']
df = pd.DataFrame(all_candles, columns=columns)

# Keep only 'Open Time', 'High', 'Low'
df = df[['Open Time', 'Open', 'High', 'Low']]
df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')

# Save to CSV
csv_filename = 'dataset.csv'
df.to_csv(csv_filename, index=False)

# Print result
print(f"\n✅ Saved data to {csv_filename}")
print(f"📈 Total rows saved: {len(df)}")