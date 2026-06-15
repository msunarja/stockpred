import yfinance as yf
import joblib
import tensorflow as tf
import pandas as pd
import threading
import time
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed
import signaltrends.ticker_model_builders as tmb
import os

# To track time to process.
start = time.time()

# Create a single lock for all threads to share
yf_lock = threading.Lock()

trade_opportunities = []
process_failed_tickers = []

# Load watchlist
project_dir = os.path.dirname(__file__)
csv_path = os.path.join(project_dir, "watchlist.csv")
watchlist_df = pd.read_csv(csv_path)

interval = '1m'

# Parallel execution
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(tmb.ticker_prediction, rows=row, thread_lock=yf_lock,use_ensemble=True, interval = interval) for index, row in watchlist_df.iterrows()]
    for future in as_completed(futures):
        result, failed = future.result()
        if result:
            trade_opportunities.append(result)
        if failed:
            process_failed_tickers.append(failed)

end = time.time()

print(f"Execution time: {end - start:.2f} seconds")


import signaltrends.tradepredict_communications as tpc

html,text = tpc.build_prediction_email(trade_opportunities, .4, interval = interval)
print(tpc.send_email("tradepredict247 email",
               ["msunarja@hotmail.com"],
               f"Stock Day Trade ({interval}) Opportunities (Machine Learning)",
               html,
               text))