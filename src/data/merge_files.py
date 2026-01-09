import os
import glob
import pandas as pd

CRYPTO_LIST = ['BTC', 'ETH', 'SOL', ]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
CRYPTO_PATHS = [os.path.join(DATA_PATH, 'binance_data', x) for x in CRYPTO_LIST]
BINANCE_KLINE_COLUMNS = [
    "open_time",  # int64 (ms or µs)
    "open",  # float
    "high",  # float
    "low",  # float
    "close",  # float
    "volume",  # float (base asset volume)
    "close_time",  # int64 (ms or µs)
    "quote_asset_volume",  # float
    "number_of_trades",  # int
    "taker_buy_base_volume",  # float
    "taker_buy_quote_volume",  # float
    "ignore"  # float (always ignore)
]

files = [glob.glob(os.path.join(x, '*.csv')) for x in CRYPTO_PATHS]


for i,file in enumerate(files):
    print(file)
    df = pd.concat((pd.read_csv(f, header=None) for f in file), ignore_index=True)
    df.columns = BINANCE_KLINE_COLUMNS
    df = df.sort_values(by='open_time').reset_index(drop=True)
    df.to_parquet(os.path.join(CRYPTO_PATHS[i], f'{CRYPTO_LIST[i]}.parquet'))