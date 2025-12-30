from datetime import datetime, timedelta
import pytz
import requests
import json

all_clob_token_ids = []
crypto_symbols = ["btc", "eth", "sol", "xrp"]
crypto_names = ["bitcoin", "ethereum", "solana", "xrp"]
ET = pytz.timezone("US/Eastern")

# Current time in ET
now_et = datetime.now(ET)

# Floor to last 15-minute boundary
bucket_start_et = now_et.replace(
    minute=(now_et.minute // 15) * 15,
    second=0,
    microsecond=0
)

bucket_starts_15_min = [
    int((bucket_start_et + timedelta(minutes=15 * i)).timestamp())
    for i in range(3)
]
print(bucket_starts_15_min)
print("Now ET:", now_et)
print("15-min bucket start ET:", bucket_start_et)

for symbol in crypto_symbols:
    for i, unix_ts in enumerate(bucket_starts_15_min):
        r = requests.get(f"https://gamma-api.polymarket.com/events/slug/{symbol}-updown-15m-{unix_ts}")
        data = r.json()
        yes_clob_token_id = eval(data["markets"][0]["clobTokenIds"])[0]
        all_clob_token_ids.append(
            {"slug": data['slug'], "clob_token_id": yes_clob_token_id, "market_position": i, "category": "15m",
             "primary_market_timestamp": str(datetime.fromtimestamp(unix_ts, ET)),
             "question": data["markets"][0]["question"]})
bucket_start_et_hour = now_et.replace(
    hour=now_et.hour,
    minute=0,
    second=0,
    microsecond=0
)
bucket_starts_1_hour = [
    f"{dt.strftime('%B').lower()}-{dt.day}-{dt.strftime('%I%p').lstrip('0').lower()}"
    for dt in (bucket_start_et_hour + timedelta(hours=i) for i in range(2))
]
for symbol in crypto_names:
    for i, timestamp in enumerate(bucket_starts_1_hour):
        r = requests.get(f"https://gamma-api.polymarket.com/events/slug/{symbol}-up-or-down-{timestamp}-et")
        data = r.json()
        yes_clob_token_id = eval(data["markets"][0]["clobTokenIds"])[0]
        all_clob_token_ids.append(
            {"slug": data['slug'], "clob_token_id": yes_clob_token_id, "market_position": i, "category": "1h",
             "primary_market_timestamp": str(bucket_start_et_hour + timedelta(hours=i)),
             "question": data["markets"][0]["question"]})
bucket_start_et_day = now_et.replace(
    day=now_et.day,
    hour=0,
    minute=0,
    second=0,
    microsecond=0
)
bucket_starts_1_day = [
    f"{dt.strftime('%B').lower()}-{dt.day}"
    for dt in (bucket_start_et_day + timedelta(days=i) for i in range(2))
]
print(bucket_starts_1_day)
for symbol in crypto_names:
    for i, timestamp in enumerate(bucket_starts_1_day):
        r = requests.get(f"https://gamma-api.polymarket.com/events/slug/{symbol}-up-or-down-on-{timestamp}")
        data = r.json()
        yes_clob_token_id = eval(data["markets"][0]["clobTokenIds"])[0]
        all_clob_token_ids.append(
            {"slug": data['slug'], "clob_token_id": yes_clob_token_id, "market_position": i, "category": "1d",
             "primary_market_timestamp": str(bucket_start_et_day + timedelta(days=i)),
             "question": data["markets"][0]["question"]})

bucket_starts_weekly = [
    f"{dt.strftime('%B').lower()}-{dt.day}"
    for dt in (bucket_start_et_day + timedelta(days=i) for i in range(5))
]
print(bucket_starts_weekly)
for symbol in crypto_names:
    for i, timestamp in enumerate(bucket_starts_weekly):
        r = requests.get(f"https://gamma-api.polymarket.com/events/slug/{symbol}-above-on-{timestamp}")
        data = r.json()
        for market_data in data["markets"]:
            try:
                max_price = eval(market_data['outcomePrices'])
            except KeyError as e:
                continue
            max_price = max([float(x) for x in max_price])
            if float(max_price) > 0.96:
                continue
            yes_clob_token_id = eval(market_data["clobTokenIds"])[0]
            ##all_clob_token_ids.append(
             ##   {"slug": data['slug'], "clob_token_id": yes_clob_token_id, "market_position": i, "category": "weekly",
             ##    "primary_market_timestamp": str(bucket_start_et_day + timedelta(days=i)),
              ##   "question": market_data["question"]})

#print(all_clob_token_ids)
print(len(all_clob_token_ids))
with open("/home/mithil/PycharmProjects/PolymarketPred/data/clob_token_ids.jsonl", "w") as f:
    for row in all_clob_token_ids:
        f.write(json.dumps(row) + "\n")
