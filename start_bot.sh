python src/bot/get_active_slugs.py &
sleep 0.1
python src/bot/market_websocket.py &
sleep 0.1
python src/bot/market_bucket_publisher.py &
sleep 0.1
python src/bot/save_strike_price.py &