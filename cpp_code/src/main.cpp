#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXUserAgent.h>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>     // std::isdigit, std::isspace

// Requires nlohmann/json: https://github.com/nlohmann/json
#include <json.hpp>

using json = nlohmann::json;

const std::string URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market";
const std::string ASSET_ID = "94123550151947355465084155553085412002538984957339940876169723664064204017612";

// Shared State (Thread Safe)
struct MarketState {
    double best_bid = 0.0;
    double best_ask = 0.0;
    double last_trade = 0.0;
};

std::mutex state_mutex;
MarketState current_state;

// Helper: Remove leading numbers (e.g. "42[...]" -> "[...]")
std::string clean_message(std::string_view raw) {
    size_t start = 0;
    while (start < raw.size() && std::isdigit(static_cast<unsigned char>(raw[start]))) start++;
    while (start < raw.size() && std::isspace(static_cast<unsigned char>(raw[start]))) start++;
    return std::string(raw.substr(start));
}

// Helper: Get current timestamp in milliseconds
uint64_t get_timestamp_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

// Helper: Update state from JSON event
void process_event(const json& event) {
    if (!event.contains("event_type")) return;
    std::string et = event["event_type"];

    std::lock_guard<std::mutex> lock(state_mutex);

    if (et == "book") {
        if (event.value("asset_id", "") != ASSET_ID) return;

        if (event.contains("bids")) {
            double max_bid = -1.0;
            for (const auto& item : event["bids"]) {
                double price = std::stod(item["price"].get<std::string>());
                if (price > max_bid) max_bid = price;
            }
            if (max_bid > 0) current_state.best_bid = max_bid;
        }

        if (event.contains("asks")) {
            double min_ask = 1e9;
            bool found = false;
            for (const auto& item : event["asks"]) {
                double price = std::stod(item["price"].get<std::string>());
                if (price < min_ask) {
                    min_ask = price;
                    found = true;
                }
            }
            if (found) current_state.best_ask = min_ask;
        }
    }
    else if (et == "last_trade_price") {
        if (event.value("asset_id", "") != ASSET_ID) return;
        if (event.contains("price")) {
            current_state.last_trade = event["price"].get<double>();
        }
    }
    else if (et == "price_change") {
        if (event.contains("price_changes")) {
            for (const auto& ch : event["price_changes"]) {
                if (ch.value("asset_id", "") != ASSET_ID) continue;
                if (ch.contains("best_bid")) current_state.best_bid = std::stod(ch["best_bid"].get<std::string>());
                if (ch.contains("best_ask")) current_state.best_ask = std::stod(ch["best_ask"].get<std::string>());
            }
        }
    }
}

static bool file_is_empty(std::ofstream& f) {
    // If you open with ios::app, tellp() gives current end position.
    return f.tellp() == std::streampos(0);
}

int main() {
    ix::initNetSystem();

    // --- CSV output ---
    std::ofstream csv("market_data.csv", std::ios::app);
    if (!csv.is_open()) {
        std::cerr << "Failed to open market_data.csv\n";
        return 1;
    }
    csv << std::ios::unitbuf; // instant flush

    // Write header if file is empty
    if (file_is_empty(csv)) {
        csv << "ts_ms,bid,ask\n";
    }

    std::atomic<uint64_t> message_count(0);

    ix::WebSocket webSocket;
    webSocket.setUrl(URL);

    webSocket.setOnMessageCallback([&csv, &message_count](const ix::WebSocketMessagePtr& msg) {
        if (msg->type == ix::WebSocketMessageType::Open) {
            std::cout << "Connected!" << std::endl;
        }
        else if (msg->type == ix::WebSocketMessageType::Message) {

            // 1. Parse Logic (Update State FIRST)
            bool updated = false;
            try {
                std::string cleaned = clean_message(msg->str);
                if (!cleaned.empty()) {
                    json j = json::parse(cleaned);
                    if (j.is_array()) {
                        for (const auto& item : j) {
                            if (item.is_object()) process_event(item);
                        }
                        updated = true;
                    } else if (j.is_object()) {
                        process_event(j);
                        updated = true;
                    }
                }
            } catch (...) {
                // Ignore parse errors
            }

            // 2. Save CSV (Only if parse succeeded)
            if (updated) {
                uint64_t ts = get_timestamp_ms();
                double bid, ask;

                {
                    std::lock_guard<std::mutex> lock(state_mutex);
                    bid = current_state.best_bid;
                    ask = current_state.best_ask;
                }

                // CSV row
                csv << ts << "," << bid << "," << ask << "\n";
                message_count++;
            }
        }
        else if (msg->type == ix::WebSocketMessageType::Error) {
            std::cout << "Error: " << msg->errorInfo.reason << std::endl;
        }
    });

    std::cout << "Connecting..." << std::endl;
    webSocket.start();

    while (webSocket.getReadyState() != ix::ReadyState::Open) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Subscribe
    std::string payload = "{\"type\": \"market\", \"assets_ids\": [\"" + ASSET_ID + "\"]}";
    webSocket.send(payload);
    std::cout << "Listening..." << std::endl;

    // Main Loop: ONLY prints Rate/Total
    uint64_t last_count = 0;
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        uint64_t current = message_count.load();
        uint64_t rate = current - last_count;

        std::cout << "Rate: " << rate << " updates/s | Total: " << current << std::endl;

        last_count = current;
    }

    return 0;
}
