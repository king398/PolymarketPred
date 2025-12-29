# Quick Start Guide - Fixed Fair Price Engine

## ✅ What Was Fixed

**CRITICAL:** The original `fair_price.py` had a bug that **inverted all trading signals** (buy when should sell, sell when should buy). This has been fixed along with 6 other issues.

See [BUGFIX_SUMMARY.md](BUGFIX_SUMMARY.md) for complete details.

---

## 🚀 Quick Start

### 1. Verify Your Data Directory

Make sure you have the parquet data files:

```bash
# Check if data directory exists
ls -lh data/polymarket_minute_parquet/

# Should see files like:
# 01234567-89ab-cdef-0123-456789abcdef.parquet
```

If you don't have data yet, you need to run the data collection scripts first:
```bash
python3 src/data/get_market_ids.py
python3 src/data/scrape_price_history.py
```

### 2. Run the Test Suite (Recommended)

Verify everything works on your machine:

```bash
python3 src/process/test_fair_price.py
```

You should see:
```
✅ ALL TESTS COMPLETED
```

### 3. Configure Date Range

**Option A:** Environment variables (recommended)
```bash
export START_UTC="2024-11-01"
export DAYS=7
python3 src/process/fair_price.py
```

**Option B:** Edit the file directly
Open `src/process/fair_price.py` and modify line 53:
```python
START_UTC = os.environ.get("START_UTC", "2024-11-01")  # Change this date
DAYS = int(os.environ.get("DAYS", "7"))                # Change this number
```

### 4. Run Fair Price Analysis

```bash
python3 src/process/fair_price.py
```

**Expected output:**
```
Found 156 parquet files.
Active markets for copula: 23
Computing Kendall tau correlation matrix (optimized)...
Selected 45 strong pairs (|tau| >= 0.35).
Computing copula fair prices (family=gaussian, N_MC=30000)...
✅ Saved price pivot: polymarket_price_pivot.csv
✅ Saved fair prices: copula_fair_prices.csv
✅ Saved top opportunities: top_opportunities.csv
```

### 5. Examine Results

**Top opportunities CSV:**
```bash
head -20 top_opportunities.csv
```

Look for:
- `mispricing_A` / `mispricing_B`:
  - **Positive** = underpriced → **BUY signal**
  - **Negative** = overpriced → **SELL signal**
- `abs_mispricing_max`: Magnitude of opportunity (larger = stronger signal)
- `tau`: Correlation strength (higher = more reliable)

---

## 📊 Understanding the Output

### Fair Price CSV Columns

| Column | Meaning |
|--------|---------|
| `A`, `B` | The two correlated markets |
| `tau` | Kendall tau correlation (-1 to +1) |
| `pA_now` | Current price of market A |
| `pB_now` | Current price of market B |
| `fair_A_given_B_mean` | Fair price of A given B's current price |
| `fair_B_given_A_mean` | Fair price of B given A's current price |
| `mispricing_A` | `fair_A - pA_now` (positive = buy A) |
| `mispricing_B` | `fair_B - pB_now` (positive = buy B) |
| `abs_mispricing_max` | Max absolute mispricing (for ranking) |

### Example Interpretation

```csv
A,B,tau,pA_now,fair_A_given_B_mean,mispricing_A
"BTC > $100k by Dec?","BTC > $95k by Dec?",0.85,0.60,0.72,+0.12
```

**Read as:**
- These markets have 0.85 Kendall correlation (very strong)
- Market A currently trades at 0.60
- **Fair price is 0.72** (based on B's price via copula model)
- **Mispricing: +0.12** → A is **underpriced by 12%**
- **Signal: BUY market A** (it should be 0.72, not 0.60)

---

## ⚙️ Configuration Options

Edit `src/process/fair_price.py` lines 56-63:

```python
FREQ = "1min"                 # Time alignment (1min, 5min, etc.)
FAMILY = "gaussian"           # Copula: "gaussian" | "clayton" | "gumbel"
STRONG_TAU_ABS = 0.35         # Min |correlation| threshold (0.25-0.5)
MAX_PAIRS = 200               # Max pairs to analyze (runtime control)
N_MC = 30_000                 # Monte Carlo samples (higher = slower but accurate)
SEED = 42                     # Random seed for reproducibility
```

**Tuning tips:**
- Lower `STRONG_TAU_ABS` (e.g., 0.25) → more pairs, longer runtime
- Higher `N_MC` (e.g., 50,000) → better estimates, slower
- Use `FAMILY = "clayton"` for tail dependence in crashes
- Use `FAMILY = "gumbel"` for tail dependence in rallies

---

## 🔍 Common Issues

### Issue: "No parquet files found"
**Solution:** Check `DIR` path in line 47 or set data directory:
```bash
# Make sure this exists:
ls data/polymarket_minute_parquet/
```

### Issue: "No pairs passed tau threshold"
**Solution:** Lower `STRONG_TAU_ABS` to 0.25 or use longer date range (more data)

### Issue: Script is very slow
**Solution:**
- Reduce `MAX_PAIRS` to 50
- Reduce `N_MC` to 10,000
- Use shorter `DAYS` range (1-2 days instead of 7)

### Issue: All mispricing values are small (<0.01)
**Solution:** Markets may be efficiently priced, or:
- Try different date range
- Check if markets are active (not resolved/closed)
- Lower threshold to see smaller opportunities

---

## 📈 Next Steps

### 1. Backtest Your Strategy

Use `fair_price_self.py` to simulate trades:
```bash
python3 src/process/fair_price_self.py
```

This will:
- Find top correlated pairs
- Enter trades when mispricing > threshold (3%)
- Exit after holding period (10 minutes)
- Calculate P&L and win rate

**⚠️ Note:** This also had the same bug, so results from old runs are inverted!

### 2. Add Transaction Costs

The backtest currently ignores fees. Edit `fair_price_self.py` line 284:
```python
# Add this line:
fees = 0.02 * (abs(pA_enter) + abs(pA_exit))  # 2% round-trip
profit = s * (pA_exit - pA_enter) - fees
```

### 3. Optimize Parameters

Grid search for best threshold and holding period:
```python
for threshold in [0.01, 0.02, 0.03, 0.05]:
    for H in [5, 10, 15, 30]:
        # Run backtest and track P&L
```

### 4. Try Different Copulas

```bash
# Clayton (lower tail dependence - good for crashes)
sed -i '' 's/FAMILY = "gaussian"/FAMILY = "clayton"/' src/process/fair_price.py

# Gumbel (upper tail dependence - good for rallies)
sed -i '' 's/FAMILY = "gaussian"/FAMILY = "gumbel"/' src/process/fair_price.py
```

---

## 🧪 Development Workflow

**Always test before deploying:**

```bash
# 1. Make changes to fair_price.py
vim src/process/fair_price.py

# 2. Run test suite
python3 src/process/test_fair_price.py

# 3. Test on small date range first
START_UTC="2024-11-01" DAYS=1 python3 src/process/fair_price.py

# 4. If it works, run full analysis
START_UTC="2024-11-01" DAYS=30 python3 src/process/fair_price.py

# 5. Backtest
python3 src/process/fair_price_self.py
```

---

## 📚 Further Reading

- [BUGFIX_SUMMARY.md](BUGFIX_SUMMARY.md) - Detailed bug analysis
- [Copula Theory Primer](https://en.wikipedia.org/wiki/Copula_(probability_theory))
- [Kendall Tau Correlation](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient)

---

## 🆘 Getting Help

If you encounter issues:

1. **Check test suite output:**
   ```bash
   python3 src/process/test_fair_price.py 2>&1 | tee test_output.txt
   ```

2. **Verify data exists:**
   ```bash
   find data/polymarket_minute_parquet -name "*.parquet" | wc -l
   ```

3. **Enable debug mode (add to top of fair_price.py):**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

4. **Check Python version:**
   ```bash
   python3 --version  # Should be 3.8+
   ```

---

**Remember:** The old mispricing calculation was backwards. Any previous backtest results showing losses might actually be wins (and vice versa). Re-run everything with the fixed version!
