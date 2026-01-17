# Bitcoin and Gold: An Empirical Inquiry into Monetary Saturation

Analysis code and data for the paper investigating Bitcoin's long-term price dynamics relative to gold using a saturating exponential model.

## Model

The analysis fits a saturating exponential model to the Bitcoin/Gold price ratio:

```
ln(R(t)) = C + g*t + A(1 - e^(-λt))
```

Where:
- R(t) is Bitcoin's price in ounces of gold
- g is fixed at 0.02 (gold's ~2% annual supply growth)
- C, A, and λ are fitted parameters

## Files

- `btc_gold_analysis_rev14.py` - Main analysis script
- `btc_gold_training_2015_2024.csv` - Training data (2015-2024)
- `btc_gold_test_2025(&26Jan).csv` - Out-of-sample test data (2025-Jan 2026)
- `figures/` - Generated figures and analysis results

## Usage

```bash
python btc_gold_analysis_rev14.py
```

Outputs are saved to the `figures/` directory.

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- scipy

## Data Sources

- **Gold**: World Gold Council via GitHub + Kitco/Gold.org for recent months
- **Bitcoin**: CoinGecko historical data

## Author

S. James Biggs

## License

MIT
