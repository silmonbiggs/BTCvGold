"""
Live Dashboard Generator for Bitcoin's Gold Price Model
Fetches daily BTC and Gold prices, generates plots and HTML scorecard.

Usage:
    python update_dashboard.py

Outputs:
    index.html            - Dashboard page (GitHub Pages root)
    dashboard_ratio.png   - Daily BTC/Gold ratio vs model
    dashboard_cusum.png   - Monthly CUSUM scorecard
    data/daily_prices.csv - Append-only daily price log
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).parent
SITE = ROOT  # GitHub Pages serves from repo root
DATA_DIR = ROOT / 'data'
DAILY_CSV = DATA_DIR / 'daily_prices.csv'

# Training and test data (for historical context in plots)
TRAINING_CSV = ROOT / 'btc_gold_training_2015_2024.csv'
TEST_CSV = ROOT / 'btc_gold_test_2025(&26Jan).csv'
UPDATE_CSV = ROOT / 'btc_gold_update.csv'

# Fitted model parameters (from training on 2015-01 to 2024-12)
MODEL_C = -2.252333
MODEL_A = 5.617963
MODEL_LAMBDA = 0.283095
MODEL_G = 0.020
START_DATE = pd.to_datetime('2015-01-01')

# Residual standard deviations (log space)
SIGMA_POST2023 = 0.2554
SIGMA_FULL = 0.4782

# Sequential testing boundary constants (alpha=0.05, K=120 looks, 2M MC sims)
C_IID = 2.625
C_CORRECTED = 8.50
C_TRAJECTORY = C_CORRECTED * (SIGMA_FULL / SIGMA_POST2023)

RHO_TRAINING = 0.898


# ============================================================================
# MODEL
# ============================================================================

def model_ln_ratio(t):
    """Saturating exponential: ln(R(t)) = C + g*t + A(1 - e^(-lambda*t))"""
    return MODEL_C + MODEL_G * t + MODEL_A * (1 - np.exp(-MODEL_LAMBDA * t))


# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_prices():
    """Fetch recent BTC and Gold prices via yfinance. Returns DataFrame."""
    import yfinance as yf

    # Determine start date: day after last entry in daily CSV, or 2015-01-01
    if DAILY_CSV.exists():
        existing = pd.read_csv(DAILY_CSV)
        existing['Date'] = pd.to_datetime(existing['Date'])
        last_date = existing['Date'].max()
        fetch_start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        existing = pd.DataFrame()
        fetch_start = '2015-01-01'

    today = datetime.now().strftime('%Y-%m-%d')
    if fetch_start > today:
        print("  Daily CSV is up to date.")
        return existing

    print(f"  Fetching prices from {fetch_start} to {today}...")

    btc = yf.download('BTC-USD', start=fetch_start, end=today, progress=False)
    gold = yf.download('GC=F', start=fetch_start, end=today, progress=False)

    if btc.empty or gold.empty:
        print("  No new data available from yfinance.")
        return existing

    # Handle MultiIndex columns from yfinance
    if isinstance(btc.columns, pd.MultiIndex):
        btc = btc.droplevel(level=1, axis=1)
    if isinstance(gold.columns, pd.MultiIndex):
        gold = gold.droplevel(level=1, axis=1)

    # Align on dates where both traded
    btc_close = btc[['Close']].rename(columns={'Close': 'USD_per_Bitcoin'})
    gold_close = gold[['Close']].rename(columns={'Close': 'USD_per_Gold_oz'})
    merged = btc_close.join(gold_close, how='inner').dropna()

    if merged.empty:
        print("  No overlapping trading days found.")
        return existing

    merged['Gold_oz_per_Bitcoin'] = merged['USD_per_Bitcoin'] / merged['USD_per_Gold_oz']
    merged = merged.reset_index()
    merged = merged.rename(columns={'index': 'Date', 'Datetime': 'Date'})
    # Normalize date column name (yfinance may use 'Date' or 'Datetime')
    if 'Date' not in merged.columns:
        for col in merged.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                merged = merged.rename(columns={col: 'Date'})
                break
    merged['Date'] = pd.to_datetime(merged['Date']).dt.date

    new_rows = merged[['Date', 'USD_per_Gold_oz', 'USD_per_Bitcoin', 'Gold_oz_per_Bitcoin']]
    print(f"  Fetched {len(new_rows)} new daily rows.")

    if not existing.empty:
        existing['Date'] = existing['Date'].dt.date
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset='Date', keep='last')
        combined = combined.sort_values('Date').reset_index(drop=True)
    else:
        combined = new_rows.sort_values('Date').reset_index(drop=True)

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(DAILY_CSV, index=False)
    print(f"  Saved {len(combined)} total rows to {DAILY_CSV}")

    # Restore Date as datetime for downstream use
    combined['Date'] = pd.to_datetime(combined['Date'])
    return combined


# ============================================================================
# MONTHLY DATA (for CUSUM)
# ============================================================================

def load_monthly_data():
    """Load all monthly data from training + test + update CSVs."""
    frames = []
    for path in [TRAINING_CSV, TEST_CSV, UPDATE_CSV]:
        if path.exists():
            df = pd.read_csv(path)
            df['Date'] = pd.to_datetime(df['Date'])
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset='Date', keep='last')
    combined = combined.sort_values('Date').reset_index(drop=True)
    return combined


def compute_oos_cusum(monthly_df):
    """Compute z-scores and CUSUM for out-of-sample months (2025-01 onward)."""
    oos = monthly_df[monthly_df['Date'] >= '2025-01-01'].copy()
    if oos.empty:
        return oos
    t = (oos['Date'] - START_DATE).dt.days / 365.25
    ln_actual = np.log(oos['Gold_oz_per_Bitcoin'].values)
    ln_pred = model_ln_ratio(t.values)
    oos['ln_actual'] = ln_actual
    oos['ln_pred'] = ln_pred
    oos['residual'] = ln_actual - ln_pred
    oos['z_score'] = oos['residual'] / SIGMA_POST2023
    oos['cusum'] = oos['z_score'].cumsum()
    return oos


# ============================================================================
# PLOT 1: DAILY RATIO VS MODEL
# ============================================================================

def create_ratio_plot(daily_df, output_path):
    """Daily BTC/Gold ratio with model curve and confidence bands."""
    plt.rcParams.update({
        'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
        'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12,
    })
    fig, ax = plt.subplots(figsize=(10, 7))

    # Model curve from 2015 to 2036
    t_model = np.linspace(0, 21, 500)
    dates_model = np.array([START_DATE + timedelta(days=d*365.25) for d in t_model])
    ln_model = model_ln_ratio(t_model)
    ratio_model = np.exp(ln_model)

    # Regime change at 2023-01-01: wider bands before, tighter after
    regime_date = pd.to_datetime('2023-01-01')
    pre = dates_model < regime_date
    post = ~pre

    # Pre-2023 bands (full-sample sigma)
    for k, alpha, label in [(1, 0.15, ''), (2, 0.10, ''), (3, 0.06, '')]:
        upper = np.exp(ln_model + k * SIGMA_FULL)
        lower = np.exp(ln_model - k * SIGMA_FULL)
        ax.fill_between(dates_model[pre], lower[pre], upper[pre],
                        color='#F18F01', alpha=alpha)

    # Post-2023 bands (reduced volatility sigma)
    for k, alpha, label in [(1, 0.15, '68% band'), (2, 0.10, '95% band'),
                            (3, 0.06, '99.7% band')]:
        upper = np.exp(ln_model + k * SIGMA_POST2023)
        lower = np.exp(ln_model - k * SIGMA_POST2023)
        ax.fill_between(dates_model[post], lower[post], upper[post],
                        color='#F18F01', alpha=alpha, label=label)

    # Model line
    ax.semilogy(dates_model, ratio_model, '-', color='#F18F01', linewidth=2,
                label='Model prediction', zorder=5)

    # Daily data as solid line
    if not daily_df.empty:
        dates = pd.to_datetime(daily_df['Date'])
        ratios = daily_df['Gold_oz_per_Bitcoin'].values
        ax.semilogy(dates, ratios, '-', color='#2E86AB', linewidth=1.0,
                    alpha=0.7, label='Daily BTC/Gold ratio', zorder=3)

        # Latest point
        latest = daily_df.iloc[-1]
        ax.semilogy(pd.to_datetime(latest['Date']), latest['Gold_oz_per_Bitcoin'],
                    'o', color='#2E86AB', markersize=8, zorder=10,
                    markeredgecolor='white', markeredgewidth=1.0,
                    label=f"Latest: {latest['Gold_oz_per_Bitcoin']:.1f} oz")

    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Bitcoin Price (oz Gold)', fontweight='bold')
    ax.set_title("Bitcoin's Gold Price vs Saturating Exponential Model",
                 fontweight='bold')
    ax.set_xlim(pd.to_datetime('2015-01-01'), pd.to_datetime('2036-01-01'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:g}'))
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# PLOT 2: CUSUM SCORECARD
# ============================================================================

def create_cusum_plot(oos_df, output_path):
    """CUSUM chart with green/yellow/red zones and boundary lines."""
    plt.rcParams.update({
        'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
        'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12,
    })
    fig, ax = plt.subplots(figsize=(10, 7))

    n_points = len(oos_df)
    if n_points == 0:
        ax.text(0.5, 0.5, 'No out-of-sample data yet',
                transform=ax.transAxes, ha='center', fontsize=16)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        return

    dates = oos_df['Date'].values
    z_scores = oos_df['z_score'].values
    cusum = oos_df['cusum'].values

    # Boundaries: extend forward 5 years
    n_extend = n_points + 60
    ns_boundary = np.arange(1, n_extend + 1)
    boundary_iid = C_IID * np.sqrt(ns_boundary)
    boundary_corr = C_CORRECTED * np.sqrt(ns_boundary)
    boundary_traj = C_TRAJECTORY * np.sqrt(ns_boundary)

    dates_boundary = pd.date_range(
        start=oos_df['Date'].iloc[0], periods=n_extend, freq='MS')

    y_lim = boundary_traj[n_extend - 1] + 5
    ax.set_ylim(-y_lim, y_lim)

    # Colored zones
    ax.fill_between(dates_boundary, boundary_traj, y_lim,
                    color='#E74C3C', alpha=0.10, zorder=0)
    ax.fill_between(dates_boundary, -boundary_traj, -y_lim,
                    color='#E74C3C', alpha=0.10, zorder=0)
    ax.fill_between(dates_boundary, boundary_corr, boundary_traj,
                    color='#F39C12', alpha=0.10, zorder=1)
    ax.fill_between(dates_boundary, -boundary_corr, -boundary_traj,
                    color='#F39C12', alpha=0.10, zorder=1)
    ax.fill_between(dates_boundary, -boundary_corr, boundary_corr,
                    color='#27AE60', alpha=0.10, zorder=2)

    # Boundary lines
    ax.plot(dates_boundary, boundary_iid, '--', color='gray', linewidth=1.5,
            alpha=0.5, label='iid boundary (reference)', zorder=5)
    ax.plot(dates_boundary, -boundary_iid, '--', color='gray', linewidth=1.5,
            alpha=0.5, zorder=5)
    ax.plot(dates_boundary, boundary_corr, '--', color='#8E44AD', linewidth=2.0,
            alpha=0.7,
            label=f'Reduced volatility (\u03c3={SIGMA_POST2023}, c={C_CORRECTED})',
            zorder=4)
    ax.plot(dates_boundary, -boundary_corr, '--', color='#8E44AD', linewidth=2.0,
            alpha=0.7, zorder=4)
    ax.plot(dates_boundary, boundary_traj, '--', color='#C0392B', linewidth=2.0,
            alpha=0.7,
            label=f'Trajectory (\u03c3={SIGMA_FULL}, c={C_TRAJECTORY:.1f})',
            zorder=4)
    ax.plot(dates_boundary, -boundary_traj, '--', color='#C0392B', linewidth=2.0,
            alpha=0.7, zorder=4)

    ax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5, zorder=3)

    # CUSUM line
    ax.plot(dates, cusum, 'o-', color='#2E86AB', linewidth=2.5,
            markersize=6, label='Cumulative z-score ($S_n$)',
            zorder=10, markeredgecolor='white', markeredgewidth=0.5)

    # Color points by z-score sign
    for i in range(n_points):
        color = '#27AE60' if z_scores[i] >= 0 else '#8B4513'
        ax.plot(dates[i], cusum[i], 'o', color=color, markersize=8,
                zorder=11, markeredgecolor='white', markeredgewidth=1.0)

    # Summary stats box
    current_S = cusum[-1]
    margin_corr = boundary_corr[n_points - 1] - abs(current_S)
    margin_traj = boundary_traj[n_points - 1] - abs(current_S)
    mean_z = z_scores.mean()
    std_z = z_scores.std(ddof=1) if n_points > 1 else 0

    textstr = (
        f'Out-of-sample months: {n_points}\n'
        f'Mean z-score: {mean_z:+.3f}\n'
        f'Cumulative sum: {current_S:+.2f}\n'
        f'\n'
        f'Reduced vol. margin: {margin_corr:+.1f}\n'
        f'Trajectory margin:   {margin_traj:+.1f}\n'
        f'\n'
        f'\u03c3 post-2023: {SIGMA_POST2023}\n'
        f'\u03c3 full-sample: {SIGMA_FULL}\n'
        f'\u03c1 = {RHO_TRAINING} (lag-1 autocorr.)'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right', bbox=props,
            fontfamily='monospace', zorder=100)

    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Cumulative Z-Score Sum ($S_n$)', fontweight='bold')
    ax.set_title('Sequential Test of Trajectory Hypothesis\n'
                 'CUSUM of Out-of-Sample Residuals', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# HTML GENERATION
# ============================================================================

def generate_html(oos_df, daily_df, output_path):
    """Generate a minimal, clean dashboard page."""

    # Compute stats for the table
    if not oos_df.empty:
        n_months = len(oos_df)
        current_S = oos_df['cusum'].iloc[-1]
        latest_z = oos_df['z_score'].iloc[-1]
        margin_corr = C_CORRECTED * np.sqrt(n_months) - abs(current_S)
        margin_traj = C_TRAJECTORY * np.sqrt(n_months) - abs(current_S)

        if abs(current_S) < C_CORRECTED * np.sqrt(n_months):
            status = "Green"
            status_desc = "Both hypotheses supported"
        elif abs(current_S) < C_TRAJECTORY * np.sqrt(n_months):
            status = "Yellow"
            status_desc = "Reduced volatility rejected, trajectory intact"
        else:
            status = "Red"
            status_desc = "Both hypotheses rejected"
    else:
        n_months = 0
        current_S = latest_z = margin_corr = margin_traj = 0
        status = "N/A"
        status_desc = "No out-of-sample data"

    # Latest price data
    if not daily_df.empty:
        latest = daily_df.iloc[-1]
        latest_ratio = f"{latest['Gold_oz_per_Bitcoin']:.1f}"
        latest_gold = f"${latest['USD_per_Gold_oz']:,.0f}"
        latest_btc = f"${latest['USD_per_Bitcoin']:,.0f}"
        latest_date = pd.to_datetime(latest['Date']).strftime('%Y-%m-%d')
    else:
        latest_ratio = latest_gold = latest_btc = "N/A"
        latest_date = "N/A"

    # Optional commentary
    commentary_file = SITE / 'commentary.txt'
    commentary = ""
    if commentary_file.exists():
        text = commentary_file.read_text().strip()
        if text:
            commentary = f'<div class="commentary"><strong>Commentary:</strong> {text}</div>'

    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    status_colors = {"Green": "#27AE60", "Yellow": "#F39C12", "Red": "#E74C3C"}
    status_color = status_colors.get(status, "#666")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bitcoin's Gold Price</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
         line-height: 1.5; max-width: 800px; margin: 0 auto; padding: 16px;
         color: #333; background: #fafafa; font-size: 15px; }}
  h1 {{ font-size: 1.5em; margin-bottom: 0.2em; }}
  h2 {{ font-size: 1.15em; color: #555; margin-top: 1.5em; }}
  .section-divider {{ border: none; border-top: 2px solid #ddd; margin: 2em 0 1.2em; }}
  table {{ border-collapse: collapse; margin: 0.8em 0; width: 100%; }}
  th, td {{ padding: 5px 10px; text-align: left; border-bottom: 1px solid #ddd; }}
  th {{ border-top: 2px solid #333; font-weight: 600; font-size: 0.9em; }}
  .price-table td {{ font-size: 0.95em; }}
  .status {{ font-weight: bold; color: {status_color}; }}
  img {{ max-width: 100%; height: auto; margin: 0.8em 0; border: 1px solid #eee; }}
  .commentary {{ background: #f0f0f0; padding: 10px 14px; border-left: 3px solid #2E86AB;
                 margin: 1em 0; font-size: 0.9em; }}
  .footer {{ margin-top: 2em; padding-top: 1em; border-top: 1px solid #ddd;
             font-size: 0.8em; color: #888; }}
  a {{ color: #0066cc; }}
  p {{ font-size: 0.93em; }}
  details {{ font-size: 0.93em; }}
  @media (max-width: 600px) {{
    body {{ padding: 10px; font-size: 14px; }}
    h1 {{ font-size: 1.3em; }}
    th, td {{ padding: 4px 6px; font-size: 0.85em; }}
  }}
</style>
</head>
<body>

<h1>Bitcoin's Gold Price</h1>

<img src="dashboard_ratio.png" alt="Bitcoin priced in ounces of gold">

<table class="price-table">
  <tr><th>Gold (USD/oz)</th><th>Bitcoin (USD)</th><th>Bitcoin's Gold Price</th><th>Date</th></tr>
  <tr><td>{latest_gold}</td><td>{latest_btc}</td><td>{latest_ratio} oz</td><td>{latest_date}</td></tr>
</table>

{commentary}

<hr class="section-divider">

<h2>About the Trendline</h2>
<p>The orange curve and shaded bands above are from
  <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5110528"><em>Bitcoin's Gold Price: History, Model, and Falsifiable Predictions through 2035</em></a>
  (Biggs, 2026), which fits a saturating exponential model to the Bitcoin/Gold ratio.
  The model predicts Bitcoin will plateau near 40&ndash;45 oz of Gold by the mid-2030s.
  The bands widen before 2023 (volatile early era) and tighten after (reduced volatility regime).
  The chart below tracks whether the prediction is holding up.</p>

<h2>Live Model Scorecard</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Out-of-sample months</td><td>{n_months}</td></tr>
  <tr><td>Latest monthly z-score</td><td>{latest_z:+.2f}</td></tr>
  <tr><td>Cumulative sum (S<sub>n</sub>)</td><td>{current_S:+.2f}</td></tr>
  <tr><td>Reduced volatility margin</td><td>{margin_corr:+.1f}</td></tr>
  <tr><td>Trajectory margin</td><td>{margin_traj:+.1f}</td></tr>
  <tr><td>Status</td><td class="status">{status} &mdash; {status_desc}</td></tr>
</table>

<img src="dashboard_cusum.png" alt="CUSUM scorecard with rejection boundaries">

<details>
<summary><strong>Zone definitions</strong></summary>
<table>
  <tr><th>Zone</th><th>Meaning</th></tr>
  <tr><td style="color:#27AE60;font-weight:bold;">Green</td>
      <td>Both trajectory and reduced volatility hypotheses supported</td></tr>
  <tr><td style="color:#F39C12;font-weight:bold;">Yellow</td>
      <td>Reduced volatility hypothesis rejected; trajectory intact</td></tr>
  <tr><td style="color:#E74C3C;font-weight:bold;">Red</td>
      <td>Both hypotheses rejected</td></tr>
</table>
<p style="font-size:0.9em; color:#666;">
  Each month, the model's prediction error is standardized and added to a running total.
  If that total drifts outside a boundary, the hypothesis is rejected at &alpha;&nbsp;=&nbsp;0.05.
  Boundaries are calibrated via Monte Carlo (2M simulations, 120 monthly looks) and account
  for the observed boom-bust autocorrelation (&rho;&nbsp;=&nbsp;0.90).
  See Section&nbsp;10 of the paper for full methodology.</p>
</details>

<div class="footer">
  <p>Last updated: {now}</p>
  <p>Data: BTC-USD and Gold futures (closing prices) via Yahoo Finance.</p>
  <p><a href="bitcoin_gold_biggs.pdf">Paper (PDF)</a> |
     <a href="bitcoin_gold_Biggs_(20260127).html">Paper (HTML)</a> |
     <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5110528">SSRN</a> |
     <a href="https://github.com/silmonbiggs/BTCvGold">Source code</a></p>
</div>

</body>
</html>"""

    Path(output_path).write_text(html, encoding='utf-8')
    print(f"  Saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("Dashboard Update")
    print("=" * 60)

    # Ensure output dirs exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fetch daily prices
    print("\n1. Fetching daily prices...")
    daily_df = fetch_prices()

    # 2. Load monthly data and compute CUSUM
    print("\n2. Computing CUSUM from monthly data...")
    monthly_df = load_monthly_data()
    oos_df = compute_oos_cusum(monthly_df)
    if not oos_df.empty:
        print(f"  {len(oos_df)} out-of-sample months, S_n = {oos_df['cusum'].iloc[-1]:+.2f}")

    # 3. Generate plots
    print("\n3. Generating plots...")
    create_ratio_plot(daily_df, SITE / 'dashboard_ratio.png')
    create_cusum_plot(oos_df, SITE / 'dashboard_cusum.png')

    # 4. Generate HTML
    print("\n4. Generating HTML...")
    generate_html(oos_df, daily_df, SITE / 'index.html')

    print("\nDone.")


if __name__ == '__main__':
    main()
