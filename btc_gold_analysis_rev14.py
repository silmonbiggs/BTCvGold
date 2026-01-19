"""
Bitcoin's Gold Price - History, Model, and Falsifiable Predictions through 2035
Complete analysis script: data loading, model fitting, visualization, and statistics

Author: S. James Biggs
Date: October 2025
Revision: 14

Acknowledgment: The author thanks Anthropic's Claude Code (Opus 4.5) for contributions
to coding, data analysis, and presentation of this work.

Data Sources:
- Gold: World Gold Council via GitHub (https://github.com/datasets/gold-prices)
         + Kitco/Gold.org for Aug 2025 - Jan 2026
- Bitcoin: CoinGecko historical data

Model: Saturating exponential
  ln(R(t)) = C + g*t + A(1 - e^(-λt))
  
Where R(t) is the BTC/Gold ratio, g is fixed at 0.02 (gold's supply leak rate)

UPDATED: 
- Single-axis USD prices plot
- Consistent orange color for model lines
- Improved y-axis formatting
- 150% larger fonts throughout
- Rev7: Standardized fonts across all figures, improved text box formatting
- Rev8: X-axis shows every year in tight_projections figure
- Rev9: Added ±1σ error bars to projection points in trailing_average figure
- Rev10: Added ±1σ and ±2σ error bars to tight_projections figure
- Rev11: Text box moved to front (on top of error bars) in tight_projections
- Rev12: Updated title in trailing_average figure
- Rev13: Changed all ratio terminology to "Bitcoin's Gold Price"
- Rev14: Added R², p-value, and slope output to Rolling Volatility Analysis linear trend; Combined 2015-2019 and 2020-2022 periods into single 2015-2022 period in residuals analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from datetime import datetime
import sys
from matplotlib.ticker import FuncFormatter

# Set larger default font sizes (150% of original)
plt.rcParams.update({
    'font.size': 15,           # was 10
    'axes.labelsize': 18,       # was 12
    'axes.titlesize': 21,       # was 14
    'xtick.labelsize': 15,      # was 10
    'ytick.labelsize': 15,      # was 10
    'legend.fontsize': 15,      # was 10
})

# ============================================================================
# CONFIGURATION - CHANGE THESE PATHS TO YOUR DATA DIRECTORY
# ============================================================================

DATA_PATH = './'  # Directory containing CSV files
TRAINING_FILE = 'btc_gold_training_2015_2024.csv'
TEST_FILE = 'btc_gold_test_2025(&26Jan).csv'
OUTPUT_PATH = './figures/'  # Where to save figures and results

# Model parameters
GOLD_SUPPLY_LEAK_RATE = 0.02  # Fixed: gold's 2% annual supply growth

# ============================================================================
# MODEL DEFINITION
# ============================================================================

def saturating_exponential(t, C, A, lambda_param, g):
    """
    Saturating exponential model for log(BTC/Gold ratio)
    
    ln(R(t)) = C + g*t + A(1 - e^(-λt))
    
    Parameters:
        t: time in years from start date
        C: baseline log ratio
        A: early adoption acceleration amplitude
        lambda_param (λ): decay rate of adoption acceleration
        g: long-run differential growth rate (gold's supply leak)
    
    Returns:
        log of BTC/Gold ratio
    """
    return C + g * t + A * (1 - np.exp(-lambda_param * t))


def fit_model(df, g_fixed=0.02):
    """
    Fit the saturating exponential model to training data
    
    Args:
        df: DataFrame with columns ['Date', 'Gold_oz_per_Bitcoin']
        g_fixed: fixed long-term growth rate
    
    Returns:
        popt: fitted parameters [C, A, lambda]
        pcov: covariance matrix
        t_years: time array in years
        fit_stats: dictionary of fit statistics
    """
    # Convert dates to years from start
    dates = pd.to_datetime(df['Date'])
    start_date = dates.iloc[0]
    t_years = np.array([(d - start_date).days / 365.25 for d in dates])
    
    # Take log of ratio
    ratio = df['Gold_oz_per_Bitcoin'].values
    log_ratio = np.log(ratio)
    
    # Define model with g fixed
    def model_fixed_g(t, C, A, lambda_param):
        return saturating_exponential(t, C, A, lambda_param, g_fixed)
    
    # Initial parameter guesses
    p0 = [log_ratio[0], 2.0, 0.3]
    
    # Fit the model
    popt, pcov = curve_fit(model_fixed_g, t_years, log_ratio, p0=p0, maxfev=10000)
    
    # Calculate fit statistics
    log_ratio_fitted = saturating_exponential(t_years, popt[0], popt[1], popt[2], g_fixed)
    residuals = log_ratio - log_ratio_fitted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_ratio - np.mean(log_ratio))**2)
    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean(residuals**2))
    
    fit_stats = {
        'r_squared': r_squared,
        'rmse': rmse,
        'n_points': len(t_years),
        'residuals': residuals
    }
    
    return popt, pcov, t_years, fit_stats


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def add_year_labels(ax):
    """Helper to add year labels to x-axis"""
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

def create_figure_usd_prices(df_train, df_test, output_path):
    """
    USD prices for Bitcoin and Gold (single y-axis, semilogy)
    Modified to show both assets on the same scale for better comparison
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Combine all data - no train/test distinction
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    dates_all = pd.to_datetime(df_all['Date'])
    
    # Plot both on the same axis
    ax.semilogy(dates_all, df_all['USD_per_Bitcoin'], 
                'o-', color='#2E86AB', linewidth=2, markersize=3, 
                label='Bitcoin', alpha=0.8)
    ax.semilogy(dates_all, df_all['USD_per_Gold_oz'], 
                '^-', color='#F18F01', linewidth=2, markersize=3, 
                label='Gold', alpha=0.8)
    
    # Labels and title
    ax.set_xlabel('Year')
    ax.set_ylabel('Dollars per Bitcoin or Ounce of Gold')
    ax.set_title('Bitcoin and Gold USD Prices\n2015-2026', fontsize=21, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    
    # Year labels
    add_year_labels(ax)
    
    # Legend
    ax.legend(loc='upper left')
    
    # Set y-axis limits to capture both series nicely
    ax.set_ylim(100, 200000)  # $100 to $200k range should capture both
    
    # Format y-axis with friendly dollar labels
    def format_dollars(value, tick_number):
        if value >= 1000:
            return f'${value/1000:.0f}k' if value < 1000000 else f'${value/1000000:.1f}M'
        else:
            return f'${value:.0f}'
    
    ax.yaxis.set_major_formatter(FuncFormatter(format_dollars))
    
    # Set specific tick locations for cleaner display
    ax.set_yticks([100, 1000, 10000, 100000])
    ax.set_yticklabels(['$100', '$1,000', '$10,000', '$100,000'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def create_figure_ratio_history(df_train, df_test, output_path):
    """
    Figure 1: Bitcoin's Gold Price over time (training data only, semilogy)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    dates_train = pd.to_datetime(df_train['Date'])
    
    # Training data only
    ax.semilogy(dates_train, df_train['Gold_oz_per_Bitcoin'], 
                'o-', color='#2E86AB', linewidth=2, markersize=4, 
                label='Historical Data (2015-2024)', alpha=0.8)
    
    ax.set_xlabel('Date', fontsize=18, fontweight='bold')
    ax.set_ylabel('Bitcoin Price (oz of Gold)', fontsize=18, fontweight='bold')
    ax.set_title("Bitcoin's Gold Price\n2015-2024", 
                 fontsize=21, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.legend(fontsize=15, loc='upper left')
    
    # Format y-axis with regular numbers instead of exponential
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}' if y >= 1 else f'{y:.1f}'))
    
    # Set x-axis to show every year
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def create_figure_model_comparison(df_train, popt, g_fixed, output_path):
    """
    Figure 2: Compare saturating exponential vs alternatives
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    dates = pd.to_datetime(df_train['Date'])
    start_date = dates.iloc[0]
    t_years = np.array([(d - start_date).days / 365.25 for d in dates])
    log_ratio = np.log(df_train['Gold_oz_per_Bitcoin'].values)
    
    # Actual data
    ax.plot(dates, log_ratio, 'o', color='black', markersize=5, 
            label='Observed Data', alpha=0.7, zorder=5)
    
    # Saturating exponential (our model)
    C, A, lambda_param = popt
    log_ratio_saturating = saturating_exponential(t_years, C, A, lambda_param, g_fixed)
    ax.plot(dates, log_ratio_saturating, '-', color='#F18F01', linewidth=3,
            label='Saturating Exponential (This Model)', zorder=4)
    
    # Alternative 1: Flat (like gold - no growth)
    log_ratio_flat = np.full_like(t_years, log_ratio.mean())
    ax.plot(dates, log_ratio_flat, '--', color='#888888', linewidth=2,
            label='Flat Model ("Just Like Gold")', alpha=0.7)
    
    # Alternative 2: Linear growth (perpetual exponential)
    z = np.polyfit(t_years, log_ratio, 1)
    log_ratio_linear = z[0] * t_years + z[1]
    ax.plot(dates, log_ratio_linear, '-.', color='#666666', linewidth=2,
            label='Linear Model ("More Forever")', alpha=0.7)
    
    # Calculate and display residuals
    residual_saturating = np.sum((log_ratio - log_ratio_saturating)**2)
    residual_flat = np.sum((log_ratio - log_ratio_flat)**2)
    residual_linear = np.sum((log_ratio - log_ratio_linear)**2)
    
    ax.set_xlabel('Date', fontsize=18, fontweight='bold')
    ax.set_ylabel('ln(Bitcoin\'s Gold Price)', fontsize=18, fontweight='bold')
    ax.set_title("Bitcoin's Gold Price\n2015-01 to 2024-12", 
                 fontsize=21, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=15, loc='upper left')
    
    # Set x-axis to show every year with rotation
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def create_figure_model_fit(df_train, df_test, popt, g_fixed, output_path):
    """
    Figure 3: Bitcoin's Gold Price with fitted model and confidence bounds (training data only, semilogy)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data
    dates_train = pd.to_datetime(df_train['Date'])
    start_date = dates_train.iloc[0]
    
    # Training data - actual ratio values
    t_train = np.array([(d - start_date).days / 365.25 for d in dates_train])
    ratio_train = df_train['Gold_oz_per_Bitcoin'].values
    
    # Calculate residuals in log space to get sigma
    C, A, lambda_param = popt
    log_ratio_train_predicted = saturating_exponential(t_train, C, A, lambda_param, g_fixed)
    log_ratio_train_actual = np.log(ratio_train)
    log_residuals_train = log_ratio_train_actual - log_ratio_train_predicted
    sigma_log = np.std(log_residuals_train)
    
    # Plot training data (semilogy)
    ax.semilogy(dates_train, ratio_train, 'o', color='#2E86AB', 
                markersize=5, label='Training Data', alpha=0.7)
    
    # Fitted model on training period - convert from log space to actual values
    log_ratio_fit_train = saturating_exponential(t_train, C, A, lambda_param, g_fixed)
    ratio_fit_train = np.exp(log_ratio_fit_train)
    ax.semilogy(dates_train, ratio_fit_train, '-', color='#F18F01', 
                linewidth=2.5, label=f'Fitted Model (g={g_fixed:.1%})', alpha=0.9)
    
    # Add 1σ confidence bounds
    log_ratio_1sigma_upper = log_ratio_fit_train + sigma_log
    log_ratio_1sigma_lower = log_ratio_fit_train - sigma_log
    ratio_1sigma_upper = np.exp(log_ratio_1sigma_upper)
    ratio_1sigma_lower = np.exp(log_ratio_1sigma_lower)
    ax.semilogy(dates_train, ratio_1sigma_upper, '--', color='gray', linewidth=1.5, 
                alpha=0.6, label='±1σ')
    ax.semilogy(dates_train, ratio_1sigma_lower, '--', color='gray', linewidth=1.5, alpha=0.6)
    
    # Add 2σ confidence bounds
    log_ratio_2sigma_upper = log_ratio_fit_train + 2*sigma_log
    log_ratio_2sigma_lower = log_ratio_fit_train - 2*sigma_log
    ratio_2sigma_upper = np.exp(log_ratio_2sigma_upper)
    ratio_2sigma_lower = np.exp(log_ratio_2sigma_lower)
    ax.semilogy(dates_train, ratio_2sigma_upper, '--', color='lightgray', linewidth=1.5, 
                alpha=0.5, label='±2σ')
    ax.semilogy(dates_train, ratio_2sigma_lower, '--', color='lightgray', linewidth=1.5, alpha=0.5)
    
    ax.set_xlabel('Date', fontsize=18, fontweight='bold')
    ax.set_ylabel('Bitcoin Price (oz of Gold)', fontsize=18, fontweight='bold')
    ax.set_title("Bitcoin's Gold Price with Saturating Exponential Fit\n" + 
                 f'Model: ln(R) = C + {g_fixed:.1%}·t + A(1 - e^(-λt))',
                 fontsize=21, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.legend(fontsize=15, loc='upper left')
    
    # Set x-axis limit to 2026
    ax.set_xlim(dates_train.iloc[0], pd.to_datetime('2026-01'))
    
    # Set y-axis limits to accommodate confidence bands
    ax.set_ylim(0.1, 100)
    
    # Format y-axis to show readable numbers
    from matplotlib.ticker import FuncFormatter
    def format_func(value, tick_number):
        if value == 0.1:
            return '0.1'
        elif value == 1:
            return '1'
        elif value == 10:
            return '10'
        elif value == 100:
            return '100'
        else:
            return f'{value:.1f}'
    ax.yaxis.set_major_formatter(FuncFormatter(format_func))
    
    # Set x-axis to show every year with rotation
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def create_figure_projection(df_train, df_test, popt_dict, output_path):
    """
    Figure 4: Model extrapolation to test data with confidence bounds (semilogy)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    dates_train = pd.to_datetime(df_train['Date'])
    dates_test = pd.to_datetime(df_test['Date'])
    start_date = dates_train.iloc[0]
    
    # Calculate sigma from training data residuals (using primary model g=0.020)
    popt_primary, g_primary = popt_dict['0.020']
    C, A, lambda_param = popt_primary
    t_train = np.array([(d - start_date).days / 365.25 for d in dates_train])
    log_ratio_train_predicted = saturating_exponential(t_train, C, A, lambda_param, g_primary)
    log_ratio_train_actual = np.log(df_train['Gold_oz_per_Bitcoin'].values)
    log_residuals_train = log_ratio_train_actual - log_ratio_train_predicted
    sigma_log = np.std(log_residuals_train)
    
    # Historical data
    ax.semilogy(dates_train, df_train['Gold_oz_per_Bitcoin'], 
                'o-', color='#2E86AB', linewidth=2, markersize=4, 
                label='Training Data', alpha=0.8)
    ax.semilogy(dates_test, df_test['Gold_oz_per_Bitcoin'],
                's-', color='#A23B72', linewidth=2, markersize=6,
                label='Test Data (2025-2026)', alpha=0.8)
    
    # Model projection for primary model only (g=0.020)
    t_current = (dates_test.iloc[-1] - start_date).days / 365.25
    t_projection = np.linspace(0, t_current, 300)
    dates_projection = [start_date + pd.Timedelta(days=365.25*t) for t in t_projection]
    
    # Store 2030 and 2035 values for return (calculate for all models even if not plotting)
    projections_2030 = {}
    projections_2035 = {}
    
    # Plot only primary model (g=0.020)
    log_ratio_projection = saturating_exponential(t_projection, C, A, lambda_param, g_primary)
    ratio_projection = np.exp(log_ratio_projection)
    
    ax.semilogy(dates_projection, ratio_projection, 
                '-', color='#F18F01', linewidth=2.5, 
                label=f'Model (g={g_primary:.1%})', alpha=0.9)
    
    # Add confidence bounds for primary model
    # 1σ bands
    log_ratio_1sigma_upper = log_ratio_projection + sigma_log
    log_ratio_1sigma_lower = log_ratio_projection - sigma_log
    ratio_1sigma_upper = np.exp(log_ratio_1sigma_upper)
    ratio_1sigma_lower = np.exp(log_ratio_1sigma_lower)
    ax.semilogy(dates_projection, ratio_1sigma_upper, '--', color='gray', 
                linewidth=1.5, alpha=0.6, label='±1σ')
    ax.semilogy(dates_projection, ratio_1sigma_lower, '--', color='gray', 
                linewidth=1.5, alpha=0.6)
    
    # 2σ bands
    log_ratio_2sigma_upper = log_ratio_projection + 2*sigma_log
    log_ratio_2sigma_lower = log_ratio_projection - 2*sigma_log
    ratio_2sigma_upper = np.exp(log_ratio_2sigma_upper)
    ratio_2sigma_lower = np.exp(log_ratio_2sigma_lower)
    ax.semilogy(dates_projection, ratio_2sigma_upper, '--', color='lightgray', 
                linewidth=1.5, alpha=0.5, label='±2σ')
    ax.semilogy(dates_projection, ratio_2sigma_lower, '--', color='lightgray', 
                linewidth=1.5, alpha=0.5)
    
    # Calculate 2030 and 2035 projections for all models (for consistency with other figures)
    for g_str, (popt, g_val) in popt_dict.items():
        C_g, A_g, lambda_g = popt
        t_2030 = (pd.to_datetime('2030-01') - start_date).days / 365.25
        t_2035 = (pd.to_datetime('2035-01') - start_date).days / 365.25
        
        ratio_2030 = np.exp(saturating_exponential(t_2030, C_g, A_g, lambda_g, g_val))
        ratio_2035 = np.exp(saturating_exponential(t_2035, C_g, A_g, lambda_g, g_val))
        
        projections_2030[g_str] = ratio_2030
        projections_2035[g_str] = ratio_2035
    
    ax.set_xlabel('Date', fontsize=18, fontweight='bold')
    ax.set_ylabel('Bitcoin Price (oz of Gold)', fontsize=18, fontweight='bold')
    
    # Build title
    title = "Bitcoin's Gold Price: Model Extrapolation to Test Data"
    
    ax.set_title(title, fontsize=21, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.legend(fontsize=15, loc='upper left')
    
    # Set x-axis from 2014 to 2027 (to show full test data)
    ax.set_xlim(pd.to_datetime('2014-01'), pd.to_datetime('2027-01'))
    
    # Set y-axis to accommodate confidence bands
    ax.set_ylim(0.1, 100)
    
    # Format y-axis with readable numbers
    from matplotlib.ticker import FuncFormatter
    def format_func(value, tick_number):
        if value == 0.1:
            return '0.1'
        elif value == 1:
            return '1'
        elif value == 10:
            return '10'
        elif value == 100:
            return '100'
        else:
            return f'{value:.1f}'
    ax.yaxis.set_major_formatter(FuncFormatter(format_func))
    
    # Set x-axis to show every year with rotation
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    return projections_2030, projections_2035


def create_figure_trailing_average(df_train, df_test, popt_dict, output_path):
    """
    Figure 5: Bitcoin's Gold Price with model predictions and confidence bands (semilogy)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Combine train and test
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    dates_all = pd.to_datetime(df_all['Date'])
    ratio_all = df_all['Gold_oz_per_Bitcoin'].values
    
    # Calculate residuals from TRAINING data using primary model (g=0.020)
    dates_train = pd.to_datetime(df_train['Date'])
    dates_test = pd.to_datetime(df_test['Date'])
    start_date = dates_train.iloc[0]
    
    # Get primary model parameters
    popt_primary, g_primary = popt_dict['0.020']
    C, A, lambda_param = popt_primary
    
    # Calculate predictions for TRAINING data to get sigma IN LOG SPACE
    t_train = np.array([(d - start_date).days / 365.25 for d in dates_train])
    log_ratio_train_predicted = saturating_exponential(t_train, C, A, lambda_param, g_primary)
    log_ratio_train_actual = np.log(df_train['Gold_oz_per_Bitcoin'].values)
    
    # Calculate residuals in LOG SPACE (where model was fit)
    log_residuals_train = log_ratio_train_actual - log_ratio_train_predicted
    sigma_log = np.std(log_residuals_train)
    
    # Now check if TEST data falls within these bands
    t_test = np.array([(d - start_date).days / 365.25 for d in dates_test])
    log_ratio_test_predicted = saturating_exponential(t_test, C, A, lambda_param, g_primary)
    log_ratio_test_actual = np.log(df_test['Gold_oz_per_Bitcoin'].values)
    log_residuals_test = log_ratio_test_actual - log_ratio_test_predicted
    
    # Check how many test points fall within 1σ and 2σ (in log space)
    within_1sigma = np.sum(np.abs(log_residuals_test) <= sigma_log)
    within_2sigma = np.sum(np.abs(log_residuals_test) <= 2*sigma_log)
    
    print(f"\nConfidence Band Statistics:")
    print(f"  Training sigma (log space): {sigma_log:.4f}")
    print(f"  Approx multiplicative factor at 1-sigma: x{np.exp(sigma_log):.2f} or /{np.exp(-sigma_log):.2f}")
    print(f"  Test points within 1-sigma: {within_1sigma}/{len(log_residuals_test)} ({within_1sigma/len(log_residuals_test)*100:.0f}%)")
    print(f"  Test points within 2-sigma: {within_2sigma}/{len(log_residuals_test)} ({within_2sigma/len(log_residuals_test)*100:.0f}%)")
    print(f"  Expected: ~68% within 1-sigma, ~95% within 2-sigma")
    
    # Plot historical data (semilogy)
    ax.semilogy(dates_all, ratio_all, ':', color='#666666', linewidth=1.5, 
                label='Monthly Ratio', alpha=0.8)
    
    # Add projection lines extending to 2040
    colors = {'0.015': '#F18F01', '0.020': '#006BA6', '0.025': '#C73E1D'}
    
    t_current = (dates_test.iloc[-1] - start_date).days / 365.25
    t_projection = np.linspace(0, t_current + 15, 300)  # Extend to 2040
    dates_projection = [start_date + pd.Timedelta(days=365.25*t) for t in t_projection]
    
    # Calculate 2030 and 2035 dates (January)
    date_2030 = pd.to_datetime('2030-01')
    date_2035 = pd.to_datetime('2035-01')
    t_2030 = (date_2030 - start_date).days / 365.25
    t_2035 = (date_2035 - start_date).days / 365.25
    
    # Store projections for text box
    projections_2030 = {}
    projections_2035 = {}
    
    for g_str, (popt, g_val) in popt_dict.items():
        C_g, A_g, lambda_g = popt
        log_ratio_projection = saturating_exponential(t_projection, C_g, A_g, lambda_g, g_val)
        ratio_projection = np.exp(log_ratio_projection)
        
        # Plot model lines as solid
        ax.semilogy(dates_projection, ratio_projection, 
                    '-', color=colors[g_str], linewidth=2.5, 
                    label=f'Model g={g_val:.1%}', alpha=0.9)
        
        # Calculate and plot 2030 and 2035 projection points
        ratio_2030 = np.exp(saturating_exponential(t_2030, C_g, A_g, lambda_g, g_val))
        ratio_2035 = np.exp(saturating_exponential(t_2035, C_g, A_g, lambda_g, g_val))
        
        # Store for text box
        projections_2030[g_str] = ratio_2030
        projections_2035[g_str] = ratio_2035
        
        # Calculate ±1σ error bars in ratio space
        # In log space: error is ±sigma_log
        # In ratio space: upper = ratio * exp(sigma_log), lower = ratio * exp(-sigma_log)
        error_2030_lower = ratio_2030 * (1 - np.exp(-sigma_log))
        error_2030_upper = ratio_2030 * (np.exp(sigma_log) - 1)
        error_2035_lower = ratio_2035 * (1 - np.exp(-sigma_log))
        error_2035_upper = ratio_2035 * (np.exp(sigma_log) - 1)
        
        # Stagger x-positions slightly to avoid overlap
        # g=0.015: -1 month, g=0.020: 0 months, g=0.025: +1 month
        offsets = {'0.015': -1, '0.020': 0, '0.025': 1}
        date_2030_offset = date_2030 + pd.DateOffset(months=offsets[g_str])
        date_2035_offset = date_2035 + pd.DateOffset(months=offsets[g_str])
        
        # Plot the projection points with error bars
        ax.errorbar([date_2030_offset, date_2035_offset], 
                    [ratio_2030, ratio_2035],
                    yerr=[[error_2030_lower, error_2035_lower], 
                          [error_2030_upper, error_2035_upper]],
                    fmt='o', color=colors[g_str], markersize=8, 
                    capsize=5, capthick=2, elinewidth=2,
                    zorder=10, markeredgecolor='black', markeredgewidth=1.5)
    
    # Add confidence bands for primary model (g=0.020)
    # Calculate log predictions first
    log_ratio_projection_primary = saturating_exponential(t_projection, C, A, lambda_param, g_primary)
    
    # Apply confidence bands in LOG SPACE (where residuals were calculated)
    # 1σ bands
    log_ratio_1sigma_upper = log_ratio_projection_primary + sigma_log
    log_ratio_1sigma_lower = log_ratio_projection_primary - sigma_log
    # Transform to linear space for plotting
    ratio_1sigma_upper = np.exp(log_ratio_1sigma_upper)
    ratio_1sigma_lower = np.exp(log_ratio_1sigma_lower)
    ax.semilogy(dates_projection, ratio_1sigma_upper, '--', color='gray', linewidth=1.5, 
                alpha=0.6, label='±1σ')
    ax.semilogy(dates_projection, ratio_1sigma_lower, '--', color='gray', linewidth=1.5, alpha=0.6)
    
    # 2σ bands
    log_ratio_2sigma_upper = log_ratio_projection_primary + 2*sigma_log
    log_ratio_2sigma_lower = log_ratio_projection_primary - 2*sigma_log
    # Transform to linear space for plotting
    ratio_2sigma_upper = np.exp(log_ratio_2sigma_upper)
    ratio_2sigma_lower = np.exp(log_ratio_2sigma_lower)
    ax.semilogy(dates_projection, ratio_2sigma_upper, '--', color='lightgray', linewidth=1.5, 
                alpha=0.5, label='±2σ')
    ax.semilogy(dates_projection, ratio_2sigma_lower, '--', color='lightgray', linewidth=1.5, alpha=0.5)
    
    ax.set_xlabel('Date', fontsize=18, fontweight='bold')
    ax.set_ylabel('Bitcoin Price (oz of Gold)', fontsize=18, fontweight='bold')
    ax.set_title("Bitcoin's Gold Price - Monthly Data, and Long Range Model Predictions", 
                 fontsize=21, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.legend(fontsize=15, loc='lower left')
    
    # Add text box with projections ± 1σ in lower right
    # Note: sigma_log is in log space, so we need to transform properly
    textstr = f'Projections (1σ = ×{np.exp(sigma_log):.2f} or /{np.exp(-sigma_log):.2f}):\n'
    for g in ['0.015', '0.020', '0.025']:
        val_2030 = projections_2030[g]
        val_2035 = projections_2035[g]
        # Calculate upper and lower bounds in log space, then transform
        upper_2030 = val_2030 * np.exp(sigma_log)
        lower_2030 = val_2030 * np.exp(-sigma_log)
        upper_2035 = val_2035 * np.exp(sigma_log)
        lower_2035 = val_2035 * np.exp(-sigma_log)
        textstr += f"g={g}:\n"
        textstr += f"  2030: {val_2030:.1f} ({lower_2030:.1f}-{upper_2030:.1f}) oz\n"
        textstr += f"  2035: {val_2035:.1f} ({lower_2035:.1f}-{upper_2035:.1f}) oz\n"
    
    # Create text box properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=15,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    # Set x-axis to 2020-2040
    ax.set_xlim(pd.to_datetime('2020-01'), pd.to_datetime('2040-01'))
    
    # Set y-axis to 1-100 (10^0 to 10^2)
    ax.set_ylim(1, 100)
    
    # Format y-axis to show 1, 10, 100 instead of scientific notation
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))
    
    # Set x-axis to show every year with rotation
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()




def analyze_residuals(df_train, df_test, popt, g_fixed, output_path):
    """
    Analyze residuals to explore changing volatility over time
    Uses 36-month rolling window for clear long-term trends
    
    Returns: Dictionary of statistics for reporting
    """
    from scipy import stats as scipy_stats
    from scipy.stats import linregress, levene
    
    # Combine all data
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    dates_all = pd.to_datetime(df_all['Date'])
    ratio_all = df_all['Gold_oz_per_Bitcoin'].values
    
    # Convert dates to years from start
    start_date = dates_all.iloc[0]
    t_years = np.array([(d - start_date).days / 365.25 for d in dates_all])
    
    # Calculate predictions and residuals
    C, A, lambda_param = popt
    log_ratio_predicted = saturating_exponential(t_years, C, A, lambda_param, g_fixed)
    log_ratio_actual = np.log(ratio_all)
    residuals = log_ratio_actual - log_ratio_predicted
    
    # Set rolling window
    window = 36  # 36-month window for clearer long-term trends
    
    # Calculate rolling statistics
    abs_residuals = np.abs(residuals)
    rolling_std = pd.Series(residuals).rolling(window=window, center=True).std()
    rolling_abs_mean = pd.Series(abs_residuals).rolling(window=window, center=True).mean()
    
    # Period cutoffs - combining pre-2023 into one period
    cutoff_2023 = pd.to_datetime('2023-01-01')
    idx_2023 = np.where(dates_all >= cutoff_2023)[0][0]
    
    # Calculate volatility by period
    period1_residuals = residuals[:idx_2023]  # 2015-2022
    period2_residuals = residuals[idx_2023:]  # 2023-present
    
    vol_2015_2022 = np.std(period1_residuals)
    vol_2023_2025 = np.std(period2_residuals)
    
    # Test for trend in absolute residuals
    slope, intercept, r_value, p_value, std_err = linregress(t_years, abs_residuals)
    
    # Calculate percentage within 1 sigma for each period
    overall_std = np.std(residuals[:len(df_train)])  # Use training std
    pct_within_1sig_2015_2022 = np.mean(np.abs(period1_residuals) <= overall_std) * 100
    pct_within_1sig_2023_2025 = np.mean(np.abs(period2_residuals) <= overall_std) * 100
    
    # Peak and recent volatility from rolling window
    peak_vol = np.nanmax(rolling_std)
    # Get the last valid value for recent volatility
    last_valid_idx = rolling_std.last_valid_index()
    if last_valid_idx is not None:
        recent_vol = rolling_std.iloc[last_valid_idx]
        vol_reduction = (1 - recent_vol/peak_vol) * 100
    else:
        recent_vol = vol_2023_2025  # Fall back to period volatility
        vol_reduction = (1 - recent_vol/peak_vol) * 100
    
    # Create residuals figure 1 (Qualitative)
    create_figure_residuals_qualitative(
        dates_all, residuals, rolling_abs_mean, window,
        output_path + 'figure_residuals_qualitative.png'
    )
    
    # Create residuals figure 2 (Quantitative) - returns Levene test results, rolling volatility R-squared, p-value, and slope
    levene_stat, levene_df1, levene_df2, levene_p, sigma_2023_present, rolling_volatility_r_squared, rolling_volatility_p_value, rolling_volatility_slope = create_figure_residuals_quantitative(
        dates_all, residuals, rolling_std, cutoff_2023, overall_std, window,
        output_path + 'figure_residuals_quantitative.png'
    )
    
    # Create Figure: Projections with tighter post-2023 volatility bounds
    create_figure_tight_projections(
        pd.concat([df_train, df_test]).reset_index(drop=True).iloc[:len(df_train)],  # Training data
        df_test,
        popt,
        g_fixed,
        sigma_2023_present,
        output_path + 'figure_tight_projections.png'
    )
    
    # Note: This figure has been revised to show tighter bounds based on post-2023 volatility
    
    # Return statistics dictionary with new additions
    return {
        'vol_2015_2022': vol_2015_2022,
        'vol_2023_2025': vol_2023_2025,
        'sigma_2023_present': sigma_2023_present,
        'levene_stat': levene_stat,
        'levene_df1': levene_df1,
        'levene_df2': levene_df2,
        'levene_p': levene_p,
        'trend_slope': slope,
        'trend_pvalue': p_value,
        'rolling_volatility_r_squared': rolling_volatility_r_squared,
        'rolling_volatility_p_value': rolling_volatility_p_value,
        'rolling_volatility_slope': rolling_volatility_slope,
        'peak_vol': peak_vol,
        'recent_vol': recent_vol,
        'vol_reduction': vol_reduction,
        'pct_within_1sig_2015_2022': pct_within_1sig_2015_2022,
        'pct_within_1sig_2023_2025': pct_within_1sig_2023_2025
    }


def create_figure_residuals_qualitative(dates_all, residuals, rolling_abs_mean, window, output_path):
    """
    Create qualitative residuals analysis figure (Percentage and Absolute residuals)
    """
    fig = plt.figure(figsize=(16, 8))
    
    # Color by time for all scatter plots - REVERSED colormap (yellow early, purple late)
    t_normalized = np.linspace(0, 1, len(dates_all))
    
    # Panel 1: Percentage residuals (LEFT)
    ax1 = plt.subplot(1, 2, 1)
    pct_residuals = (np.exp(residuals) - 1) * 100
    ax1.scatter(dates_all, pct_residuals, c=t_normalized, cmap='viridis_r', alpha=0.6, s=30)
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('% Deviation', fontsize=18)
    ax1.set_xlabel('Date', fontsize=18)
    ax1.set_title('Percentage Residuals', fontsize=21)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Absolute residuals with rolling mean (RIGHT)
    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(dates_all, np.abs(residuals), c=t_normalized, cmap='viridis_r', alpha=0.6, s=30)
    ax2.plot(dates_all, rolling_abs_mean, color='red', linewidth=2.5, 
             label=f'{window}-month rolling mean')
    ax2.set_ylabel('|Log Residuals|', fontsize=18)
    ax2.set_xlabel('Date', fontsize=18)
    ax2.set_title('Absolute Residuals (Volatility Proxy)', fontsize=21)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=15)
    
    # Format x-axes for both panels
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Master title
    plt.suptitle('Residuals, Qualitative', fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def create_figure_residuals_quantitative(dates_all, residuals, rolling_std, 
                                        cutoff_2023, overall_std, window, output_path):
    """
    Create quantitative residuals analysis figure (Rolling volatility and Test for reduced volatility)
    Returns Levene test results and 2023-present sigma
    """
    from scipy.stats import levene
    
    fig = plt.figure(figsize=(16, 8))
    
    # Color by time for all scatter plots - REVERSED colormap (yellow early, purple late)
    t_normalized = np.linspace(0, 1, len(dates_all))
    
    # Calculate Levene test for pre-2023 vs 2023-present
    idx_2023 = np.where(dates_all >= cutoff_2023)[0][0]
    pre_2023_residuals = residuals[:idx_2023]
    post_2023_residuals = residuals[idx_2023:]
    
    # Levene test
    levene_stat, levene_p = levene(pre_2023_residuals, post_2023_residuals)
    
    # Calculate degrees of freedom for Levene test
    # df1 = k - 1 (number of groups - 1), df2 = N - k (total observations - number of groups)
    levene_df1 = 2 - 1  # Two groups (pre-2023 and post-2023)
    levene_df2 = len(pre_2023_residuals) + len(post_2023_residuals) - 2
    
    # Calculate sigma for 2023-present
    sigma_2023_present = np.std(post_2023_residuals)
    
    # Panel 1: Rolling standard deviation as LINE PLOT (LEFT - MOVED FROM RIGHT)
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(dates_all, rolling_std, color='darkblue', linewidth=2.5,
             label=f'{window}-month rolling σ')

    # Add trend line (only where rolling_std is not NaN)
    mask = ~np.isnan(rolling_std)
    rolling_volatility_r_squared = None  # Initialize
    rolling_volatility_p_value = None  # Initialize
    rolling_volatility_slope = None  # Initialize
    if np.sum(mask) > 2:
        t_years = np.array([(d - dates_all[0]).days / 365.25 for d in dates_all])
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(t_years[mask], rolling_std[mask])
        rolling_volatility_r_squared = r_value ** 2  # Calculate R-squared
        rolling_volatility_p_value = p_value  # Store p-value
        rolling_volatility_slope = slope  # Store slope
        trend_line = slope * t_years + intercept
        ax1.plot(dates_all, trend_line, 'r--', linewidth=2, alpha=0.7,
                label=f'Trend: {slope:.4f}/year')

    # Find peak and annotate it
    peak_idx = np.nanargmax(rolling_std)
    peak_date = dates_all[peak_idx]
    peak_value = rolling_std.iloc[peak_idx]
    ax1.annotate(f'Peak: {peak_value:.3f}\n({peak_date.strftime("%Y-%m")})',
                xy=(peak_date, peak_value),
                xytext=(peak_date, peak_value + 0.1),
                ha='center',
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

    # Add annotation for the last valid point FROM BELOW
    last_valid_idx = rolling_std.last_valid_index()
    if last_valid_idx is not None:
        last_date = dates_all[last_valid_idx]
        last_value = rolling_std.iloc[last_valid_idx]

        # Calculate reduction percentage
        reduction = (1 - last_value/peak_value) * 100

        # Annotate from below with upward arrow
        ax1.annotate(f'Recent: {last_value:.3f}\n({reduction:.0f}% reduction)',
                    xy=(last_date, last_value),
                    xytext=(last_date, last_value - 0.15),
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))

    ax1.set_ylabel('Rolling Std Dev', fontsize=18)
    ax1.set_xlabel('Date', fontsize=18)
    ax1.set_title(f'{window}-Month Rolling Volatility', fontsize=21)
    ax1.set_ylim(0.0, 0.8)  # Set y-axis range as requested
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=15)

    # Panel 2: Raw residuals over time with pre/post 2023 shading (RIGHT)
    ax2 = plt.subplot(1, 2, 2)
    
    # Add pre/post 2023 shading FIRST (so it's in background)
    ax2.axvspan(dates_all[0], cutoff_2023, alpha=0.1, color='gray', label='Pre-2023')
    ax2.axvspan(cutoff_2023, dates_all.iloc[-1], alpha=0.1, color='lightgreen', label='2023-Present')
    
    # Then add the scatter plot with reversed colormap
    scatter1 = ax2.scatter(dates_all, residuals, c=t_normalized, cmap='viridis_r', alpha=0.6, s=30)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
    # Add training sigma lines
    ax2.axhline(overall_std, color='red', linestyle='--', alpha=0.5, label=f'±1σ (training: {overall_std:.3f})')
    ax2.axhline(-overall_std, color='red', linestyle='--', alpha=0.5)
    
    # Add 2023-present sigma bounds (dotted lines only in 2023-present region)
    dates_2023_present = dates_all[idx_2023:]
    sigma_upper = [sigma_2023_present] * len(dates_2023_present)
    sigma_lower = [-sigma_2023_present] * len(dates_2023_present)
    ax2.plot(dates_2023_present, sigma_upper, ':', color='green', linewidth=1.5, 
             label=f'±1σ (2023-present: {sigma_2023_present:.3f})')
    ax2.plot(dates_2023_present, sigma_lower, ':', color='green', linewidth=1.5)
    
    ax2.set_ylabel('Log Residuals', fontsize=18)
    ax2.set_xlabel('Date', fontsize=18)
    ax2.set_title('Test for Reduced Volatility', fontsize=21)  # CHANGED TITLE
    ax2.set_ylim(-1, 2.0)  # Set y-axis range to prevent data hiding under legend
    ax2.grid(True, alpha=0.3)
    
    # Reorder legend as requested
    handles, labels = ax2.get_legend_handles_labels()
    order = [0, 1, 2, 3]  # Pre-2023, 2023-Present, training sigma, 2023 sigma
    ax2.legend([handles[idx] for idx in order], [labels[idx] for idx in order], 
               loc='upper right', fontsize=15)
    
    # Format x-axes for both panels
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Master title
    plt.suptitle('Residuals, Quantitative', fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    # Return Levene test results (including F-statistic and df), 2023-present sigma, rolling volatility R-squared, p-value, and slope
    return levene_stat, levene_df1, levene_df2, levene_p, sigma_2023_present, rolling_volatility_r_squared, rolling_volatility_p_value, rolling_volatility_slope

def create_figure_tight_projections(df_train, df_test, popt, g_fixed, sigma_2023, output_path):
    """
    Figure 7: Projections with tighter bounds based on post-2023 volatility
    Shows falsifiable predictions for both trajectory and volatility hypotheses
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data
    dates_train = pd.to_datetime(df_train['Date'])
    dates_test = pd.to_datetime(df_test['Date'])
    start_date = dates_train.iloc[0]
    
    # Training and test data
    dates_all = pd.concat([dates_train, dates_test])
    ratio_all = pd.concat([df_train['Gold_oz_per_Bitcoin'], df_test['Gold_oz_per_Bitcoin']])
    
    # Create projection dates (2020 to 2040)
    date_range_start = pd.to_datetime('2020-01')
    date_range_end = pd.to_datetime('2040-01')
    dates_projection = pd.date_range(start=date_range_start, end=date_range_end, freq='M')
    t_projection = np.array([(d - start_date).days / 365.25 for d in dates_projection])
    
    # Model parameters
    C, A, lambda_param = popt
    
    # Calculate projection
    log_ratio_projection = saturating_exponential(t_projection, C, A, lambda_param, g_fixed)
    ratio_projection = np.exp(log_ratio_projection)
    
    # Plot actual data (only where we have it)
    mask_actual = dates_all >= date_range_start
    ax.semilogy(dates_all[mask_actual], ratio_all[mask_actual], 
                'o--', color='#2E86AB', markersize=4, linewidth=1,
                label='Actual Data', alpha=0.6)
    
    # Plot model projection
    ax.semilogy(dates_projection, ratio_projection, 
                '-', color='#F18F01', linewidth=2.5, 
                label=f'Model (g={g_fixed:.1%})', alpha=0.9)
    
    # Find index for 2023 start
    idx_2023 = np.where(dates_projection >= pd.to_datetime('2023-01'))[0][0]
    
    # Add confidence bounds ONLY from 2023 forward
    dates_from_2023 = dates_projection[idx_2023:]
    log_ratio_from_2023 = log_ratio_projection[idx_2023:]
    
    # 1σ bands (post-2023 volatility)
    log_ratio_1sigma_upper = log_ratio_from_2023 + sigma_2023
    log_ratio_1sigma_lower = log_ratio_from_2023 - sigma_2023
    ratio_1sigma_upper = np.exp(log_ratio_1sigma_upper)
    ratio_1sigma_lower = np.exp(log_ratio_1sigma_lower)
    
    ax.semilogy(dates_from_2023, ratio_1sigma_upper, '--', color='green', 
                linewidth=1.5, alpha=0.7, label=f'±1σ (post-2023: {sigma_2023:.3f})')
    ax.semilogy(dates_from_2023, ratio_1sigma_lower, '--', color='green', 
                linewidth=1.5, alpha=0.7)
    
    # 2σ bands (post-2023 volatility)
    log_ratio_2sigma_upper = log_ratio_from_2023 + 2*sigma_2023
    log_ratio_2sigma_lower = log_ratio_from_2023 - 2*sigma_2023
    ratio_2sigma_upper = np.exp(log_ratio_2sigma_upper)
    ratio_2sigma_lower = np.exp(log_ratio_2sigma_lower)
    
    ax.semilogy(dates_from_2023, ratio_2sigma_upper, ':', color='darkgreen', 
                linewidth=1.5, alpha=0.5, label=f'±2σ (post-2023: {2*sigma_2023:.3f})')
    ax.semilogy(dates_from_2023, ratio_2sigma_lower, ':', color='darkgreen', 
                linewidth=1.5, alpha=0.5)
    
    # Add shading to show the post-2023 regime
    ax.axvspan(pd.to_datetime('2023-01'), date_range_end, alpha=0.05, color='green',
               label='Post-2023 Volatility Regime')
    
    # Calculate specific projections for 2030 and 2035
    t_2030 = (pd.to_datetime('2030-10') - start_date).days / 365.25
    t_2035 = (pd.to_datetime('2035-10') - start_date).days / 365.25
    
    log_ratio_2030 = saturating_exponential(t_2030, C, A, lambda_param, g_fixed)
    log_ratio_2035 = saturating_exponential(t_2035, C, A, lambda_param, g_fixed)
    
    ratio_2030 = np.exp(log_ratio_2030)
    ratio_2035 = np.exp(log_ratio_2035)
    
    # Calculate bounds for 2030 and 2035
    ratio_2030_1sig_lower = np.exp(log_ratio_2030 - sigma_2023)
    ratio_2030_1sig_upper = np.exp(log_ratio_2030 + sigma_2023)
    ratio_2030_2sig_lower = np.exp(log_ratio_2030 - 2*sigma_2023)
    ratio_2030_2sig_upper = np.exp(log_ratio_2030 + 2*sigma_2023)
    
    ratio_2035_1sig_lower = np.exp(log_ratio_2035 - sigma_2023)
    ratio_2035_1sig_upper = np.exp(log_ratio_2035 + sigma_2023)
    ratio_2035_2sig_lower = np.exp(log_ratio_2035 - 2*sigma_2023)
    ratio_2035_2sig_upper = np.exp(log_ratio_2035 + 2*sigma_2023)
    
    # Mark the projection points with error bars
    # Calculate error amounts (asymmetric in ratio space because model works in log space)
    error_2030_1sig_lower = ratio_2030 - ratio_2030_1sig_lower
    error_2030_1sig_upper = ratio_2030_1sig_upper - ratio_2030
    error_2030_2sig_lower = ratio_2030 - ratio_2030_2sig_lower
    error_2030_2sig_upper = ratio_2030_2sig_upper - ratio_2030
    
    error_2035_1sig_lower = ratio_2035 - ratio_2035_1sig_lower
    error_2035_1sig_upper = ratio_2035_1sig_upper - ratio_2035
    error_2035_2sig_lower = ratio_2035 - ratio_2035_2sig_lower
    error_2035_2sig_upper = ratio_2035_2sig_upper - ratio_2035
    
    # Stagger x-positions slightly: 1σ at -1 month, 2σ at +1 month
    date_2030_1sig = pd.to_datetime('2030-10') - pd.DateOffset(months=1)
    date_2030_2sig = pd.to_datetime('2030-10') + pd.DateOffset(months=1)
    date_2035_1sig = pd.to_datetime('2035-10') - pd.DateOffset(months=1)
    date_2035_2sig = pd.to_datetime('2035-10') + pd.DateOffset(months=1)
    
    # Plot 1σ error bars
    ax.errorbar([date_2030_1sig, date_2035_1sig], 
                [ratio_2030, ratio_2035],
                yerr=[[error_2030_1sig_lower, error_2035_1sig_lower], 
                      [error_2030_1sig_upper, error_2035_1sig_upper]],
                fmt='o', color='black', markersize=8, 
                markerfacecolor='white', markeredgewidth=2,
                capsize=5, capthick=2, elinewidth=2, ecolor='green',
                zorder=10, label='Projection ±1σ')
    
    # Plot 2σ error bars
    ax.errorbar([date_2030_2sig, date_2035_2sig], 
                [ratio_2030, ratio_2035],
                yerr=[[error_2030_2sig_lower, error_2035_2sig_lower], 
                      [error_2030_2sig_upper, error_2035_2sig_upper]],
                fmt='s', color='black', markersize=8, 
                markerfacecolor='white', markeredgewidth=2,
                capsize=5, capthick=2, elinewidth=2, ecolor='darkgreen',
                zorder=10, label='Projection ±2σ')
    
    # Create text for predictions box
    textstr = (
        f'Falsifiable Predictions (g={g_fixed:.1%}):\n'
        f'─────────────────────────\n'
        f'2030-10: {ratio_2030:.1f} oz\n'
        f'  68% CI: [{ratio_2030_1sig_lower:.1f}, {ratio_2030_1sig_upper:.1f}]\n'
        f'  95% CI: [{ratio_2030_2sig_lower:.1f}, {ratio_2030_2sig_upper:.1f}]\n'
        f'\n'
        f'2035-10: {ratio_2035:.1f} oz\n'
        f'  68% CI: [{ratio_2035_1sig_lower:.1f}, {ratio_2035_1sig_upper:.1f}]\n'
        f'  95% CI: [{ratio_2035_2sig_lower:.1f}, {ratio_2035_2sig_upper:.1f}]\n'
        f'\n'
        f'Rejection criteria:\n'
        f'If 2035 price outside [{ratio_2035_2sig_lower:.0f}, {ratio_2035_2sig_upper:.0f}] oz,\n'
        f'reject stability hypothesis'
    )
    
    # Create text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=15,
            verticalalignment='bottom', horizontalalignment='right', bbox=props,
            fontfamily='monospace', zorder=100)
    
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Bitcoin Price (oz of Gold)', fontsize=18)
    ax.set_title("Bitcoin's Gold Price Projections with Post-2023 Volatility Bounds\n" + 
                 'Testing Both Trajectory and Stability Hypotheses', fontsize=21)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.legend(loc='upper left', fontsize=15)
    
    # Set axis limits
    ax.set_xlim(date_range_start, date_range_end)
    ax.set_ylim(10, 100)
    
    # Format x-axis - show every year
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Format y-axis ticks
    ax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """
    Main analysis pipeline
    """
    print("\n" + "="*70)
    print("Bitcoin and Gold: Monetary Saturation Analysis")
    print("="*70 + "\n")
    
    # Open results file
    results_file = OUTPUT_PATH + 'analysis_results.txt'
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("Bitcoin and Gold: Monetary Saturation Analysis\n")
        f.write("="*70 + "\n")
        f.write(f"Analysis run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print("STEP 1: Loading Data")
    print("-" * 70)
    
    try:
        df_train = pd.read_csv(DATA_PATH + TRAINING_FILE)
        df_test = pd.read_csv(DATA_PATH + TEST_FILE)
        print(f"  Training data: {len(df_train)} months ({df_train['Date'].iloc[0]} to {df_train['Date'].iloc[-1]})")
        print(f"  Test data: {len(df_test)} months ({df_test['Date'].iloc[0]} to {df_test['Date'].iloc[-1]})")
    except FileNotFoundError as e:
        print(f"ERROR: Could not find data files in {DATA_PATH}")
        print(f"  Looking for: {TRAINING_FILE} and {TEST_FILE}")
        print(f"  {e}")
        sys.exit(1)
    
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write("DATA\n")
        f.write("-" * 70 + "\n")
        f.write(f"Training: {len(df_train)} months ({df_train['Date'].iloc[0]} to {df_train['Date'].iloc[-1]})\n")
        f.write(f"Test: {len(df_test)} months ({df_test['Date'].iloc[0]} to {df_test['Date'].iloc[-1]})\n\n")
    
    # ========================================================================
    # STEP 2: Fit Model
    # ========================================================================
    print("\nSTEP 2: Fitting Models")
    print("-" * 70)
    
    # Fit saturating exponential for g = 0.015, 0.020, 0.025
    g_values = [0.015, 0.020, 0.025]
    popt_dict = {}
    fit_stats_dict = {}
    
    for g_val in g_values:
        popt, pcov, t_years_train, fit_stats = fit_model(df_train, g_val)
        popt_dict[f'{g_val:.3f}'] = (popt, g_val)
        fit_stats_dict[f'{g_val:.3f}'] = fit_stats
    
    # Use g=0.020 as primary model
    popt_primary, pcov, t_years_train, fit_stats_primary = fit_model(df_train, GOLD_SUPPLY_LEAK_RATE)
    C, A, lambda_param = popt_primary
    
    print(f"\nSaturating Exponential Model: ln(R(t)) = C + g*t + A(1 - e^(-lambda*t))")
    print(f"\nPrimary Model (g = {GOLD_SUPPLY_LEAK_RATE:.3f}):")
    print(f"  C (baseline log ratio):    {C:.4f}")
    print(f"  A (adoption amplitude):    {A:.4f}")
    print(f"  lambda (decay rate):       {lambda_param:.4f}")
    print(f"  R^2 = {fit_stats_primary['r_squared']:.4f}")
    print(f"  RMSE = {fit_stats_primary['rmse']:.4f}")
    
    # Also fit flat and linear models for comparison
    dates_train = pd.to_datetime(df_train['Date'])
    start_date = dates_train.iloc[0]
    t_years = np.array([(d - start_date).days / 365.25 for d in dates_train])
    log_ratio = np.log(df_train['Gold_oz_per_Bitcoin'].values)
    
    # Flat model
    log_ratio_flat = np.full_like(t_years, log_ratio.mean())
    ss_res_flat = np.sum((log_ratio - log_ratio_flat)**2)
    ss_tot = np.sum((log_ratio - np.mean(log_ratio))**2)
    r2_flat = 1 - (ss_res_flat / ss_tot)
    
    # Linear model
    z = np.polyfit(t_years, log_ratio, 1)
    log_ratio_linear = z[0] * t_years + z[1]
    ss_res_linear = np.sum((log_ratio - log_ratio_linear)**2)
    r2_linear = 1 - (ss_res_linear / ss_tot)
    
    print(f"\nAlternative Models:")
    print(f"  Flat Model:   R^2 = {r2_flat:.4f}")
    print(f"  Linear Model: R^2 = {r2_linear:.4f}, slope = {z[0]:.4f}, intercept = {z[1]:.4f}")
    
    print(f"\nSaturating Exponential with Different g Values:")
    for g_str in ['0.015', '0.020', '0.025']:
        popt_g, g_val = popt_dict[g_str]
        stats_g = fit_stats_dict[g_str]
        C_g, A_g, lambda_g = popt_g
        print(f"  g = {g_val:.3f}: C={C_g:.4f}, A={A_g:.4f}, lambda={lambda_g:.4f}, R^2={stats_g['r_squared']:.4f}")
    
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write("MODELS\n")
        f.write("-" * 70 + "\n")
        f.write("Saturating Exponential:\n")
        f.write("  ln(R(t)) = C + g·t + A(1 - e^(-λt))\n\n")
        f.write(f"Primary Model (g = {GOLD_SUPPLY_LEAK_RATE:.3f}):\n")
        f.write(f"  C = {C:.6f}\n")
        f.write(f"  A = {A:.6f}\n")
        f.write(f"  λ = {lambda_param:.6f}\n")
        f.write(f"  R² = {fit_stats_primary['r_squared']:.6f}\n")
        f.write(f"  RMSE = {fit_stats_primary['rmse']:.6f}\n\n")
        f.write("Alternative Models:\n")
        f.write(f"  Flat Model:   R² = {r2_flat:.6f}\n")
        f.write(f"  Linear Model: R² = {r2_linear:.6f}\n")
        f.write(f"    slope = {z[0]:.6f}, intercept = {z[1]:.6f}\n\n")
        f.write("Saturating Exponential with Different g Values:\n")
        for g_str in ['0.015', '0.020', '0.025']:
            popt_g, g_val = popt_dict[g_str]
            stats_g = fit_stats_dict[g_str]
            C_g, A_g, lambda_g = popt_g
            f.write(f"  g = {g_val:.3f}:\n")
            f.write(f"    C = {C_g:.6f}, A = {A_g:.6f}, λ = {lambda_g:.6f}\n")
            f.write(f"    R² = {stats_g['r_squared']:.6f}, RMSE = {stats_g['rmse']:.6f}\n")
        f.write("\n")
    
    # ========================================================================
    # STEP 3: Test Model on 2025 Data
    # ========================================================================
    print("\nSTEP 3: Testing Model on 2025 Data")
    print("-" * 70)
    
    # Calculate predictions for test data (using primary model g=0.020)
    dates_test = pd.to_datetime(df_test['Date'])
    start_date = pd.to_datetime(df_train['Date'].iloc[0])
    t_test = np.array([(d - start_date).days / 365.25 for d in dates_test])
    
    log_ratio_test_actual = np.log(df_test['Gold_oz_per_Bitcoin'].values)
    log_ratio_test_predicted = saturating_exponential(t_test, C, A, lambda_param, GOLD_SUPPLY_LEAK_RATE)
    ratio_test_predicted = np.exp(log_ratio_test_predicted)
    
    # Calculate test error
    test_residuals = log_ratio_test_actual - log_ratio_test_predicted
    test_rmse = np.sqrt(np.mean(test_residuals**2))
    test_mae = np.mean(np.abs(ratio_test_predicted - df_test['Gold_oz_per_Bitcoin'].values))
    
    print(f"\nTest Set Performance:")
    print(f"  RMSE (log scale): {test_rmse:.4f}")
    print(f"  MAE (oz gold):    {test_mae:.2f}")
    print(f"\nActual vs Predicted (2025):")
    print(f"  {'Date':<10} {'Actual':<10} {'Predicted':<10} {'Error':<10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write("TEST SET PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"RMSE (log scale): {test_rmse:.6f}\n")
        f.write(f"MAE (oz gold): {test_mae:.4f}\n\n")
        f.write(f"{'Date':<12} {'Actual':>10} {'Predicted':>10} {'Error':>10}\n")
        f.write(f"{'-'*12} {'-'*10} {'-'*10} {'-'*10}\n")
    
    for i in range(len(df_test)):
        date = df_test['Date'].iloc[i]
        actual = df_test['Gold_oz_per_Bitcoin'].iloc[i]
        predicted = ratio_test_predicted[i]
        error = actual - predicted
        print(f"  {date:<10} {actual:<10.2f} {predicted:<10.2f} {error:+10.2f}")
        
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write(f"{date:<12} {actual:>10.2f} {predicted:>10.2f} {error:>+10.2f}\n")
    
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write("\n")
    
    # ========================================================================
    # STEP 4: Generate Projections
    # ========================================================================
    print("\nSTEP 4: Generating Projections")
    print("-" * 70)
    
    # Current state
    current_ratio = df_test['Gold_oz_per_Bitcoin'].iloc[-1]
    current_date = df_test['Date'].iloc[-1]
    
    # 2030 and 2035 projections for different g values
    date_2030 = pd.to_datetime('2030-10')
    date_2035 = pd.to_datetime('2035-10')
    t_2030 = (date_2030 - start_date).days / 365.25
    t_2035 = (date_2035 - start_date).days / 365.25
    
    print(f"\nCurrent State ({current_date}):")
    print(f"  Bitcoin's Gold Price: {current_ratio:.2f} oz")
    
    print(f"\nProjections for Different g Values:")
    print(f"  {'g':<6} {'2030':<10} {'2035':<10} {'CAGR':<10} {'Pass?'}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write("PROJECTIONS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Current State ({current_date}):\n")
        f.write(f"  Bitcoin's Gold Price: {current_ratio:.4f} oz\n\n")
        f.write("Projections for Different g Values:\n")
        f.write(f"  {'g':<8} {'2030 (oz)':<12} {'2035 (oz)':<12} {'CAGR':<10} {'Valid?'}\n")
        f.write(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*10}\n")
    
    for g_str in ['0.015', '0.020', '0.025']:
        popt_g, g_val = popt_dict[g_str]
        C_g, A_g, lambda_g = popt_g
        
        ratio_2030 = np.exp(saturating_exponential(t_2030, C_g, A_g, lambda_g, g_val))
        ratio_2035 = np.exp(saturating_exponential(t_2035, C_g, A_g, lambda_g, g_val))
        
        years_to_2035 = (date_2035 - pd.to_datetime(current_date)).days / 365.25
        cagr = (ratio_2035 / current_ratio) ** (1/years_to_2035) - 1
        passes = 'PASS' if ratio_2035 > 32 else 'FAIL'
        
        print(f"  {g_val:<6.3f} {ratio_2030:<10.2f} {ratio_2035:<10.2f} {cagr:<10.2%} {passes}")
        
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write(f"  {g_val:<8.3f} {ratio_2030:<12.4f} {ratio_2035:<12.4f} {cagr:<10.4%} {passes}\n")
    
    # Detailed output for primary model (g=0.020)
    ratio_2035_primary = np.exp(saturating_exponential(t_2035, C, A, lambda_param, GOLD_SUPPLY_LEAK_RATE))
    years_to_2035 = (date_2035 - pd.to_datetime(current_date)).days / 365.25
    cagr_vs_gold = (ratio_2035_primary / current_ratio) ** (1/years_to_2035) - 1
    
    # Expected returns
    gold_real_drift = 0.02
    cpi_inflation = 0.025
    total_nominal_return = (1 + cpi_inflation) * (1 + gold_real_drift) * (1 + cagr_vs_gold) - 1
    
    print(f"\nExpected Annual Returns (next decade, g={GOLD_SUPPLY_LEAK_RATE:.3f}):")
    print(f"  BTC vs Gold:          {cagr_vs_gold:.2%}")
    print(f"  Gold real drift:      {gold_real_drift:.2%}")
    print(f"  CPI inflation:        {cpi_inflation:.2%}")
    print(f"  Total nominal return: {total_nominal_return:.2%}")
    
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write(f"\nExpected Annual Returns (next decade, primary model g={GOLD_SUPPLY_LEAK_RATE:.3f}):\n")
        f.write(f"  BTC vs Gold:          {cagr_vs_gold:.4%}\n")
        f.write(f"  Gold real drift:      {gold_real_drift:.4%}\n")
        f.write(f"  CPI inflation:        {cpi_inflation:.4%}\n")
        f.write(f"  Total nominal return: {total_nominal_return:.4%}\n\n")
    
    # ========================================================================
    # STEP 5: Generate Figures
    # ========================================================================
    print("\nSTEP 5: Generating Figures")
    print("-" * 70)
    
    create_figure_usd_prices(
        df_train, df_test,
        OUTPUT_PATH + 'figure_usd_prices.png'
    )
    
    create_figure_ratio_history(
        df_train, df_test, 
        OUTPUT_PATH + 'figure_ratio_history.png'
    )
    
    create_figure_model_comparison(
        df_train, popt_primary, GOLD_SUPPLY_LEAK_RATE,
        OUTPUT_PATH + 'figure_model_comparison.png'
    )
    
    create_figure_model_fit(
        df_train, df_test, popt_primary, GOLD_SUPPLY_LEAK_RATE,
        OUTPUT_PATH + 'figure_model_fit.png'
    )
    
    projections_2030, projections_2035 = create_figure_projection(
        df_train, df_test, popt_dict,
        OUTPUT_PATH + 'figure_projection.png'
    )
    
    create_figure_trailing_average(
        df_train, df_test, popt_dict,
        OUTPUT_PATH + 'figure_trailing_average.png'
    )
    
    # ========================================================================
    # STEP 6: Residuals Analysis - Exploring Volatility Changes
    # ========================================================================
    print("\nSTEP 6: Residuals Analysis")
    print("-" * 70)
    
    # Perform residuals analysis
    residuals_stats = analyze_residuals(
        df_train, df_test, popt_primary, GOLD_SUPPLY_LEAK_RATE,
        OUTPUT_PATH
    )
    
    # Write residuals analysis to results file
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write("\nRESIDUALS ANALYSIS (36-MONTH ROLLING WINDOW)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Volatility by Period (std of log residuals):\n")
        f.write(f"  2015-2022: {residuals_stats['vol_2015_2022']:.4f}\n")
        f.write(f"  2023-present: {residuals_stats['vol_2023_2025']:.4f}\n\n")
        f.write(f"2023-Present Analysis:\n")
        f.write(f"  Sigma (2023-present): ±{residuals_stats['sigma_2023_present']:.4f}\n")
        f.write(f"  Levene test: F({residuals_stats['levene_df1']},{residuals_stats['levene_df2']}) = {residuals_stats['levene_stat']:.2f}, p = {residuals_stats['levene_p']:.4f}\n")
        if residuals_stats['levene_p'] < 0.05:
            f.write(f"  Interpretation: Variances are significantly different\n\n")
        else:
            f.write(f"  Interpretation: No significant difference in variances\n\n")
        f.write(f"Trend in Absolute Residuals:\n")
        f.write(f"  Slope: {residuals_stats['trend_slope']:.6f} per year\n")
        f.write(f"  P-value: {residuals_stats['trend_pvalue']:.4f}\n")
        if residuals_stats['trend_pvalue'] < 0.05:
            direction = "decreasing" if residuals_stats['trend_slope'] < 0 else "increasing"
            f.write(f"  Interpretation: Statistically significant {direction} trend\n\n")
        else:
            f.write(f"  Interpretation: No statistically significant trend\n\n")
        f.write(f"Rolling Volatility Analysis (36-month window):\n")
        f.write(f"  Peak volatility: {residuals_stats['peak_vol']:.4f}\n")
        f.write(f"  Recent volatility: {residuals_stats['recent_vol']:.4f}\n")
        f.write(f"  Reduction from peak: {residuals_stats['vol_reduction']:.1f}%\n")
        if residuals_stats['rolling_volatility_r_squared'] is not None:
            f.write(f"  Linear trend R²: {residuals_stats['rolling_volatility_r_squared']:.4f}\n")
            if residuals_stats['rolling_volatility_p_value'] is not None:
                f.write(f"  Linear trend p-value: {residuals_stats['rolling_volatility_p_value']:.4f}\n")
            if residuals_stats['rolling_volatility_slope'] is not None:
                f.write(f"  Linear trend slope: {residuals_stats['rolling_volatility_slope']:.6f} per year\n\n")
            else:
                f.write(f"\n")
        else:
            f.write(f"  Linear trend R²: Not calculated\n\n")
        f.write(f"Observations within ±1σ by period:\n")
        f.write(f"  2015-2022: {residuals_stats['pct_within_1sig_2015_2022']:.1f}%\n")
        f.write(f"  2023-present: {residuals_stats['pct_within_1sig_2023_2025']:.1f}%\n\n")
    
    print(f"\nVolatility Analysis Complete:")
    print(f"  Peak rolling volatility (36m): {residuals_stats['peak_vol']:.4f}")
    print(f"  Recent rolling volatility (36m): {residuals_stats['recent_vol']:.4f}")
    print(f"  Reduction from peak: {residuals_stats['vol_reduction']:.1f}%")
    print(f"  2023-present observations within +/-1 sigma: {residuals_stats['pct_within_1sig_2023_2025']:.1f}%")
    print(f"  Levene test: F({residuals_stats['levene_df1']},{residuals_stats['levene_df2']}) = {residuals_stats['levene_stat']:.2f}, p = {residuals_stats['levene_p']:.4f}")
    
    # ========================================================================
    # STEP 7: Generating Falsifiable Predictions with Post-2023 Volatility
    # ========================================================================
    print("\nSTEP 7: Generating Falsifiable Predictions with Post-2023 Volatility")
    print("-" * 70)
    
    # Calculate the specific predictions for writing to results
    C, A, lambda_param = popt_primary
    sigma_2023 = residuals_stats['sigma_2023_present']
    
    # Calculate projections for 2030 and 2035
    start_date = pd.to_datetime(df_train['Date'].iloc[0])
    t_2030 = (pd.to_datetime('2030-10') - start_date).days / 365.25
    t_2035 = (pd.to_datetime('2035-10') - start_date).days / 365.25
    
    log_ratio_2030 = saturating_exponential(t_2030, C, A, lambda_param, GOLD_SUPPLY_LEAK_RATE)
    log_ratio_2035 = saturating_exponential(t_2035, C, A, lambda_param, GOLD_SUPPLY_LEAK_RATE)
    
    ratio_2030 = np.exp(log_ratio_2030)
    ratio_2035 = np.exp(log_ratio_2035)
    
    # Calculate tight bounds
    ratio_2030_1sig_lower = np.exp(log_ratio_2030 - sigma_2023)
    ratio_2030_1sig_upper = np.exp(log_ratio_2030 + sigma_2023)
    ratio_2030_2sig_lower = np.exp(log_ratio_2030 - 2*sigma_2023)
    ratio_2030_2sig_upper = np.exp(log_ratio_2030 + 2*sigma_2023)
    
    ratio_2035_1sig_lower = np.exp(log_ratio_2035 - sigma_2023)
    ratio_2035_1sig_upper = np.exp(log_ratio_2035 + sigma_2023)
    ratio_2035_2sig_lower = np.exp(log_ratio_2035 - 2*sigma_2023)
    ratio_2035_2sig_upper = np.exp(log_ratio_2035 + 2*sigma_2023)
    
    # Write falsifiable predictions to results file
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write("\nFALSIFIABLE PREDICTIONS (Using Post-2023 Volatility)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Sigma used for bounds: {sigma_2023:.4f}\n\n")
        
        f.write("2030 Predictions:\n")
        f.write(f"  Mean projection: {ratio_2030:.1f} oz gold per BTC\n")
        f.write(f"  68% confidence interval (±1σ): [{ratio_2030_1sig_lower:.1f}, {ratio_2030_1sig_upper:.1f}] oz\n")
        f.write(f"  95% confidence interval (±2σ): [{ratio_2030_2sig_lower:.1f}, {ratio_2030_2sig_upper:.1f}] oz\n\n")
        
        f.write("2035 Predictions:\n")
        f.write(f"  Mean projection: {ratio_2035:.1f} oz gold per BTC\n")
        f.write(f"  68% confidence interval (±1σ): [{ratio_2035_1sig_lower:.1f}, {ratio_2035_1sig_upper:.1f}] oz\n")
        f.write(f"  95% confidence interval (±2σ): [{ratio_2035_2sig_lower:.1f}, {ratio_2035_2sig_upper:.1f}] oz\n\n")
        
        f.write("Hypothesis Testing Criteria:\n")
        f.write("The combined hypothesis of saturating adoption and volatility stabilization\n")
        f.write("should be rejected if:\n")
        f.write(f"  - 2030: Bitcoin's Gold Price falls outside [{ratio_2030_2sig_lower:.1f}, {ratio_2030_2sig_upper:.1f}] oz\n")
        f.write(f"  - 2035: Bitcoin's Gold Price falls outside [{ratio_2035_2sig_lower:.1f}, {ratio_2035_2sig_upper:.1f}] oz\n\n")
        f.write("These bounds assume the post-2023 volatility regime persists.\n")
        f.write("External shocks (security issues, regulatory changes) would constitute\n")
        f.write("failures of the volatility stability hypothesis.\n\n")
    
    print(f"\nFalsifiable Predictions Generated:")
    print(f"  2030: {ratio_2030:.1f} oz (95% CI: [{ratio_2030_2sig_lower:.1f}, {ratio_2030_2sig_upper:.1f}])")
    print(f"  2035: {ratio_2035:.1f} oz (95% CI: [{ratio_2035_2sig_lower:.1f}, {ratio_2035_2sig_upper:.1f}])")
    
    # Create the figure
    create_figure_tight_projections(
        df_train, df_test, popt_primary, GOLD_SUPPLY_LEAK_RATE, 
        residuals_stats['sigma_2023_present'],
        OUTPUT_PATH + 'figure_tight_projections.png'
    )
    
    # ========================================================================
    # Done
    # ========================================================================
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"\nResults saved to: {results_file}")
    print(f"\nFigures saved:")
    print(f"  - figure_usd_prices.png")
    print(f"  - figure_ratio_history.png")
    print(f"  - figure_model_comparison.png")
    print(f"  - figure_model_fit.png")
    print(f"  - figure_projection.png")
    print(f"  - figure_trailing_average.png")
    print(f"  - figure_residuals_qualitative.png")
    print(f"  - figure_residuals_quantitative.png")
    print(f"  - figure_tight_projections.png")
    print()


if __name__ == "__main__":
    main()
