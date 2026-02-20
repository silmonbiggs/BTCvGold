"""
Out-of-Sample Z-Score and CUSUM Analysis
Bitcoin's Gold Price - Sequential Hypothesis Testing

Computes z-scores in log space against the saturating exponential model,
then plots the cumulative sum (CUSUM) against sequential rejection boundaries.

Two boundaries are shown:
  1. iid boundary (c=2.625): assumes independent monthly residuals
  2. Autocorrelation-corrected boundary (c=8.50): accounts for the observed
     boom-bust cyclicality (lag-1 autocorrelation rho=0.90 in training residuals)

Data sources:
  - Training:   btc_gold_training_2015_2024.csv   (model calibration)
  - Test set:   btc_gold_test_2025(&26Jan).csv     (published with paper)
  - Update set: btc_gold_update.csv                (post-publication monthly data)

To add new monthly data, append rows to btc_gold_update.csv.

Author: S. James Biggs
Acknowledgment: The author thanks Anthropic's Claude Code (Opus 4.6) for contributions
to coding, data analysis, and presentation of this work.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = './'
TRAINING_FILE = 'btc_gold_training_2015_2024.csv'
TEST_FILE = 'btc_gold_test_2025(&26Jan).csv'
UPDATE_FILE = 'btc_gold_update.csv'
OUTPUT_PATH = './figures/'

# Fitted model parameters (from training on 2015-01 to 2024-12)
MODEL_C = -2.252333
MODEL_A = 5.617963
MODEL_LAMBDA = 0.283095
MODEL_G = 0.020
SIGMA_POST2023 = 0.2554  # post-2023 log-space standard deviation
SIGMA_FULL = 0.4782      # full-sample residual std (2015-2024)

# Training start date (for computing t in years)
START_DATE = pd.to_datetime('2015-01-01')

# Sequential testing boundary constants (one-sided, alpha=0.05, K=120 looks)
# Computed via Monte Carlo (2M simulations each)
C_IID = 2.625       # assumes iid monthly z-scores
C_CORRECTED = 8.50   # AR(1) with rho=0.90 (estimated from training residuals)

# Training residual lag-1 autocorrelation (for reporting)
RHO_TRAINING = 0.898


# ============================================================================
# MODEL
# ============================================================================

def model_ln_ratio(t):
    """
    Saturating exponential model prediction in log space.

        ln(R(t)) = C + g*t + A(1 - e^(-lambda*t))

    Args:
        t: time in years from START_DATE

    Returns:
        predicted ln(Gold_oz_per_Bitcoin)
    """
    return MODEL_C + MODEL_G * t + MODEL_A * (1 - np.exp(-MODEL_LAMBDA * t))


def load_oos_data():
    """
    Load out-of-sample data from the published test CSV and the update CSV.

    Returns:
        df: DataFrame with all OOS rows
        n_test: number of rows from the published test set
    """
    df_test = pd.read_csv(DATA_PATH + TEST_FILE)
    n_test = len(df_test)

    try:
        df_update = pd.read_csv(DATA_PATH + UPDATE_FILE)
        df = pd.concat([df_test, df_update], ignore_index=True)
    except FileNotFoundError:
        print(f"  No update file found ({UPDATE_FILE}); using test set only.")
        df = df_test

    df['Date'] = pd.to_datetime(df['Date'])
    return df, n_test


def compute_z_scores(df):
    """
    Compute z-scores for out-of-sample observations.

    Z-scores are computed in log space:
        z_i = (ln(R_actual) - ln(R_predicted)) / sigma

    where sigma is the post-2023 log-space standard deviation.
    """
    t = (df['Date'] - START_DATE).dt.days / 365.25
    ln_actual = np.log(df['Gold_oz_per_Bitcoin'].values)
    ln_predicted = model_ln_ratio(t.values)
    residuals = ln_actual - ln_predicted
    z_scores = residuals / SIGMA_POST2023

    df = df.copy()
    df['Predicted'] = np.exp(ln_predicted)
    df['ln_actual'] = ln_actual
    df['ln_pred'] = ln_predicted
    df['residual'] = residuals
    df['z_score'] = z_scores
    return df


def compute_training_autocorrelation():
    """
    Compute and display autocorrelation of training residuals.
    """
    df = pd.read_csv(DATA_PATH + TRAINING_FILE)
    dates = pd.to_datetime(df['Date'])
    t = np.array([(d - START_DATE).days / 365.25 for d in dates])
    ln_ratio = np.log(df['Gold_oz_per_Bitcoin'].values)
    ln_pred = model_ln_ratio(t)
    residuals = ln_ratio - ln_pred

    r_mean = residuals.mean()
    r_var = np.sum((residuals - r_mean)**2)

    print("TRAINING RESIDUAL AUTOCORRELATION")
    print("-" * 45)
    acfs = []
    for lag in range(1, 19):
        acf = np.sum((residuals[lag:] - r_mean) * (residuals[:-lag] - r_mean)) / r_var
        acfs.append(acf)
        bar = '#' * int(abs(acf) * 30)
        sign = '+' if acf > 0 else '-'
        print(f'  lag {lag:>2}: {acf:+.3f}  {sign}{bar}')

    print(f'\n  Lag-1 rho = {acfs[0]:.3f} (AR(1) half-life: '
          f'{np.log(2)/(-np.log(acfs[0])):.1f} months)')
    print(f'  Zero-crossing at ~lag 9; most negative at ~lag 15-16')
    print(f'  Consistent with boom-bust half-cycles of ~1.5 years')
    print()

    return acfs[0]


# ============================================================================
# FIGURE: CUSUM CHART
# ============================================================================

def create_figure_cusum(df, n_test, output_path):
    """
    CUSUM chart for sequential trajectory testing with symmetric two-sided
    boundaries and colored rejection/safe zones.

    Plots the cumulative sum of z-scores against two pairs of rejection
    boundaries (±c*sqrt(n)):
      1. iid boundary (c=2.625): assumes independent residuals
      2. Corrected boundary (c=8.50): accounts for autocorrelation

    Zones:
      - Green:        inside iid boundaries (model strongly supported)
      - Light green:  between iid and corrected boundaries (model supported)
      - Light red:    outside corrected boundaries (model rejected)

    Boundary constants computed via Monte Carlo (2M simulations, K=120
    monthly looks, alpha=0.05).
    """
    plt.rcParams.update({
        'font.size': 15,
        'axes.labelsize': 18,
        'axes.titlesize': 21,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 15,
    })

    fig, ax = plt.subplots(figsize=(12, 8))

    dates = df['Date'].values
    z_scores = df['z_score'].values
    n_points = len(z_scores)

    # Cumulative sum
    cusum = np.cumsum(z_scores)

    # Boundaries: extend forward 5 years for visual context
    n_extend = n_points + 60
    ns_boundary = np.arange(1, n_extend + 1)
    boundary_iid = C_IID * np.sqrt(ns_boundary)
    boundary_corr = C_CORRECTED * np.sqrt(ns_boundary)
    C_TRAJECTORY = C_CORRECTED * (SIGMA_FULL / SIGMA_POST2023)
    boundary_traj = C_TRAJECTORY * np.sqrt(ns_boundary)

    dates_boundary = pd.date_range(
        start=df['Date'].iloc[0],
        periods=n_extend,
        freq='MS'
    )

    # Symmetric y-axis
    y_lim = boundary_traj[n_extend - 1] + 5
    ax.set_ylim(-y_lim, y_lim)

    # --- Colored zones (background, zorder=0-2) ---
    # Red outside trajectory boundary (both hypotheses rejected)
    ax.fill_between(dates_boundary, boundary_traj, y_lim,
                    color='#E74C3C', alpha=0.10, zorder=0)
    ax.fill_between(dates_boundary, -boundary_traj, -y_lim,
                    color='#E74C3C', alpha=0.10, zorder=0)

    # Yellow between corrected and trajectory boundaries (volatility rejected, trajectory intact)
    ax.fill_between(dates_boundary, boundary_corr, boundary_traj,
                    color='#F39C12', alpha=0.10, zorder=1)
    ax.fill_between(dates_boundary, -boundary_corr, -boundary_traj,
                    color='#F39C12', alpha=0.10, zorder=1)

    # Green inside corrected boundaries (both hypotheses supported)
    ax.fill_between(dates_boundary, -boundary_corr, boundary_corr,
                    color='#27AE60', alpha=0.10, zorder=2)

    # --- Boundary lines ---
    # iid boundary (reference only)
    ax.plot(dates_boundary, boundary_iid, '--', color='gray', linewidth=1.5,
            alpha=0.5, label=f'iid boundary (for reference)', zorder=5)
    ax.plot(dates_boundary, -boundary_iid, '--', color='gray', linewidth=1.5,
            alpha=0.5, zorder=5)

    # Corrected boundary (reduced volatility hypothesis)
    ax.plot(dates_boundary, boundary_corr, '--', color='#8E44AD', linewidth=2.0,
            alpha=0.7,
            label=f'Reduced volatility boundary (\u03c3={SIGMA_POST2023:.3f}, c={C_CORRECTED})',
            zorder=4)
    ax.plot(dates_boundary, -boundary_corr, '--', color='#8E44AD', linewidth=2.0,
            alpha=0.7, zorder=4)

    # Trajectory boundary (trajectory hypothesis, full-sample sigma)
    ax.plot(dates_boundary, boundary_traj, '--', color='#C0392B', linewidth=2.0,
            alpha=0.7,
            label=f'Trajectory boundary (\u03c3={SIGMA_FULL:.3f}, c={C_TRAJECTORY:.1f})',
            zorder=4)
    ax.plot(dates_boundary, -boundary_traj, '--', color='#C0392B', linewidth=2.0,
            alpha=0.7, zorder=4)

    # Zero line
    ax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5, zorder=3)

    # CUSUM line
    ax.plot(dates, cusum, 'o-', color='#2E86AB', linewidth=2.5,
            markersize=6, label='Cumulative z-score sum ($S_n$)',
            zorder=10, markeredgecolor='white', markeredgewidth=0.5)

    # Color individual points by z-score sign
    for i in range(n_points):
        color = '#27AE60' if z_scores[i] >= 0 else '#8B4513'
        ax.plot(dates[i], cusum[i], 'o', color=color, markersize=8,
                zorder=11, markeredgecolor='white', markeredgewidth=1.0)

    # Margin annotations
    current_S = cusum[-1]
    bnd_iid_now = boundary_iid[n_points - 1]
    bnd_corr_now = boundary_corr[n_points - 1]
    bnd_traj_now = boundary_traj[n_points - 1]
    margin_iid = bnd_iid_now - abs(current_S)
    margin_corr = bnd_corr_now - abs(current_S)
    margin_traj = bnd_traj_now - abs(current_S)

    # Mark where update data begins
    n_update = n_points - n_test
    if n_update > 0:
        boundary_date = dates[n_test - 1] + (dates[n_test] - dates[n_test - 1]) / 2
        ax.axvline(x=boundary_date,
                   color='gray', linewidth=1, linestyle=':', alpha=0.5)
        ax.text(dates[n_test] + pd.DateOffset(months=2),
                y_lim * 0.92,
                'Update', fontsize=12, color='gray', alpha=0.7,
                ha='center', va='top', style='italic')

    # Summary statistics text box
    mean_z = z_scores.mean()
    std_z = z_scores.std(ddof=1)
    textstr = (
        f'Sequential Hypothesis Test\n'
        f'\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n'
        f'Out-of-sample months: {n_points}\n'
        f'Mean z-score: {mean_z:+.3f}\n'
        f'Std of z-scores: {std_z:.3f}\n'
        f'Cumulative sum: {current_S:+.2f}\n'
        f'\n'
        f'Reduced vol. margin:  {margin_corr:+.1f}\n'
        f'Trajectory margin:  {margin_traj:+.1f}\n'
        f'\n'
        f'\u03c3 post-2023: {SIGMA_POST2023}\n'
        f'\u03c3 full-sample: {SIGMA_FULL}\n'
        f'\u03c1 = {RHO_TRAINING} (lag-1 autocorrelation)'
    )

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=13,
            verticalalignment='bottom', horizontalalignment='right', bbox=props,
            fontfamily='monospace', zorder=100)

    # Labels and title
    ax.set_xlabel('Date', fontsize=18, fontweight='bold')
    ax.set_ylabel('Cumulative Z-Score Sum ($S_n = \\Sigma\\, z_i$)',
                  fontsize=18, fontweight='bold')
    ax.set_title(
        'Sequential Test of Trajectory Hypothesis\n'
        'Cumulative Z-Scores of Out-of-Sample Residuals',
        fontsize=21, fontweight='bold', pad=20
    )

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=13)

    # X-axis formatting
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


# ============================================================================
# FIGURE: AUTOCORRELATION VISUALIZATION
# ============================================================================

def create_figure_autocorrelation(output_path):
    """
    Visualize the ~15-month boom-bust cyclicality in training residuals.

    Top panel: log-space residuals over time (training + OOS)
    Bottom panel: residuals overlaid with 15-month lagged copy, showing
                  anti-correlation (peaks align with troughs when lagged)
    """
    plt.rcParams.update({
        'font.size': 15,
        'axes.labelsize': 18,
        'axes.titlesize': 21,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 15,
    })

    # Load training data
    df_train = pd.read_csv(DATA_PATH + TRAINING_FILE)
    dates_train = pd.to_datetime(df_train['Date'])
    t_train = np.array([(d - START_DATE).days / 365.25 for d in dates_train])
    ln_ratio_train = np.log(df_train['Gold_oz_per_Bitcoin'].values)
    ln_pred_train = model_ln_ratio(t_train)
    resid_train = ln_ratio_train - ln_pred_train

    # Load OOS data
    df_oos, n_test = load_oos_data()
    t_oos = (df_oos['Date'] - START_DATE).dt.days.values / 365.25
    ln_ratio_oos = np.log(df_oos['Gold_oz_per_Bitcoin'].values)
    ln_pred_oos = model_ln_ratio(t_oos)
    resid_oos = ln_ratio_oos - ln_pred_oos

    # Combine
    all_dates = pd.concat([dates_train, df_oos['Date']]).reset_index(drop=True)
    all_resid = np.concatenate([resid_train, resid_oos])
    n_train = len(resid_train)
    n_total = len(all_resid)

    # Lag for boom-bust half-cycle (~15 months)
    LAG = 15  # months

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 1])

    # --- Top panel: residuals over time ---
    # Training
    ax1.bar(all_dates[:n_train], all_resid[:n_train], width=25,
            color=['#27AE60' if r >= 0 else '#E74C3C' for r in all_resid[:n_train]],
            alpha=0.7, label='Training residuals')
    # OOS
    ax1.bar(all_dates[n_train:], all_resid[n_train:], width=25,
            color=['#27AE60' if r >= 0 else '#E74C3C' for r in all_resid[n_train:]],
            alpha=0.7, edgecolor='black', linewidth=0.5)

    ax1.axhline(y=0, color='black', linewidth=0.8)
    ax1.axhline(y=SIGMA_POST2023, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax1.axhline(y=-SIGMA_POST2023, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax1.axhline(y=2*SIGMA_POST2023, color='gray', linewidth=1, linestyle=':', alpha=0.4)
    ax1.axhline(y=-2*SIGMA_POST2023, color='gray', linewidth=1, linestyle=':', alpha=0.4)

    # Mark training/OOS boundary
    ax1.axvline(x=dates_train.iloc[-1] + pd.DateOffset(days=15),
                color='gray', linewidth=1.5, linestyle=':', alpha=0.6)
    ax1.text(dates_train.iloc[-1] + pd.DateOffset(months=2), ax1.get_ylim()[0],
             'Out-of-sample →', fontsize=11, color='gray', alpha=0.7,
             va='bottom', ha='left', style='italic')

    ax1.set_ylabel('Log-Space Residual', fontsize=18, fontweight='bold')
    ax1.set_title('Model Residuals: Boom-Bust Cyclicality\n'
                  'ln(Actual) − ln(Predicted)',
                  fontsize=21, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Labels for sigma bands
    ax1.text(all_dates.iloc[0] + pd.DateOffset(months=2), SIGMA_POST2023 + 0.02,
             '+1σ', fontsize=11, color='gray', alpha=0.6)
    ax1.text(all_dates.iloc[0] + pd.DateOffset(months=2), -SIGMA_POST2023 + 0.02,
             '−1σ', fontsize=11, color='gray', alpha=0.6)
    ax1.text(all_dates.iloc[0] + pd.DateOffset(months=2), 2*SIGMA_POST2023 + 0.02,
             '+2σ', fontsize=11, color='gray', alpha=0.5)
    ax1.text(all_dates.iloc[0] + pd.DateOffset(months=2), -2*SIGMA_POST2023 + 0.02,
             '−2σ', fontsize=11, color='gray', alpha=0.5)

    ax1.xaxis.set_major_locator(mdates.YearLocator(1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # --- Bottom panel: original vs circularly-permuted lagged residuals ---
    # Circular permutation: lagged[i] = resid[(i - LAG) mod N]
    # Uses all N data points instead of N - LAG
    resid_lagged = np.roll(all_resid, LAG)  # shift forward by LAG

    ax2.plot(all_dates, all_resid, 'o-', color='#2E86AB', linewidth=1.5,
             markersize=3, label='Residual at month $t$', alpha=0.8)
    ax2.plot(all_dates, resid_lagged, 's-', color='#E74C3C', linewidth=1.5,
             markersize=3, label=f'Residual at month $t − {LAG}$ (circular)',
             alpha=0.6)

    ax2.axhline(y=0, color='black', linewidth=0.8)

    # Shade regions where they have opposite signs (anti-correlation)
    for i in range(n_total):
        if all_resid[i] * resid_lagged[i] < 0:  # opposite signs
            ax2.axvspan(all_dates.iloc[i] - pd.DateOffset(days=15),
                        all_dates.iloc[i] + pd.DateOffset(days=15),
                        alpha=0.08, color='green', zorder=0)

    # Circular correlation statistic (using training data)
    resid_lagged_train = np.roll(resid_train, LAG)
    corr = np.corrcoef(resid_train, resid_lagged_train)[0, 1]

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.98, 0.98,
             f'Lag-{LAG} circular correlation: {corr:+.3f}\n'
             f'(half-cycle ≈ {LAG} mo,\n full cycle ≈ {2*LAG} mo)',
             transform=ax2.transAxes, fontsize=13,
             verticalalignment='top', horizontalalignment='right',
             bbox=props, fontfamily='monospace')

    ax2.set_xlabel('Date', fontsize=18, fontweight='bold')
    ax2.set_ylabel('Log-Space Residual', fontsize=18, fontweight='bold')
    ax2.set_title(f'Residuals with {LAG}-Month Lag Overlay (Circular)\n'
                  'Anti-Correlation Demonstrates ~30-Month Boom-Bust Cycle',
                  fontsize=21, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper left', fontsize=13)

    ax2.xaxis.set_major_locator(mdates.YearLocator(1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("Out-of-Sample Z-Score Analysis")
    print("=" * 70 + "\n")

    # Show training autocorrelation
    rho = compute_training_autocorrelation()

    # Load OOS data from CSVs
    df_raw, n_test = load_oos_data()
    n_total = len(df_raw)
    n_update = n_total - n_test
    print(f"  Test data:   {n_test} months ({TEST_FILE})")
    print(f"  Update data: {n_update} months ({UPDATE_FILE})")
    print(f"  Total OOS:   {n_total} months")
    print()

    # Compute z-scores
    df = compute_z_scores(df_raw)

    # Print z-score table
    print(f"Model: ln(R(t)) = {MODEL_C:.4f} + {MODEL_G}·t"
          f" + {MODEL_A:.4f}(1 - e^(-{MODEL_LAMBDA:.4f}·t))")
    print(f"Post-2023 sigma (log space): {SIGMA_POST2023}")
    print()
    print("Z-SCORES (computed in log space)")
    print("=" * 80)
    print(f"{'Month':>10} {'Actual':>8} {'Pred':>8} {'ln(act)':>8} {'ln(pred)':>8} "
          f"{'resid':>8} {'z':>7} {'src':>6}")
    print("-" * 80)

    for i, (_, row) in enumerate(df.iterrows()):
        flag = '  **' if abs(row['z_score']) > 2 else ''
        src = 'test' if i < n_test else 'upd'
        print(f"{row['Date'].strftime('%Y-%m'):>10} "
              f"{row['Gold_oz_per_Bitcoin']:8.2f} "
              f"{row['Predicted']:8.2f} "
              f"{row['ln_actual']:8.4f} {row['ln_pred']:8.4f} "
              f"{row['residual']:+8.4f} {row['z_score']:+7.2f}"
              f"  {src}{flag}")

    z = df['z_score'].values
    n = len(z)
    S = z.sum()
    bnd_iid = -C_IID * np.sqrt(n)
    bnd_corr = -C_CORRECTED * np.sqrt(n)

    print()
    print(f"n = {n}")
    print(f"Mean z-score:       {z.mean():+.4f}")
    print(f"Std of z-scores:    {z.std(ddof=1):.4f}")
    print(f"Cumulative sum S_n: {S:+.2f}")
    print()
    print(f"iid boundary (c={C_IID}):       {bnd_iid:.2f}   margin = {S - bnd_iid:+.2f}")
    print(f"Corrected boundary (c={C_CORRECTED}):  {bnd_corr:.2f}  margin = {S - bnd_corr:+.2f}")
    print()

    within_1s = np.sum(np.abs(z) <= 1)
    within_2s = np.sum(np.abs(z) <= 2)
    print(f"Within ±1σ: {within_1s}/{n} ({within_1s/n:.0%}, expected ~68%)")
    print(f"Within ±2σ: {within_2s}/{n} ({within_2s/n:.0%}, expected ~95%)")
    print()

    if S > bnd_corr:
        print(f"STATUS: Trajectory hypothesis NOT rejected")
        if S > bnd_iid:
            print(f"  (above both boundaries)")
        else:
            print(f"  (below iid boundary but above corrected boundary)")
    else:
        print(f"STATUS: Trajectory hypothesis REJECTED at alpha=0.05")
        print(f"  (below corrected boundary)")

    # Generate figures
    print()
    print("Generating figures...")
    create_figure_cusum(df, n_test, OUTPUT_PATH + 'figure_cusum_trajectory.png')
    create_figure_autocorrelation(OUTPUT_PATH + 'figure_autocorrelation.png')

    print("\nDone.")


if __name__ == '__main__':
    main()
