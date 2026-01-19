import matplotlib.pyplot as plt
import numpy as np

# Create the Kelly allocation figure with even percentages
fig, ax = plt.subplots(figsize=(12, 7))

# Data for three columns
labels = ['Full Kelly\n(Rational)', 'Half Kelly\n(Conservative)', 'Quarter Kelly\n(Status-Quo-Favoring)']

# Full Kelly: 25% Gold, 75% Bitcoin
# Half Kelly: 50% of 75% Bitcoin = 37.5% Bitcoin, 62.5% Gold
# Quarter Kelly: 25% of 75% Bitcoin = 18.75% Bitcoin, 81.25% Gold
bitcoin_pct = [75, 37.5, 18.75]
gold_pct = [25, 62.5, 81.25]

x = np.arange(len(labels))
width = 0.6

# Create stacked bar chart
bars1 = ax.bar(x, bitcoin_pct, width, label='Bitcoin', color='#FF8C00')  # Orange
bars2 = ax.bar(x, gold_pct, width, bottom=bitcoin_pct, label='Gold', color='#FFD700')  # Gold/Yellow

# Add percentage labels on bars
for i, (btc, gld) in enumerate(zip(bitcoin_pct, gold_pct)):
    # Bitcoin label (in lower portion)
    ax.text(i, btc/2, f'{btc}%', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    # Gold label (in upper portion)
    ax.text(i, btc + gld/2, f'{gld}%', ha='center', va='center', fontsize=14, fontweight='bold', color='#333333')

ax.set_ylabel('Allocation (%)', fontsize=14, fontweight='bold')
ax.set_title('Within-Bucket Kelly Allocations\n(Bitcoin vs Gold, Post-2023 Volatility Regime)', fontsize=21, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.set_ylim(0, 100)

# Add gridlines
ax.yaxis.grid(True, linestyle='-', alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('C:/dev/GoldvBTC/figure_kelly_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Figure 10 regenerated successfully with corrected percentages:")
print("Full Kelly: 75% Bitcoin / 25% Gold")
print("Half Kelly: 37.5% Bitcoin / 62.5% Gold")
print("Quarter Kelly: 18.75% Bitcoin / 81.25% Gold")
