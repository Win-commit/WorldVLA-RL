import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.ticker as ticker

# Config (ICLR style)
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.size": 12,
#     "axes.labelsize": 14,
#     "axes.titlesize": 15,
#     "legend.fontsize": 12,
#     "xtick.labelsize": 12,
#     "ytick.labelsize": 12,
#     "figure.dpi": 600,
#     "savefig.dpi": 600,
#     "text.usetex": False,
# })

# Constants
c0 = np.exp(-1) / (1 - np.exp(-1))  # ≈ 0.582
K_values_max = 15
K_values = np.arange(3, K_values_max+1)
z_values = norm.ppf(1 - 1 / K_values)

# μ values and rollout noise lines
mu_list = [0.5, 0.75, 
           1.0, 1.25, 1.5,1.75, 
           2.0, 2.25]
# colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(mu_list)))
colors = plt.cm.magma(np.linspace(0.1, 0.9, len(mu_list)))
# tau_actual_list = [0.3, 0.5, 0.7]
# linestyles = ['--', '-.', ':']

# Plotting
fig, ax = plt.subplots(figsize=(4, 2.4))

# Store intersection K*
intersection_table = []

for mu, color in zip(mu_list, colors):
    tau_min = c0 * mu / z_values
    ax.plot(K_values, tau_min, label=rf"$\mu = {mu}$", color=color, linewidth=1.5)
    
    # Compute intersection points K* for each τ_actual
    # for tau_actual in tau_actual_list:
    #     mask = tau_min <= tau_actual
    #     if np.any(mask):
    #         K_star = K_values[mask][0]
    #         intersection_table.append((mu, tau_actual, K_star))
    #         ax.scatter(K_star, tau_actual, marker='o', color=color, s=40,
    #                    label=rf"$K^*(\mu={mu}, \tau={tau_actual}) = {K_star}$"
    #                    )

# Plot actual rollout noise lines
# for tau_actual, ls in zip(tau_actual_list, linestyles):
#     ax.axhline(y=tau_actual, linestyle=ls, color='gray', linewidth=1.2,
#                label=rf"Actual rollout noise $\tau = {tau_actual}$"
#                )

# Final plot aesthetics
# ax.set_title(r"Filtering Gain Threshold $\tau_{\min}(K)$ and Actual Rollout Noise $\tau$")
ax.set_xlabel(r"Number of Candidates $K$")
ax.set_ylabel(r"Threshold $\tau_{\min}$")
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlim(3, K_values_max)
ax.set_ylim(0, max(c0 * max(mu_list) / z_values) * 1.1)

ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

ax.legend(loc='upper right', frameon=True, ncol=2, fontsize=8)

# Save
plt.tight_layout()
plt.savefig("tau_min_vs_K_intersections.png", dpi=600, bbox_inches='tight')
plt.show()

# Print intersection table
print("Intersection Table: Minimal K where filtering gain > 0:")
print("--------------------------------------------------------")
# for mu, tau, K_star in sorted(intersection_table):
#     print(f"μ = {mu:<4} | τ = {tau:<4} → K* = {K_star}")
