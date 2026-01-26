import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.stats import f_oneway

# Mock data matching your format
data = {
    "Policy": [
        "resnet_torques", "resnet_torques", "resnet_pressure", "resnet_pressure",
        "original_act", "original_act", "dino", "dino",
        "resnet", "resnet", "dino_torques", "dino_torques",
        "resnet_active_cam", "resnet_active_cam", "resnet_torques", "resnet_pressure",
        "original_act", "dino"
    ],
    "Color": [
        "black", "green", "black", "red", "green", "black",
        "red", "green", "black", "black", "green", "red",
        "black", "green", "black", "red", "green", "black"
    ],
    "Direction": [
        "right", "left", "right", "left", "right", "left",
        "right", "left", "right", "left", "right", "left",
        "right", "left", "right", "left", "right", "left"
    ],
    "Hand Move to Cube": [
        1, 0.8, 0.7, 0.6, 0.9, 0.5, 0.8, 0.7, 0.6, 0.5,
        0.95, 0.85, 0.75, 0.8, 1, 0.7, 0.9, 0.6
    ],
    "Hand Grasp Cube": [
        0.9, 1, 0.8, 0.7, 0.6, 0.8, 0.9, 0.8, 0.7, 0.6,
        1, 0.9, 0.8, 0.9, 0.9, 0.8, 1, 0.7
    ],
    "Hand Move to Box": [
        0.8, 0.9, 0.7, 0.6, 0.8, 0.5, 0.7, 0.8, 0.6, 0.5,
        0.9, 0.8, 0.7, 0.8, 0.8, 0.7, 0.9, 0.6
    ],
    "Hand Cube in Box": [
        0.7, 0.8, 0.6, 0.5, 0.7, 0.4, 0.6, 0.7, 0.5, 0.4,
        0.8, 0.7, 0.6, 0.7, 0.7, 0.6, 0.8, 0.5
    ]
}

df = pd.DataFrame(data)

# Calculate policy statistics
policy_stats = df.groupby("Policy").mean(numeric_only=True)
policies = policy_stats.index.tolist()
subtasks = policy_stats.columns.tolist()

# Create figure with 4 subplots
fig = plt.figure(figsize=(20, 15))
fig.suptitle("Policy Performance Comparison", fontsize=16, y=1.02)

# 1. Stacked Bar Plot with Error Bars
ax1 = plt.subplot(2, 2, 1)

# Compute means/stds per policy and subtask
policy_means = df.groupby("Policy")[subtasks].mean()
policy_stds = df.groupby("Policy")[subtasks].std()

policies = policy_means.index.tolist()
x = np.arange(len(policies))
bottom = np.zeros(len(policies))

# Toggle which error bars to show
show_component_errbars = True
show_total_errbars = True

# Colors per subtask
colors = plt.cm.Set2(np.linspace(0, 1, len(subtasks)))

# Stack bars with per-segment error bars (std of each subtask)
for i, subtask in enumerate(subtasks):
    heights = policy_means[subtask].values
    errs = policy_stds[subtask].values if show_component_errbars else None
    ax1.bar(
        x,
        heights,
        bottom=bottom,
        label=subtask,
        color=colors[i],
        yerr=errs,
        capsize=3 if show_component_errbars else 0,
        linewidth=0.5,
        edgecolor="white",
    )
    bottom += heights

# Error bars for the total stack (std of sum across subtasks per trial)
if show_total_errbars:
    # Build per-policy series of per-trial sums across subtasks
    sum_series = {
        p: df[df["Policy"] == p][subtasks].sum(axis=1) for p in policies
    }
    total_stds = np.array([s.std(ddof=1) for s in sum_series.values()])
    # Draw error bars at the top of each stack
    ax1.errorbar(
        x,
        bottom,
        yerr=total_stds,
        fmt="none",
        ecolor="k",
        elinewidth=1.5,
        capsize=4,
        label="Total ±1 std",
    )

ax1.set_title("Stacked Success Rates per Policy (with Error Bars)")
ax1.set_xlabel("Policy")
ax1.set_ylabel("Sum of Subtask Success Rates")
ax1.set_xticks(x)
ax1.set_xticklabels(policies, rotation=45, ha="right")
ax1.set_ylim(0, max(1.0, bottom.max() * 1.1))  # up to #subtasks
ax1.legend(title="Subtask", bbox_to_anchor=(1.05, 1))
ax1.grid(axis="y", linestyle="--", alpha=0.6)

# 2. Radar Chart
ax2 = plt.subplot(2, 2, 2, polar=True)
angles = [n / float(len(subtasks)) * 2 * pi for n in range(len(subtasks))]
angles += angles[:1]

ax2.set_theta_offset(pi / 2)
ax2.set_theta_direction(-1)
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(subtasks)

for policy in policies:
    values = policy_stats.loc[policy].tolist()
    values += values[:1]
    ax2.plot(angles, values, linewidth=2, label=policy)
    ax2.fill(angles, values, alpha=0.1)

ax2.set_title("Radar Chart: Overall Performance", pad=20)
ax2.legend(bbox_to_anchor=(1.1, 1.1))

# 3. Heatmap
ax3 = plt.subplot(2, 2, 3)
im = ax3.imshow(policy_stats.T, cmap="YlGnBu", aspect="auto")
plt.colorbar(im, ax=ax3, label="Success Rate")
ax3.set_yticks(range(len(subtasks)))
ax3.set_yticklabels(subtasks)
ax3.set_xticks(range(len(policies)))
ax3.set_xticklabels(policies, rotation=45)
ax3.set_title("Success Rates Across Subtasks")

# 4. Box Plot (group by policy; colored by subtask)
ax4 = plt.subplot(2, 2, 4)

n_policies = len(policies)
n_subtasks = len(subtasks)

# Prepare data ordered by policy group, then subtask
boxplot_data = []
for j, policy in enumerate(policies):
    for i, subtask in enumerate(subtasks):
        series = df[df["Policy"] == policy][subtask]
        # Ensure at least one value; boxplot handles NaN automatically
        boxplot_data.append(series)

# Compute positions: group boxes by policy, offset boxes within each group by subtask
group_spacing = n_subtasks + 1.0  # space between policy groups
group_centers = np.arange(n_policies) * group_spacing
offsets = np.linspace(-0.4, 0.4, n_subtasks)  # offsets within a group

positions = []
for j in range(n_policies):
    for i in range(n_subtasks):
        positions.append(group_centers[j] + offsets[i])

bp = ax4.boxplot(
    boxplot_data,
    positions=positions,
    widths=0.6 / max(1, n_subtasks),  # shrink width if many subtasks
    patch_artist=True,
)

# Color boxes by subtask
colors = plt.cm.tab10(np.linspace(0, 1, n_subtasks))
for j in range(n_policies):
    for i in range(n_subtasks):
        idx = j * n_subtasks + i  # index in bp['boxes']
        bp["boxes"][idx].set_facecolor(colors[i])

ax4.set_title("Variability Across Trials")
ax4.set_xlabel("Policy")
ax4.set_ylabel("Success Rate")
ax4.set_xticks(group_centers)
ax4.set_xticklabels(policies, rotation=45, ha="right")
ax4.set_ylim(0, 1)

# Legend for subtasks
for i, subtask in enumerate(subtasks):
    ax4.plot([], [], color=colors[i], label=subtask)
ax4.legend(title="Subtask", bbox_to_anchor=(1.05, 1))
ax4.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

# Statistical significance test
print("\nStatistical Significance Tests (ANOVA p-values):")
for subtask in subtasks:
    groups = [df[df["Policy"] == policy][subtask] for policy in policies]
    try:
        _, p_value = f_oneway(*groups)
        print(f"{subtask}: p = {p_value:.4f}")
    except:
        print(f"{subtask}: Could not compute p-value (insufficient data)")
