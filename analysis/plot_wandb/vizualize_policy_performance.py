import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data (manually entered from your table)
data = {
    "Color": ["black", "black", "Black", "black", "green", "black", "red", "red", "green", "green", "green", "green", "black", "black", "black", "black", "Green", "Green"],
    "Hand Move to Cube": [1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0.5, 1, 0, 1, 1, 0, 1, 1],
    "Hand Grasp Cube": [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
    "Hand Move to Box": [0, 0, 0, 1, 0, 1, 1, 0.5, 1, 1, 1, 0.5, 1, 1, 0, 0, 0, 1],
    "Hand Cube in Box": [0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
    "Direction": ["right", "right", "left", "left", "right", "right", "left", "left", "right", "left", "right", "left", "right", "right", "right", "right", "right", "right"]
}

# Convert to DataFrame
df = pd.DataFrame(data)
df["Trial"] = df.index + 1  # Add trial numbers

# Replace "Black"/"Green" with standardized colors for grouping
df["Color"] = df["Color"].str.lower()

# --- Plot 1: Stacked Bar Plot (Success Rates per Subtask) ---
plt.figure(figsize=(10, 6))
subtasks = ["Hand Move to Cube", "Hand Grasp Cube", "Hand Move to Box", "Hand Cube in Box"]
success_rates = {subtask: df[subtask].dropna().mean() for subtask in subtasks}

plt.bar(subtasks, success_rates.values(), color=['skyblue', 'salmon', 'lightgreen', 'gold'])
plt.title("Success Rate per Subtask (Average Across Trials)")
plt.ylabel("Success Rate")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# --- Plot 2: Grouped Bar Plot (Success by Color) ---
plt.figure(figsize=(12, 6))
colors = df["Color"].unique()
bar_width = 0.2
x = np.arange(len(subtasks))

for i, color in enumerate(colors):
    color_df = df[df["Color"] == color]
    means = [color_df[subtask].dropna().mean() for subtask in subtasks]
    plt.bar(x + i * bar_width, means, width=bar_width, label=color)

plt.title("Success Rate by Color Group")
plt.ylabel("Success Rate")
plt.xlabel("Subtask")
plt.xticks(x + bar_width * (len(colors) - 1) / 2, subtasks, rotation=45)
plt.legend(title="Color")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# --- Plot 3: Spider/Radar Chart (Per-Trial Performance) ---
categories = subtasks
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the loop

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories)

# Plot each trial
for trial in df["Trial"]:
    values = df.loc[trial - 1, subtasks].dropna().tolist()
    values += values[:1]  # Close the loop
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=f"Trial {trial}")
    ax.fill(angles, values, alpha=0.1)

plt.title("Spider Chart: Subtask Performance per Trial", pad=20)
plt.legend(bbox_to_anchor=(1.1, 1))
plt.show()

# --- Plot 4: Heatmap (Success by Trial and Subtask) ---
plt.figure(figsize=(10, 6))
heatmap_data = df[subtasks].T
plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
plt.colorbar(label="Success (0=Fail, 1=Success)")
plt.yticks(range(len(subtasks)), subtasks)
plt.xticks(range(len(df)), df["Trial"])
plt.title("Heatmap: Subtask Success Across Trials")
plt.show()
