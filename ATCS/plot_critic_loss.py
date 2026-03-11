import os
import pandas as pd
import matplotlib.pyplot as plt

scenarios = [
    "normal_2intersection",
    "normal_3intersection",
    "normal_4intersection",
    "crowded_2intersection",
    "crowded_3intersection",
    "crowded_4intersection",
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, scenario in enumerate(scenarios):
    csv_path = os.path.join("checkpoints", scenario, f"{scenario}_training_log.csv")
    ax = axes[idx]

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Verify columns exist
        if "Critic_Loss" in df.columns:
            # Sometime Episode restarts, we can just use the index for x-axis to be safe,
            # or cumulative episodes.
            ax.plot(range(len(df)), df["Critic_Loss"], color="purple", alpha=0.7)
            ax.set_title(scenario.replace("_", " ").title())
            ax.set_xlabel("Training Episodes")
            ax.set_ylabel("Critic Loss")
            ax.grid(True, linestyle="--", alpha=0.6)
        else:
            ax.text(0.5, 0.5, "Critic_Loss column missing", ha="center", va="center")
            ax.set_title(scenario.replace("_", " ").title())
    else:
        ax.text(0.5, 0.5, "CSV missing", ha="center", va="center")
        ax.set_title(scenario.replace("_", " ").title())

plt.tight_layout()
output_path = os.path.join("checkpoints", "critic_loss_comparison.png")
plt.savefig(output_path, dpi=300)
plt.close()
print(f"Saved critic loss plot to: {output_path}")
