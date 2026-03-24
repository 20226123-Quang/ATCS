import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from acac import load_scenario_config


REPO_ROOT = Path(__file__).resolve().parents[1]
scenarios = [scenario.name for scenario in load_scenario_config() if "compare_kpi" in scenario.tags]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, scenario in enumerate(scenarios[: len(axes)]):
    csv_path = str(REPO_ROOT / "checkpoints" / scenario / f"{scenario}_training_log.csv")
    ax = axes[idx]

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "Critic_Loss" in df.columns:
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

for ax in axes[len(scenarios):]:
    ax.axis("off")

plt.tight_layout()
output_path = str(REPO_ROOT / "checkpoints" / "critic_loss_comparison.png")
plt.savefig(output_path, dpi=300)
plt.close()
print(f"Saved critic loss plot to: {output_path}")
