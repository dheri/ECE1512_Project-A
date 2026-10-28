"""Plot saved paired accuracy changes for the poster; no training or inference."""
import csv
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT / "results/primary/paired_comparison.csv").open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["variant"] == "spatial_gated"]
    fig, ax = plt.subplots(figsize=(9.5, 2.8), layout="constrained")
    color = "#126479"
    for y, dataset in enumerate(["C16", "C17", "BRACS"]):
        selected = sorted((r for r in rows if r["dataset_key"] == dataset),
                          key=lambda r: int(r["training_seed"]))
        if [int(r["training_seed"]) for r in selected] != list(range(42, 52)):
            raise ValueError(f"Expected ten unique paired seeds for {dataset}.")
        values = [float(r["delta_test_accuracy_pct"]) for r in selected]
        mean, sd = statistics.mean(values), statistics.stdev(values)
        jitter = [(i - 4.5) * .032 for i in range(10)]
        ax.scatter(values, [y + j for j in jitter], s=22, alpha=.65, color=color,
                   zorder=3)
        ax.errorbar(mean, y - .25, xerr=sd, fmt="D", color="#172b36",
                    capsize=4, markersize=5, zorder=4)
        ax.text(1.03, y, f"{mean:+.2f} ± {sd:.2f} pp", va="center",
                ha="left", fontsize=11, transform=ax.get_yaxis_transform())
    ax.axvline(0, color="#8a949b", linewidth=1, linestyle="--")
    ax.set(yticks=range(3), yticklabels=["C16", "C17", "BRACS"],
           xlabel="Spatial − ABMIL test accuracy (percentage points)",
           xlim=(-8, 12), ylim=(2.45, -.65))
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_fontsize(12)
    ax.grid(axis="x", alpha=.18)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    destination = ROOT / "presentation/figures/spatial_accuracy_differences.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200, facecolor="white")
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()
