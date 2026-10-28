"""Regenerate only current report tables and baseline curves; no training."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/primary"
FIG = ROOT / "report/figures"
TABLE = ROOT / "report/tables"
METHODS = ["baseline", "consistency90", "spatial_gated"]
NAMES = {"baseline": "ABMIL", "consistency90": "Consistency-90", "spatial_gated": "Gated spatial"}
KEYS = ["C16", "C17", "BRACS"]
METRICS = ["test_accuracy_pct", "test_macro_auroc", "test_macro_f1", "test_loss"]
COLORS = ["#333333", "#2a6f97"]


def table(name, headers, rows):
    text = ["| " + " | ".join(map(str, headers)) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |"]
    text += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    (TABLE / (name + ".md")).write_text("\n".join(text) + "\n")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA / "all_runs.csv")
    assert len(frame) == 90
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    rows, compact = [], []
    for key in KEYS:
        for method in METHODS:
            f = frame[(frame.dataset_key == key) & (frame.variant == method)]
            assert len(f) == 10 and set(f.training_seed) == set(range(42, 52))
            cells = [f"{f[m].mean():.{2 if m == METRICS[0] else 4}f} ± "
                     f"{f[m].std(ddof=1):.{2 if m == METRICS[0] else 4}f}" for m in METRICS]
            rows.append([key, NAMES[method], *cells])
            if method == "baseline":
                compact.append([key, *cells])
    table("primary", ["Data", "Method", "Accuracy (%)", "Macro AUROC", "Macro F1", "Test CE"], rows)
    table("baseline", ["Data", "Accuracy (%)", "Macro AUROC", "Macro F1", "Test CE"], compact)
    paired_rows = []
    for key in KEYS:
        base = frame[(frame.dataset_key == key) & (frame.variant == "baseline")].set_index("training_seed")
        for method in METHODS[1:]:
            ext = frame[(frame.dataset_key == key) & (frame.variant == method)].set_index("training_seed")
            delta = ext[METRICS] - base[METRICS]
            cells = []
            for m in METRICS[:3]:
                digits = 2 if m == METRICS[0] else 4
                cells.append(f"{delta[m].mean():+.{digits}f} ± {delta[m].std(ddof=1):.{digits}f}")
            wins = "/".join(str(int((delta[m] > 0).sum())) for m in METRICS[:3])
            paired_rows.append([key, NAMES[method], *cells, wins])
    table("paired", ["Data", "Extension", "Δ acc (pp)", "Δ AUROC", "Δ F1", "Wins A/U/F"], paired_rows)

    # Baseline means and between-seed SDs at each completed epoch, not one chosen run.
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 7.2), layout="constrained")
    for row, key in enumerate(KEYS):
        histories = [pd.read_csv(DATA / "baseline" / f"seed_{seed}" / key / "history.csv")
                     for seed in range(42, 52)]
        for col, specs in enumerate([
            [("train_loss", "Train CE"), ("val_loss", "Validation CE")],
            [("train_acc", "Train"), ("val_acc", "Validation")],
            [("val_auc", "Validation AUROC"), ("val_f1", "Validation F1")],
        ]):
            ax = axes[row, col]
            for index, (metric, label) in enumerate(specs):
                values = np.stack([h[metric].to_numpy() for h in histories])
                mean, sd = values.mean(0), values.std(0, ddof=1)
                epochs = np.arange(1, 31)
                ax.plot(epochs, mean, color=COLORS[index], label=label, linewidth=1.4)
                ax.fill_between(epochs, mean - sd, mean + sd, color=COLORS[index], alpha=.12)
            ax.set_title(f"{key}: " + ["cross-entropy", "accuracy (%)", "checkpoint metrics"][col])
            ax.set_xlabel("Completed epoch")
            ax.grid(alpha=.15)
            if col == 1:
                ax.set_ylim(0, 103)
            if col == 2:
                ax.set_ylim(0, 1.03)
            ax.legend(fontsize=7)
    fig.savefig(FIG / "baseline_learning_curves.png", dpi=220)
    plt.close(fig)

    recalls = []
    for method in METHODS:
        prs = [pd.read_csv(DATA / method / f"seed_{seed}/C17/test_per_class_metrics.csv")
               for seed in range(42, 52)]
        for label in range(4):
            entries = [f.iloc[label] for f in prs]
            recalls.append(dict(method=method, class_id=label,
                                support=int(entries[0]["support"]),
                                recall=np.mean([e["recall"] for e in entries])))
    table("c17_recall", ["Class ID", "Test slides", "ABMIL recall", "Consistency recall", "Spatial recall"],
          [[i, next(x["support"] for x in recalls if x["class_id"] == i),
            *[f'{next(x["recall"] for x in recalls if x["class_id"] == i and x["method"] == m):.4f}'
              for m in METHODS]] for i in range(4)])
    dataset_rows = []
    for key in KEYS:
        audit = json.loads((DATA / "baseline/seed_42" / key / "split_audit.json").read_text())
        parts = audit["splits"]
        dataset_rows.append([key, {"C16": 2, "C17": 4, "BRACS": 7}[key],
                             *[parts[k]["slides"] for k in ["Train", "Validation", "Test"]],
                             f'{sum(v["patches"] for v in parts.values()):,}'])
    table("datasets", ["Data", "Classes", "Train", "Validation", "Test", "Total patches"], dataset_rows)
    print("Regenerated the five current report tables and baseline learning curves.")


if __name__ == "__main__":
    main()
