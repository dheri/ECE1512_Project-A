"""Regenerate all report tables and analytical figures from saved outputs; no training."""
from pathlib import Path
import json
import sys

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
COLORS = ["#333333", "#2a6f97", "#b34f2f"]


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
    paired_rows, paired = [], []
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
            for seed, values in delta.iterrows():
                paired.append(dict(dataset_key=key, variant=method, training_seed=seed,
                                   **{"delta_" + m: values[m] for m in METRICS}))
    table("paired", ["Data", "Extension", "Δ acc (pp)", "Δ AUROC", "Δ F1", "Wins A/U/F"], paired_rows)
    pd.DataFrame(paired).to_csv(TABLE / "paired_full_precision.csv", index=False)
    selected = []
    for key in KEYS:
        selected.append([key, *[
            f"{int(frame[(frame.dataset_key == key) & (frame.variant == m)].best_epoch.min())}–"
            f"{int(frame[(frame.dataset_key == key) & (frame.variant == m)].best_epoch.max())}"
            for m in METHODS]])
    table("selected_epochs", ["Data", "ABMIL range", "Consistency range", "Spatial range"], selected)

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

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 7.2), layout="constrained")
    p = pd.DataFrame(paired)
    for row, key in enumerate(KEYS):
        for col, metric in enumerate(METRICS[:3]):
            ax = axes[row, col]
            for index, method in enumerate(METHODS[1:]):
                values = p[(p.dataset_key == key) & (p.variant == method)]["delta_" + metric]
                x = index + np.linspace(-.15, .15, len(values))
                ax.scatter(x, values, color=COLORS[index + 1], s=22)
                ax.plot([index - .27, index + .27], [values.mean()] * 2,
                        color="black", linewidth=1.4)
            ax.axhline(0, color="gray", linestyle=":", linewidth=1)
            ax.set_xticks([0, 1], ["Consistency", "Spatial"])
            ax.set_title(f"{key}: Δ " + ["accuracy (pp)", "macro AUROC", "macro F1"][col])
            ax.grid(axis="y", alpha=.15)
    fig.savefig(FIG / "paired_seed_differences.png", dpi=220)
    plt.close(fig)

    recalls, matrices = [], {}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), layout="constrained")
    for index, method in enumerate(METHODS):
        cms, prs = [], []
        for seed in range(42, 52):
            path = DATA / method / f"seed_{seed}/C17"
            cm = pd.read_csv(path / "test_confusion_matrix.csv", index_col=0).to_numpy()
            cms.append(cm)
            prs.append(pd.read_csv(path / "test_per_class_metrics.csv"))
        cm = np.mean(cms, axis=0)
        matrices[method] = cm.tolist()
        norm = cm / cm.sum(1, keepdims=True)
        ax = axes[index]
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{100 * norm[i, j]:.1f}", ha="center", va="center",
                        color="white" if norm[i, j] > .55 else "black", fontsize=9)
        ax.set(xticks=range(4), yticks=range(4), xlabel="Predicted class ID",
               ylabel="True class ID", title=NAMES[method])
        for label in range(4):
            entries = [f.iloc[label] for f in prs]
            recalls.append(dict(method=method, class_id=label,
                                support=int(entries[0]["support"]),
                                recall=np.mean([e["recall"] for e in entries]),
                                precision=np.mean([e["precision"] for e in entries])))
    fig.savefig(FIG / "c17_confusion.png", dpi=220)
    plt.close(fig)
    pd.DataFrame(recalls).to_csv(TABLE / "c17_class_metrics.csv", index=False)
    table("c17_recall", ["Class ID", "Test slides", "ABMIL recall", "Consistency recall", "Spatial recall"],
          [[i, next(x["support"] for x in recalls if x["class_id"] == i),
            *[f'{next(x["recall"] for x in recalls if x["class_id"] == i and x["method"] == m):.4f}'
              for m in METHODS]] for i in range(4)])
    (TABLE / "c17_mean_confusions.json").write_text(json.dumps(matrices, indent=2) + "\n")

    historical = pd.read_csv(ROOT / "results/historical/all_runs.csv")
    history_rows = []
    comparisons = [
        ("Spatial real", "spatial_real", "baseline", "Original"),
        ("Shuffled geometry", "spatial_shuffled", "spatial_real", "Original"),
        ("Learned gate", "spatial_gated", "spatial_real", "Original"),
        ("Fixed gate", "spatial_fixed", "spatial_gated", "Original"),
        ("Compact context", "spatial_compact", "spatial_gated", "Original"),
        ("Consistency ramp", "consistency_ramp", "consistency90", "Original"),
        ("70% consistency", "consistency70", "consistency90", "Original"),
        ("70% masked CE", "dropout70", "consistency70", "Original"),
        ("Class weighting", "class_weighted", "balance_control", "Patient-disjoint"),
    ]
    for label, method, comparator, cohort in comparisons:
        deltas = []
        for key in KEYS:
            a = historical[(historical.arm == method) & (historical.dataset_key == key)]
            b = historical[(historical.arm == comparator) & (historical.dataset_key == key)]
            deltas.append(f"{a.test_accuracy_pct.mean() - b.test_accuracy_pct.mean():+.2f}")
        history_rows.append([label, comparator.replace("spatial_", ""), cohort, *deltas])
    table("historical", ["Change", "Comparator", "C17 cohort", "C16 Δ pp", "C17 Δ pp", "BRACS Δ pp"],
          history_rows)
    dataset_rows, count_rows = [], []
    for key in KEYS:
        audit = json.loads((DATA / "baseline/seed_42" / key / "split_audit.json").read_text())
        parts = audit["splits"]
        dataset_rows.append([key, {"C16": 2, "C17": 4, "BRACS": 7}[key],
                             *[parts[k]["slides"] for k in ["Train", "Validation", "Test"]],
                             f'{sum(v["patches"] for v in parts.values()):,}'])
        for name, part in parts.items():
            count_rows.append([key, name, "/".join(str(v) for v in part["class_counts"].values())])
    table("datasets", ["Data", "Classes", "Train", "Validation", "Test", "Total patches"], dataset_rows)
    table("class_counts", ["Data", "Split", "Class counts, ID order"], count_rows)
    print("Regenerated 8 tables, 3 analytical figures and full-precision supporting data.")


if __name__ == "__main__":
    main()
