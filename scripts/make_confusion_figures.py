"""Validate saved test predictions and plot all nine confusion matrices, without training."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/primary"
METHODS = ["baseline", "consistency90", "spatial_gated"]
NAMES = ["ABMIL", "Masked consistency", "Gated spatial"]
DATASETS = ["C16", "C17", "BRACS"]
CLASSES = {"C16": 2, "C17": 4, "BRACS": 7}
SIZES = {"C16": 129, "C17": 200, "BRACS": 86}
SEEDS = list(range(42, 52))


def read_run(directory, size):
    saved = pd.read_csv(directory / "test_confusion_matrix.csv", index_col=0)
    predictions = pd.read_csv(directory / "test_predictions.csv")
    cm = saved.to_numpy()
    if (cm.shape != (size, size) or not np.issubdtype(cm.dtype, np.integer)
            or (cm < 0).any() or list(saved.index) != list(saved.columns)):
        raise ValueError(f"Invalid confusion matrix: {directory}")
    labels = predictions["label"].to_numpy(dtype=int)
    predicted = predictions["prediction"].to_numpy(dtype=int)
    if (len(predictions) != cm.sum() or predictions["slide_id"].duplicated().any()
            or not ((0 <= labels) & (labels < size)).all()
            or not ((0 <= predicted) & (predicted < size)).all()):
        raise ValueError(f"Invalid prediction identities or labels: {directory}")
    recomputed = np.zeros((size, size), dtype=int)
    np.add.at(recomputed, (labels, predicted), 1)
    probabilities = predictions[[f"p_class_{i}" for i in range(size)]].to_numpy()
    if (not np.isfinite(probabilities).all()
            or not np.array_equal(cm, recomputed)
            or not np.array_equal(predicted, probabilities.argmax(1))):
        raise ValueError(f"Confusion matrix or argmax disagrees with predictions: {directory}")
    return cm, list(saved.columns), predictions[["slide_id", "label"]]


def summarize(matrices, class_names, seeds):
    stack = np.stack(matrices)
    support = stack[0].sum(1)
    if (len(stack) != len(seeds) or (support <= 0).any()
            or not np.all(stack.sum(axis=2) == support)):
        raise ValueError("All seeds must use the same nonempty true-class supports.")
    norm = (stack / stack.sum(axis=2, keepdims=True)).mean(0)
    mean = stack.mean(0)
    np.testing.assert_allclose(norm, mean / support[:, None])
    np.testing.assert_allclose(norm.sum(1), 1)
    return {
        "unique_test_slides": int(support.sum()), "training_seeds": list(seeds),
        "class_names": class_names, "support": support.tolist(),
        "mean_counts_per_seed": mean.tolist(), "row_percentages": (norm * 100).tolist(),
        "mean_accuracy_pct": float(np.trace(mean) / support.sum() * 100),
        "diagonal_recall_pct": (norm.diagonal() * 100).tolist(),
    }


def collect(data_root=DATA):
    frame = pd.read_csv(data_root / "all_runs.csv")
    index = ["variant", "training_seed", "dataset_key"]
    if len(frame) != 90 or frame.duplicated(index).any():
        raise ValueError("Expected 90 unique primary method/seed/dataset records.")
    records = frame.set_index(index)
    data = {}
    for dataset in DATASETS:
        reference = None
        reference_names = None
        data[dataset] = {}
        for method in METHODS:
            matrices = []
            for seed in SEEDS:
                directory = data_root / method / f"seed_{seed}" / dataset
                cm, names, identities = read_run(directory, CLASSES[dataset])
                if reference is None:
                    reference, reference_names = identities, names
                pd.testing.assert_frame_equal(reference, identities)
                if names != reference_names or cm.sum() != SIZES[dataset]:
                    raise ValueError(f"Test set or class order changed: {directory}")
                actual = np.trace(cm) / cm.sum() * 100
                recorded = records.loc[(method, seed, dataset), "test_accuracy_pct"]
                if not np.isclose(actual, recorded, atol=1e-4, rtol=0):
                    raise ValueError(f"Accuracy disagrees with primary results: {directory}")
                matrices.append(cm)
            info = summarize(matrices, names, SEEDS)
            if method != METHODS[0] and info["support"] != data[dataset][METHODS[0]]["support"]:
                raise ValueError(f"Class support differs between methods: {dataset}")
            data[dataset][method] = info
    return data


def class_support_labels(ax, dataset, info, *, compact):
    classes = list(range(len(info["support"])))
    if dataset == "C16" and not compact:
        labels = [f"{name}\n(n={count})"
                  for name, count in zip(info["class_names"], info["support"])]
    else:
        labels = [f"{label} (n={count})" for label, count in zip(classes, info["support"])]
    ax.set_yticks(classes, labels, fontsize=7.6)
    return labels


def panel(ax, dataset, method, info, *, first_column, compact=False):
    values = np.array(info["row_percentages"])
    size = CLASSES[dataset]
    im = ax.imshow(values, cmap="Blues", vmin=0, vmax=100)
    ticks = ["Normal", "Tumor"] if dataset == "C16" and not compact else list(range(size))
    ax.set_xticks(range(size), ticks)
    ax.set_yticks(range(size), ticks)
    ax.tick_params(labelsize=8, length=0)
    ax.set_xticks(np.arange(size + 1) - .5, minor=True)
    ax.set_yticks(np.arange(size + 1) - .5, minor=True)
    ax.grid(which="minor", color="white", linewidth=.6)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Predicted class", fontsize=8)
    if first_column:
        ax.set_ylabel("True class", fontsize=8)
        class_support_labels(ax, dataset, info, compact=compact)
    ax.set_title(f"{NAMES[METHODS.index(method)]}\nAccuracy {info['mean_accuracy_pct']:.2f}%",
                 fontsize=9, pad=7)
    for i in range(size):
        for j in range(size):
            ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center",
                    color="white" if values[i, j] >= 55 else "#172b36",
                    fontsize=11 if size == 2 else 9 if size == 4 else 7.5,
                    fontweight="bold" if i == j else "normal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def save(fig, path):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [*fig.texts, *(text for ax in fig.axes for text in
             [ax.title, ax.xaxis.label, ax.yaxis.label,
              *ax.get_xticklabels(), *ax.get_yticklabels(), *ax.texts])]
    for artist in texts:
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            if (box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1
                    or box.y1 > fig.bbox.height + 1):
                raise ValueError(f"Text outside figure bounds: {artist.get_text()}")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def draw(data, destination):
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    for dataset in DATASETS:
        fig = plt.figure(figsize=(8.4, 3.6))
        height = .23 * 8.4 / 3.6
        axes = [fig.add_axes([.105 + .28 * col, .23, .23, height]) for col in range(3)]
        for col, method in enumerate(METHODS):
            im = panel(axes[col], dataset, method, data[dataset][method], first_column=col == 0)
        bar = fig.colorbar(im, cax=fig.add_axes([.92, .23, .015, height]), ticks=[0, 50, 100])
        bar.ax.tick_params(labelsize=8)
        fig.suptitle(f"{dataset} · {SIZES[dataset]} unique test slides · ten training seeds",
                     fontsize=10, y=.98)
        fig.supxlabel(
            "n at left = unique slides in each true class, shared by all three methods.\n"
            "Cells = mean row percentages over seeds 42–51 (100% before rounding).",
            fontsize=8, y=.035)
        save(fig, destination / f"{dataset}_confusions.png")

    fig, axes = plt.subplots(3, 3, figsize=(8, 8.6), layout="constrained")
    for row, dataset in enumerate(DATASETS):
        for col, method in enumerate(METHODS):
            info = data[dataset][method]
            im = panel(axes[row, col], dataset, method, info,
                       first_column=col == 0, compact=True)
            axes[row, col].set_title(
                f"{dataset} · {NAMES[col]}\nAccuracy {info['mean_accuracy_pct']:.2f}%",
                fontsize=8.8, pad=7)
    bar = fig.colorbar(im, ax=axes, ticks=[0, 25, 50, 75, 100], fraction=.025, pad=.02)
    bar.ax.tick_params(labelsize=8)
    bar.set_label("Percentage within each true class", fontsize=9)
    fig.suptitle("Test confusion matrices · unique class n and ten-seed percentages", fontsize=10.5)
    fig.supxlabel(
        "n at left = unique test slides per true class, shared by all three methods.\n"
        "Cells = mean row percentages over seeds 42–51; repeated slides are not new patients.\n"
        "C16: 0 = Normal, 1 = Tumor. C17/BRACS: numeric class IDs.", fontsize=7.8)
    save(fig, destination / "all_nine_confusions.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=ROOT / "report/figures")
    args = parser.parse_args()
    data = collect()
    draw(data, args.figures)
    summary = ROOT / "report/tables/test_confusion_summary.json"
    summary.write_text(json.dumps(data, indent=2) + "\n")
    validation = {
        "status": "passed", "saved_runs_checked": 90, "datasets": DATASETS,
        "methods": METHODS, "seeds": SEEDS,
        "matrices_equal_saved_predictions": True, "predictions_equal_probability_argmax": True,
        "same_test_slide_order_and_labels": True, "accuracies_match_primary_results": True,
        "text_within_figure_bounds": True, "row_percentage_sums": 100,
        "count_definition": "Unique test slides per true class, shared across methods and seeds.",
        "percentage_definition": "Mean per-seed row-normalized test confusion, not ensemble predictions.",
        "supports": {dataset: data[dataset]["baseline"]["support"] for dataset in DATASETS},
        "summary": str(summary.relative_to(ROOT)), "new_scientific_training": False,
    }
    (ROOT / "checks/confusion_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
