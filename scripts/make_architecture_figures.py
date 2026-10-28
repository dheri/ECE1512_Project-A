"""Draw implementation-specific architecture figures; no data, inference or training."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "report/figures"
INK = "#243746"
BLUE = "#eaf2f8"
TEAL = "#e7f3ef"
GOLD = "#fff2d8"


def canvas(height):
    fig, ax = plt.subplots(figsize=(7.2, height))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 7.2), ylim=(0, height))
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def label(ax, x, y, text, size=9.2, **kwargs):
    return ax.text(x, y, text, ha="center", va="center", fontsize=size,
                   color=INK, linespacing=1.15, **kwargs)


def box(ax, x, y, width, height, text, color=BLUE, size=9.2):
    ax.add_patch(FancyBboxPatch(
        (x, y - height / 2), width, height,
        boxstyle="round,pad=0.015,rounding_size=0.045",
        facecolor=color, edgecolor=INK, linewidth=.9))
    label(ax, x + width / 2, y, text, size)


def wire(ax, points, *, dashed=False, arrow=True):
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=INK,
                linewidth=.95, linestyle="--" if dashed else "-")
    start, end = points[-2:]
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>" if arrow else "-",
        mutation_scale=9, linewidth=.95, color=INK,
        linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0))


def operation(ax, x, y, text):
    ax.add_patch(Circle((x, y), .11, facecolor="white",
                        edgecolor=INK, linewidth=.95))
    label(ax, x, y, text, 12)


def save(fig, name):
    path = DESTINATION / name
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(path)


def baseline():
    fig, ax = canvas(2.1)
    box(ax, .04, 1.35, 1.12, .64, "Frozen features\nN × 384")
    box(ax, 1.42, 1.35, 1.43, .64, "Linear 384→256\nReLU", color=TEAL)
    wire(ax, [(1.18, 1.35), (1.40, 1.35)])
    wire(ax, [(2.87, 1.35), (3.08, 1.35)], arrow=False)
    ax.plot(3.08, 1.35, "o", color=INK, markersize=2.5)
    label(ax, 2.96, 1.58, r"$h_i$", 10)
    box(ax, 3.43, 1.70, 1.25, .48, "Linear 256→128\ntanh")
    box(ax, 3.43, 1.03, 1.25, .48, "Linear 256→128\nsigmoid", color=GOLD)
    wire(ax, [(3.08, 1.35), (3.08, 1.70), (3.41, 1.70)])
    wire(ax, [(3.08, 1.35), (3.08, 1.03), (3.41, 1.03)])
    operation(ax, 5.03, 1.35, "×")
    wire(ax, [(4.70, 1.70), (4.84, 1.70), (4.95, 1.43)])
    wire(ax, [(4.70, 1.03), (4.84, 1.03), (4.95, 1.27)])
    box(ax, 5.42, 1.35, 1.73, .50, "Linear 128→1\nsoftmax over N")
    wire(ax, [(5.15, 1.35), (5.40, 1.35)])
    box(ax, 3.43, .28, 1.25, .48, "Weighted sum\nz: 256", color=TEAL)
    wire(ax, [(3.08, 1.35), (3.08, .28), (3.41, .28)])
    wire(ax, [(6.27, 1.08), (6.27, .74), (4.05, .74), (4.05, .54)])
    label(ax, 5.25, .86, r"$\alpha_i$", 10)
    box(ax, 5.42, .28, 1.73, .48, "Linear 256→C\nclass logits")
    wire(ax, [(4.70, .28), (5.40, .28)])
    save(fig, "architecture_baseline.png")


def consistency():
    fig, ax = canvas(2.55)
    box(ax, .04, 1.86, 1.64, .62, "Full bag\nN patches")
    box(ax, .04, .62, 1.64, .62, "Retained bag\nN′ patches")
    wire(ax, [(.86, 1.53), (.86, .95)])
    label(ax, .86, 1.24, "keep 90%", 8.8,
          bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    box(ax, 2.10, 1.86, 1.45, .62, "ABMIL θ\nfull view", color=TEAL)
    box(ax, 2.10, .62, 1.45, .62, "ABMIL θ\nmasked view", color=TEAL)
    wire(ax, [(1.70, 1.86), (2.08, 1.86)])
    wire(ax, [(1.70, .62), (2.08, .62)])
    wire(ax, [(2.83, 1.53), (2.83, .95)], dashed=True, arrow=False)
    label(ax, 2.83, 1.24, "shared weights", 8.8,
          bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    for y, probability in [(1.86, "p"), (.62, "q")]:
        box(ax, 3.94, y, .96, .62, f"softmax\n{probability}: C classes")
        wire(ax, [(3.57, y), (3.92, y)])
    box(ax, 5.42, 1.86, 1.73, .62, "Full-bag CE\nCE(y, p)", color=GOLD)
    box(ax, 5.42, .62, 1.73, .62, "Consistency\nKL(stopgrad(p) || q)", color=GOLD, size=8.6)
    wire(ax, [(4.92, 1.86), (5.40, 1.86)])
    wire(ax, [(4.92, .62), (5.40, .62)])
    label(ax, 6.28, 2.38, "Slide label y")
    wire(ax, [(6.28, 2.26), (6.28, 2.19)])
    wire(ax, [(4.42, 1.53), (4.42, 1.16), (6.28, 1.16), (6.28, .95)], dashed=True)
    label(ax, 5.34, 1.31, "stop-gradient", 8.8)
    label(ax, 5.98, .12, "L = CE + KL", 10)
    save(fig, "architecture_consistency.png")


def spatial():
    fig, ax = canvas(3.3)
    box(ax, .04, 2.87, 1.19, .62, "Frozen features\nN × 384")
    box(ax, 1.52, 2.87, 1.30, .62, "Linear 384→256\nReLU", color=TEAL)
    wire(ax, [(1.25, 2.87), (1.50, 2.87)])
    wire(ax, [(2.84, 2.87), (3.06, 2.87)], arrow=False)
    ax.plot(3.06, 2.87, "o", color=INK, markersize=2.5)
    label(ax, 3.04, 3.10, r"$h_i$", 10)
    operation(ax, 5.84, 2.87, "+")
    wire(ax, [(3.06, 2.87), (5.71, 2.87)])
    label(ax, 4.42, 3.06, "identity", 8.8)
    box(ax, 6.23, 2.87, .92, .62, "ABMIL head\nattn + pool\n256→C", size=8.7)
    wire(ax, [(5.96, 2.87), (6.21, 2.87)])
    label(ax, 6.03, 3.10, r"$h'_i$", 10)
    box(ax, .04, .49, 1.31, .66, "Coordinates\n8-NN, no self")
    box(ax, 1.58, 1.66, 1.25, .62, "Neighbor mean\nm_i: 256", color=TEAL)
    wire(ax, [(.69, .84), (.69, 1.66), (1.56, 1.66)])
    wire(ax, [(3.06, 2.87), (3.06, 2.35), (2.20, 2.35), (2.20, 1.99)])
    label(ax, 1.76, 2.22, r"neighbors' $h_j$", 8.8)
    box(ax, 3.17, 1.66, .91, .62, "Concat\n[h_i; m_i]\n512", size=8.9)
    wire(ax, [(2.85, 1.66), (3.15, 1.66)])
    wire(ax, [(3.06, 2.35), (3.62, 2.35), (3.62, 1.99)])
    box(ax, 4.42, 1.66, 1.08, .76, "Context MLP\n512→256→256\nReLU between", size=8.7)
    box(ax, 4.42, .49, 1.08, .66, "Scalar gate\nLinear 512→1\nsigmoid", color=GOLD, size=8.7)
    wire(ax, [(4.10, 1.66), (4.40, 1.66)])
    wire(ax, [(4.24, 1.66), (4.24, .49), (4.40, .49)])
    operation(ax, 5.84, 1.66, "×")
    wire(ax, [(5.52, 1.66), (5.71, 1.66)])
    wire(ax, [(5.52, .49), (5.84, .49), (5.84, 1.53)])
    label(ax, 6.04, .94, r"$g_i$", 10)
    wire(ax, [(5.84, 1.79), (5.84, 2.74)])
    label(ax, 6.14, 2.22, r"$g_i f(c_i)$", 9)
    save(fig, "architecture_spatial.png")


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    baseline()
    consistency()
    spatial()


if __name__ == "__main__":
    main()
