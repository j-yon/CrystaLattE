"""Plot generation timings from ``benchmark_driver.py``, using the median of the repeats for each run

Only the orders passed to ``--N`` are drawn. The speedup panel stays empty until v1 timings exist::

    python scripts/plot_benchmark.py timings.jsonl --N 2 3 --out figure
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

TIME_LIMIT = 3600.0
PANEL_ORDERS = (2, 3, 4, 5)
HEATMAP_ORDERS = (2, 3, 4, 5)
MULTI_REFERENCE = {"pyrazole", "fomepizole", "phenol", "pentacene", "cholesterol"}

INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
V2 = "#2a78d6"

# name, atoms per molecule, symmetry operations
HEATMAP_CRYSTALS = [
    ("Carbon dioxide", 3, 24), ("Urea", 8, 8), ("1,3,5-Triazine", 9, 36), ("γ-Glycine", 10, 3),
    ("Trioxane", 12, 18), ("Benzene", 12, 8), ("L-Alanine", 13, 4), ("Ethyl carbamate", 13, 2),
    ("1,4-Cyclohexanedione", 16, 2), ("Aspirin", 21, 4), ("Pentaerythritol", 21, 8), ("Hexamine", 22, 48),
    ("Anthracene", 24, 4), ("Adamantane", 26, 8), ("β-HMX", 28, 4), ("Hexamethylbenzene", 30, 2),
    ("Ibuprofen", 33, 4), ("Coronene", 36, 4), ("Sucrose", 45, 2),
    ("Rubrene", 70, 16), ("Ritonavir", 98, 4),
]


def load_medians(path: Path, orders: list[int], exclude: list[str]) -> dict[int, list[tuple[str, int, float]]]:
    """Return ``(crystal, unique count, median seconds)`` per order, skipping unfinished runs

    :param path: JSON Lines file written by ``benchmark_driver.py``
    :param orders: Multimer orders to keep
    :param exclude: Crystal names to leave out of the plot; the file is not changed
    :returns: Points to plot, keyed by order
    """
    runs = defaultdict(list)
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record["crystal"] in exclude:
            continue
        if record["N"] in orders and record["status"] == "ok" and record["unique"] > 0:
            runs[record["crystal"], record["N"], record["R"]].append(record)

    points = defaultdict(list)
    for (name, N, _), repeats in runs.items():
        points[N].append((name, repeats[0]["unique"], statistics.median(r["seconds"] for r in repeats)))
    return points


def draw_scatter(fig, points: dict[int, list[tuple[str, int, float]]]) -> None:
    """Draw panel a, leaving orders without data empty"""
    grid = fig.add_gridspec(2, 2, left=0.068, right=0.50, top=0.945, bottom=0.138, wspace=0.05, hspace=0.05)
    facets = {N: fig.add_subplot(grid[i]) for i, N in enumerate(PANEL_ORDERS)}
    for i, (N, ax) in enumerate(facets.items()):
        for name, unique, seconds in points.get(N, []):
            hollow = name in MULTI_REFERENCE
            ax.scatter(unique, seconds, s=13 if hollow else 10, marker="D" if hollow else "o",
                       facecolors="none" if hollow else V2, edgecolors=V2 if hollow else "none",
                       linewidths=0.8 if hollow else 0, alpha=0.95 if hollow else 0.75, zorder=3)
        if N not in points:
            ax.text(0.5, 0.42, "not yet included", transform=ax.transAxes, ha="center", va="center", fontsize=9, color=INK2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(0.6, 6e6); ax.set_ylim(0.15, TIME_LIMIT * 2.4)
        ax.set_xticks([1, 100, 10000, 1000000], ["1", "10²", "10⁴", "10⁶"])
        ax.set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
        ax.minorticks_off()
        ax.tick_params(length=0, labelleft=i % 2 == 0, labelbottom=i >= 2, labelsize=8.5, pad=2)
        ax.axhline(TIME_LIMIT, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.grid(True, color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.text(0.05, 0.78, f"N={N}", transform=ax.transAxes, fontsize=10)
    facets[2].text(0.97, TIME_LIMIT * 0.62, "1 h", transform=facets[2].get_yaxis_transform(), ha="right", va="top", fontsize=8, color=INK2)
    facets[2].set_ylabel("Wall time (s)", fontsize=9, y=-0.025, labelpad=2)
    fig.text(0.284, 0.066, "Unique multimers", fontsize=9, color=INK2, ha="center")

    handles = [Line2D([], [], ls="", marker="o", mfc=V2, mec=V2, ms=4.5),
               Line2D([], [], ls="", marker="D", mfc="none", mec=V2, ms=4.5, mew=0.9)]
    fig.legend(handles, ["v2", "v2, >1 reference"], ncol=2, frameon=False, fontsize=8.2, loc="lower center",
               bbox_to_anchor=(0.284, 0.006), handletextpad=0.05, columnspacing=0.6, labelcolor=INK2, borderpad=0)


def draw_empty_heatmap(fig, exclude: list[str]) -> None:
    """Draw panel b's layout without values, since a speedup needs v1 timings"""
    crystals = [c for c in HEATMAP_CRYSTALS if c[0].lower().replace(" ", "_") not in exclude]
    ax = fig.add_axes([0.665, 0.103, 0.262, 0.84])
    ax.set_xlim(0, len(HEATMAP_ORDERS)); ax.set_ylim(len(crystals), 0)
    for i in range(len(crystals)):
        for j in range(len(HEATMAP_ORDERS)):
            ax.add_patch(Rectangle((j + 0.025, i + 0.05), 0.95, 0.90, facecolor=GRID, linewidth=0))
    ax.text(len(HEATMAP_ORDERS) / 2, len(crystals) / 2, "needs v1 timings", ha="center", va="center", fontsize=10, color=INK2)
    ax.set_xticks([j + 0.5 for j in range(len(HEATMAP_ORDERS))], [f"N={N}" for N in HEATMAP_ORDERS], fontsize=9.5, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.5 for i in range(len(crystals))], [c[0] for c in crystals], fontsize=8.8, color=INK)
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    xform = ax.get_yaxis_transform()
    for x, header, index in ((1.09, "atoms", 1), (1.215, "ops", 2)):
        ax.text(x, 1.006, header, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.3, color=INK2)
        for i, crystal in enumerate(crystals):
            ax.text(x, i + 0.5, str(crystal[index]), transform=xform, ha="center", va="center", fontsize=8.3, color=INK2)

    diverging = LinearSegmentedColormap.from_list("v", [(0, "#e34948"), (0.25, "#f0efec"), (0.55, "#86b6ef"), (0.8, "#2a78d6"), (1, "#0d366b")])
    cax = fig.add_axes([0.665, 0.058, 0.262, 0.016])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=-1, vmax=3), cmap=diverging), cax=cax, orientation="horizontal", ticks=[-1, 0, 1, 2, 3])
    bar.ax.set_xticklabels(["0.1×", "1×", "10×", "100×", "1k×"], fontsize=8.2, color=INK2)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, pad=1.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("timings", type=Path, help="JSON Lines results file")
    parser.add_argument("--N", nargs="+", type=int, required=True, dest="orders", help="orders to plot")
    parser.add_argument("--out", type=Path, required=True, help="output path without an extension")
    parser.add_argument("--exclude", nargs="*", default=[], help="crystal names to leave out of the plot")
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": ["Arial", "DejaVu Sans"], "font.size": 9,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    })
    fig = plt.figure(figsize=(8.4, 5.0))
    fig.text(0.008, 0.968, "a", fontsize=12, weight="bold")
    fig.text(0.515, 0.968, "b", fontsize=12, weight="bold")

    points = load_medians(args.timings, args.orders, args.exclude)
    draw_scatter(fig, points)
    draw_empty_heatmap(fig, args.exclude)

    fig.savefig(f"{args.out}.png", dpi=220, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight", pad_inches=0.05)
    print({N: len(p) for N, p in sorted(points.items())})


if __name__ == "__main__":
    main()
