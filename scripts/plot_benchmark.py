"""Plot generation timings from ``benchmark_driver.py``, using the median of the repeats for each run

Only the orders passed to ``--N`` are drawn. Speedup cells stay blank where a code has no timing::

    python scripts/plot_benchmark.py timings.jsonl --v1 v1_timings.jsonl --N 2 3 4 5 --out figure
"""

import argparse
import json
import math
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
V1, V2 = "#eb6834", "#2a78d6"
DIVERGING = LinearSegmentedColormap.from_list("v", [(0, "#e34948"), (0.25, "#f0efec"), (0.55, "#86b6ef"), (0.8, "#2a78d6"), (1, "#0d366b")])
SPEEDUP_NORM = Normalize(vmin=-1, vmax=3)

# name, results-file key, atoms per molecule (row order), space group as written in the CIF used
HEATMAP_CRYSTALS = [
    ("Carbon dioxide", "carbon_dioxide", 3, r"Pa\bar{3}"), ("Urea", "urea", 8, r"P\bar{4}2_1m"),
    ("1,3,5-Triazine", "triazine", 9, r"R\bar{3}c"), ("γ-Glycine", "gamma_glycine", 10, r"P3_2"),
    ("Trioxane", "trioxane", 12, r"R3c"), ("Benzene", "benzene", 12, r"Pbca"),
    ("L-Alanine", "l_alanine", 13, r"P2_12_12_1"), ("Ethyl carbamate", "ethyl_carbamate", 13, r"P\bar{1}"),
    ("1,4-Cyclohexanedione", "14_cyclohexanedione", 16, r"P2_1"), ("Aspirin", "aspirin", 21, r"P2_1/c"),
    ("Pentaerythritol", "pentaerythritol", 21, r"I\bar{4}"), ("Hexamine", "hexamine", 22, r"I\bar{4}3m"),
    ("Anthracene", "anthracene", 24, r"P2_1/a"), ("Adamantane", "adamantane", 26, r"P\bar{4}2_1c"),
    ("β-HMX", "beta_hmx", 28, r"P2_1/n"), ("Hexamethylbenzene", "hexamethylbenzene", 30, r"P\bar{1}"),
    ("Ibuprofen", "ibuprofen", 33, r"P2_1/c"), ("Coronene", "coronene", 36, r"P2_1/n"),
    ("Sucrose", "sucrose", 45, r"P2_1"), ("Rubrene", "rubrene", 70, r"Cmca"),
    ("Ritonavir", "ritonavir_c2", 98, r"C2"),
]


def load_runs(path: Path, exclude: list[str]) -> tuple[dict, set]:
    """Return finished runs and timed-out runs, keyed by ``(crystal, N, R)``

    A run's summed multiplicity is its candidate count, since every candidate joins one unique multimer.

    :param path: JSON Lines file written by ``benchmark_driver.py``
    :param exclude: Crystal names to leave out of the plot; the file is not changed
    :returns: ``(unique, candidates, median seconds)`` per finished run, and the keys of timed-out runs
    """
    repeats, timed_out = defaultdict(list), set()
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record["crystal"] in exclude:
            continue
        key = record["crystal"], record["N"], record["R"]
        if record["status"] == "ok":
            repeats[key].append(record)
        elif record["status"] == "timeout":
            timed_out.add(key)
    runs = {
        key: (rows[0]["unique"], rows[0]["multiplicity"], statistics.median(r["seconds"] for r in rows))
        for key, rows in repeats.items()
    }
    return runs, timed_out


def draw_scatter(fig, v2: dict, v1: dict, v1_timed_out: set, orders: list[int]) -> None:
    """Draw panel a against v2's candidate count, leaving orders without data empty"""
    grid = fig.add_gridspec(2, 2, left=0.062, right=0.475, top=0.945, bottom=0.138, wspace=0.04, hspace=0.04)
    facets = {N: fig.add_subplot(grid[i]) for i, N in enumerate(PANEL_ORDERS)}

    differs_any = False
    for (name, N, R), (unique, candidates, seconds) in v2.items():
        if N not in orders or unique == 0:
            continue
        ax = facets[N]
        hollow = name in MULTI_REFERENCE
        ax.scatter(candidates, seconds, s=13 if hollow else 10, marker="D" if hollow else "o",
                   facecolors="none" if hollow else V2, edgecolors=V2 if hollow else "none",
                   linewidths=0.8 if hollow else 0, alpha=0.95 if hollow else 0.75, zorder=3)
        key = (name, N, R)
        if key in v1:
            v1_unique, _, v1_seconds = v1[key]
            differs = v1_unique != unique
            differs_any |= differs
            if differs:
                ax.scatter(candidates, v1_seconds, s=14, marker="x", color=V1, linewidths=0.9, zorder=4)
            else:
                ax.scatter(candidates, v1_seconds, s=10, color=V1, linewidths=0, alpha=0.75, zorder=2)
        elif key in v1_timed_out:
            ax.scatter(candidates, TIME_LIMIT * 1.5, s=16, marker="^", color=V1, linewidths=0, zorder=4, clip_on=False)

    for i, (N, ax) in enumerate(facets.items()):
        if N not in orders:
            ax.text(0.5, 0.42, "not yet included", transform=ax.transAxes, ha="center", va="center", fontsize=9, color=INK2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(0.6, 5e7); ax.set_ylim(0.15, TIME_LIMIT * 2.4)
        ax.set_xticks([1, 100, 10000, 1000000], ["1", "10²", "10⁴", "10⁶"])
        ax.set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
        ax.minorticks_off()
        ax.tick_params(length=0, labelleft=i % 2 == 0, labelbottom=i >= 2, labelsize=9.5, pad=2)
        ax.axhline(TIME_LIMIT, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.grid(True, color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.text(0.05, 0.78, f"N={N}", transform=ax.transAxes, fontsize=11)
    facets[2].text(0.97, TIME_LIMIT * 0.62, "1 h", transform=facets[2].get_yaxis_transform(), ha="right", va="top", fontsize=8, color=INK2)
    facets[2].set_ylabel("Wall time (s)", fontsize=10, y=-0.025, labelpad=2)
    fig.text(0.268, 0.068, "Total $N$-mer candidates", fontsize=10, color=INK2, ha="center")

    entries = []
    if v1:
        entries.append((Line2D([], [], ls="", marker="o", mfc=V1, mec=V1, ms=4.5), "v1"))
    if differs_any:
        entries.append((Line2D([], [], ls="", marker="x", mec=V1, ms=4.5, mew=0.9), "v1 count differs"))
    if v1_timed_out:
        entries.append((Line2D([], [], ls="", marker="^", mfc=V1, mec=V1, ms=4.5), "v1 timed out"))
    entries.append((Line2D([], [], ls="", marker="o", mfc=V2, mec=V2, ms=4.5), "v2"))
    entries.append((Line2D([], [], ls="", marker="D", mfc="none", mec=V2, ms=4.5, mew=0.9), "v2, Z″>1"))
    fig.legend(*zip(*entries), ncol=len(entries), frameon=False, fontsize=9.5, loc="lower center",
               bbox_to_anchor=(0.268, 0.012), handletextpad=0.05, columnspacing=0.6, labelcolor=INK2, borderpad=0)


def draw_heatmap(fig, v2: dict, v1: dict, v1_timed_out: set, cutoffs: list[float], exclude: list[str]) -> None:
    """Draw panel b at one cutoff per order, leaving cells without both timings blank"""
    crystals = [c for c in HEATMAP_CRYSTALS if c[1] not in exclude]
    ax = fig.add_axes([0.685, 0.138, 0.305, 0.807])
    ax.set_xlim(0, len(HEATMAP_ORDERS)); ax.set_ylim(len(crystals), 0)
    for i, (_, key, *_rest) in enumerate(crystals):
        for j, (N, R) in enumerate(zip(HEATMAP_ORDERS, cutoffs)):
            run = key, N, R
            color, label = GRID, None
            if run in v2 and (run in v1 or run in v1_timed_out):
                timed_out = run not in v1
                value = (TIME_LIMIT if timed_out else v1[run][2]) / v2[run][2]
                shade = SPEEDUP_NORM(math.log10(value))
                color = DIVERGING(shade)
                text = f"{value / 1000:.0f}k" if value >= 1000 else f"{value:.0f}" if value >= 10 else f"{value:.1f}"
                label = (">" if timed_out else "") + text + "×"
            ax.add_patch(Rectangle((j + 0.025, i + 0.05), 0.95, 0.90, facecolor=color, linewidth=0))
            if label:
                ax.text(j + 0.5, i + 0.52, label, ha="center", va="center", fontsize=9.3, color=SURFACE if shade > 0.68 else INK)
    ax.set_xticks([j + 0.5 for j in range(len(HEATMAP_ORDERS))], [f"N={N}" for N in HEATMAP_ORDERS], fontsize=10.5, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.5 for i in range(len(crystals))], [f"{c[0]} (${c[3]}$)" for c in crystals], fontsize=10, color=INK)
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cax = fig.add_axes([0.685, 0.082, 0.305, 0.016])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=SPEEDUP_NORM, cmap=DIVERGING), cax=cax, orientation="horizontal", ticks=[-1, 0, 1, 2, 3])
    bar.ax.set_xticklabels(["0.1×", "1×", "10×", "100×", "1k×"], fontsize=9.5, color=INK2)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, pad=1.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("timings", type=Path, help="v2 JSON Lines results file")
    parser.add_argument("--v1", type=Path, help="v1 JSON Lines results file")
    parser.add_argument("--N", nargs="+", type=int, required=True, dest="orders", help="orders to plot")
    parser.add_argument("--out", type=Path, required=True, help="output path without an extension")
    parser.add_argument("--exclude", nargs="*", default=[], help="crystal names to leave out of the plot")
    parser.add_argument("--heatmap-R", nargs=4, type=float, default=[30, 15, 10, 8], help="speedup cutoff for N=2 to 5")
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": ["Times New Roman", "DejaVu Serif"], "font.size": 9.5,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
    })
    fig = plt.figure(figsize=(8.4, 5.0))
    fig.text(0.008, 0.968, "a", fontsize=12, weight="bold")
    fig.text(0.49, 0.968, "b", fontsize=12, weight="bold")

    v2, _ = load_runs(args.timings, args.exclude)
    v1, v1_timed_out = load_runs(args.v1, args.exclude) if args.v1 else ({}, set())
    draw_scatter(fig, v2, v1, v1_timed_out, args.orders)
    draw_heatmap(fig, v2, v1, v1_timed_out, args.heatmap_R, args.exclude)

    fig.savefig(f"{args.out}.png", dpi=220, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight", pad_inches=0.05)
    print({"v2 runs": len(v2), "v1 runs": len(v1), "v1 timed out": len(v1_timed_out)})


if __name__ == "__main__":
    main()
