"""Shared visual system for saved-result charts and the study schematic."""
from pathlib import Path
import matplotlib.pyplot as plt

INK = "#203b37"
TEAL = "#176c60"
BLUE = "#326c9f"
ORANGE = "#b05c20"
MODEL_COLORS = {
    "s2_weighted": "#527d45", "s1_weighted": "#76658c",
    "early_fusion": "#647a82", "modality_dropout": BLUE,
    "occlusion_trained": ORANGE, "terramind_frozen": TEAL,
}

CLASS_COLORS = ["#8b9693", "#86bfa5", "#388d78", "#164f49"]

def configure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "figure.titlesize": 16,
        "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "text.color": INK, "axes.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.5, "lines.markersize": 6,
        "legend.frameon": False, "legend.fontsize": 10,
        "svg.fonttype": "none", "svg.hashsalt": "geoai-release",
        "savefig.dpi": 220,
    })

def save(fig, root: Path, name: str):
    path = root / "figures" / (name + ".svg")
    fig.savefig(path, facecolor="white", metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").replace("'DejaVu Sans'", "'DejaVu Sans', Arial, sans-serif").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)

def grid(ax, axis="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color="#e2e8e3", linewidth=.7)
    ax.tick_params(length=0, pad=8)
