"""Plot transport-distance generalization results."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT = Path("results/generalization/distance_generalization.csv")
OUTPUT = Path("figures/distance_generalization")

COLORS = {
    "Classical": "#4C78A8",
    "IPPO": "#F58518",
    "MAPPO": "#54A24B",
}

def main():
    df = pd.read_csv(INPUT)
    assert len(df) == 28
    assert set(df["evaluation_distance_cm"]) == {3., 5., 7., 10.}

    png = OUTPUT / "distance_generalization_success.png"
    pdf = OUTPUT / "distance_generalization_success.pdf"

    for file in (png, pdf):
        if file.exists():
            raise FileExistsError(f"Refusing to overwrite {file}")

    distances = [3., 5., 7., 10.]
    algorithms = ["Classical", "IPPO", "MAPPO"]
    x = np.arange(len(distances))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, algorithm in enumerate(algorithms):
        counts = [
            int(df[
                (df["algorithm"] == algorithm) &
                (df["evaluation_distance_cm"] == distance)
            ]["success"].sum())
            for distance in distances
        ]
        total = 1 if algorithm == "Classical" else 3
        bars = ax.bar(
            x + (i - 1) * width,
            counts,
            width,
            label=algorithm,
            color=COLORS[algorithm],
            edgecolor="#333333",
            linewidth=0.6,
        )
        for bar, count in zip(bars, counts):
            ax.annotate(
                f"{count}/{total}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )

    ax.set_xticks(x, [f"{d:g} cm" for d in distances])
    ax.set_ylabel("Successful controller configurations")
    ax.set_xlabel("Evaluation target distance")
    ax.set_ylim(0, 3.55)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_title("Transport-Distance Generalization", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.text(
        0.5, 0.025,
        "RL policies trained at 5 cm | 2 kg payload | "
        "10 mm position tolerance and other success criteria",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.055, 1, 1])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

    print("PASS: Distance-generalization figure generated.")
    print("PNG:", png)
    print("PDF:", pdf)

if __name__ == "__main__":
    main()
