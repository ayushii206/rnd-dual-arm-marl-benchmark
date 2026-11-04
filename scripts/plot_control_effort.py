"""Plot baseline mechanical control-effort comparisons."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT = Path(
    "results/comparison/control_effort_baseline_5cm.csv"
)

OUTPUT_DIR = Path(
    "figures/control_effort"
)


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)

    assert len(df) == 7, "Expected seven controller evaluations"
    assert df["success"].sum() == 4, "Expected four successes"

    df = df[df["success"]].copy()

    labels = [
        "Classical",
        "IPPO 123",
        "MAPPO 123",
        "MAPPO 456",
    ]

    df["label"] = df.apply(
        lambda row: (
            "Classical"
            if row["algorithm"] == "Classical"
            else f"{row['algorithm']} {int(row['seed'])}"
        ),
        axis=1,
    )

    df = df.set_index("label").loc[labels]

    colors = [
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#54A24B",
    ]

    metrics = [
        ("absolute_work_j", "Accumulated absolute work (J)"),
        ("mean_absolute_power_w", "Mean absolute power (W)"),
    ]

    output_png = OUTPUT_DIR / "baseline_control_effort.png"
    output_pdf = OUTPUT_DIR / "baseline_control_effort.pdf"

    for output in (output_png, output_pdf):
        if output.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing figure: {output}"
            )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, (column, title) in zip(axes, metrics):
        values = df[column].to_numpy(dtype=float)
        bars = ax.bar(
            labels,
            values,
            color=colors,
            edgecolor="#333333",
            linewidth=0.7,
        )

        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.3f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(values) * 1.22)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Baseline Mechanical Control Effort — Successful Policies",
        fontsize=14,
        fontweight="bold",
    )

    fig.text(
        0.5, 0.015,
        "2 kg payload | 5 cm target | "
        "Post-step torque–velocity estimate; excludes grippers. "
        "Not electrical energy.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.93])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_png, dpi=300)
    fig.savefig(output_pdf)
    plt.close(fig)

    print("PASS: Control-effort figure generated.")
    print("PNG:", output_png)
    print("PDF:", output_pdf)


if __name__ == "__main__":
    main()
