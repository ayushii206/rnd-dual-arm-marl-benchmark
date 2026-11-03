
"""Generate reproducible dual-arm transport benchmark figures."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT = Path("figures/benchmark_comparison_v3")

COLORS = {
    "Classical": "#2563eb",
    "IPPO": "#ea580c",
    "MAPPO": "#16a34a",
}

def policy_color(policy):
    return COLORS[policy.split()[0]]


DATASETS = {
    "mass": (
        "results/robustness/mass_robustness_5cm.csv",
        "mass_kg",
        [2.0, 3.0, 4.0],
        "Object mass (kg)",
    ),
    "friction": (
        "results/robustness/friction_robustness_5cm.csv",
        "friction_multiplier",
        [1.0, 0.75, 0.5],
        "Sliding friction multiplier",
    ),
    "observation_noise": (
        "results/robustness/observation_noise_5cm.csv",
        "noise_std_mm",
        [0.0, 2.0, 5.0],
        "Position noise standard deviation (mm)",
    ),
    "action_disturbance": (
        "results/robustness/action_disturbance_5cm.csv",
        "disturbance_std",
        [0.0, 0.05, 0.10],
        "Normalized action-noise standard deviation",
    ),
}

POLICIES = [
    "Classical",
    "IPPO 42",
    "IPPO 123",
    "IPPO 456",
    "MAPPO 42",
    "MAPPO 123",
    "MAPPO 456",
]

SUCCESSFUL = [
    "Classical",
    "IPPO 123",
    "MAPPO 123",
    "MAPPO 456",
]


def load_data():
    datasets = {}

    for name, (path, parameter, levels, _) in DATASETS.items():
        df = pd.read_csv(path)

        assert len(df) == 21, f"{name}: expected 21 rows"

        df["policy"] = df.apply(
            lambda row: (
                "Classical"
                if row["algorithm"] == "Classical"
                else f"{row['algorithm']} {int(row['seed'])}"
            ),
            axis=1,
        )

        assert len(df[["policy", parameter]].drop_duplicates()) == 21
        assert set(df["policy"]) == set(POLICIES)
        assert set(df[parameter].astype(float)) == set(levels)

        datasets[name] = df

    return datasets


def save_figure(fig, name):
    fig.tight_layout()

    for extension in ("png", "pdf"):
        destination = OUTPUT / f"{name}.{extension}"

        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite {destination}"
            )

        fig.savefig(
            destination,
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)
    print(f"Saved {name}.png and {name}.pdf")


def baseline_figures(df):
    baseline = (
        df[df["mass_kg"] == 2.0]
        .set_index("policy")
        .loc[POLICIES]
    )

    # Figure 1: Success by controller family.
    success_counts = [
        int(baseline.loc[["Classical"], "success"].sum()),
        int(baseline.loc[
            ["IPPO 42", "IPPO 123", "IPPO 456"],
            "success",
        ].sum()),
        int(baseline.loc[
            ["MAPPO 42", "MAPPO 123", "MAPPO 456"],
            "success",
        ].sum()),
    ]

    totals = [1, 3, 3]
    algorithms = ["Classical", "IPPO", "MAPPO"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        algorithms,
        [100 * n / d for n, d in zip(success_counts, totals)],
        color=[COLORS[a] for a in algorithms],
    )

    for bar, n, d in zip(bars, success_counts, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{n}/{d}",
            ha="center",
        )

    ax.set_ylim(0, 112)
    ax.set_ylabel("Successful policies (%)")
    ax.set_title("Baseline 5 cm transport success")
    ax.text(
        0.5, -0.15,
        "Classical: one controller; IPPO/MAPPO: three training seeds each",
        transform=ax.transAxes,
        ha="center", fontsize=9,
    )
    save_figure(fig, "01_baseline_success")

    # Figure 2: Completion steps, successful policies only.
    successful = baseline.loc[SUCCESSFUL]

    assert successful["success"].all()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        SUCCESSFUL,
        successful["rl_steps"].to_numpy(),
        color=[policy_color(p) for p in SUCCESSFUL],
    )
    ax.bar_label(bars, padding=4)
    ax.set_ylim(0, 370)
    ax.set_ylabel("Control steps to success")
    ax.set_title("Completion steps of successful controllers")
    ax.tick_params(axis="x", rotation=20)
    save_figure(fig, "02_baseline_completion_steps")

    # Figure 3: Maximum tilt, including failed trajectories.
    fig, ax = plt.subplots(figsize=(9, 4.5))

    positions = np.arange(len(POLICIES))
    tilt = baseline["max_tilt_deg"].to_numpy()

    ax.bar(
        positions, tilt,
        color=[policy_color(p) for p in POLICIES],
        hatch=[
            "///" if not bool(baseline.loc[p, "success"]) else ""
            for p in POLICIES
        ],
    )

    for i, policy in enumerate(POLICIES):
        if not bool(baseline.loc[policy, "success"]):
            ax.text(
                i,
                tilt[i] + 0.3,
                "Failed",
                ha="center",
                fontsize=8,
            )

    ax.set_xticks(positions, POLICIES, rotation=25)
    ax.set_ylabel("Maximum object tilt (degrees)")
    ax.set_title("Object stability during baseline evaluation")
    ax.set_ylim(0, max(tilt) * 1.2)
    save_figure(fig, "03_baseline_max_tilt")

    # Figure 4: Bilateral grasp-contact percentage.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(
        POLICIES,
        baseline["both_grasp_contact_pct"].to_numpy(),
        color=[policy_color(p) for p in POLICIES],
        hatch=[
            "///" if not bool(baseline.loc[p, "success"]) else ""
            for p in POLICIES
        ],
    )
    ax.set_ylim(0, 110)
    ax.set_ylabel("Bilateral grasp-contact time (%)")
    ax.set_title("Grasp maintenance during baseline evaluation")
    ax.text(
        0.5, -0.23,
        "Hatched bars: unsuccessful transport; percentage of monitored physics steps",
        transform=ax.transAxes,
        ha="center", fontsize=9,
    )
    ax.tick_params(axis="x", rotation=25)
    save_figure(fig, "04_baseline_grasp_contact")


def robustness_figure(name, df, parameter, levels, xlabel):
    fig, ax = plt.subplots(figsize=(8, 5))

    for policy in SUCCESSFUL:
        subset = df[df["policy"] == policy].set_index(parameter)

        subset = subset.loc[levels]

        if not subset["success"].all():
            raise ValueError(
                f"{policy} failed a {name} condition. "
                "Revise plot before interpreting completion time."
            )

        ax.plot(
            levels,
            subset["rl_steps"].to_numpy(),
            marker="s" if policy == "MAPPO 456" else "o",
            linestyle="--" if policy == "MAPPO 456" else "-",
            linewidth=2,
            color=policy_color(policy),
            label=policy,
        )

    ax.set_xticks(levels)
    if name == "friction":
        ax.invert_xaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Control steps to successful transport")
    ax.set_title(
        name.replace("_", " ").title()
        + " — successful policies"
    )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
    )
    ax.grid(alpha=0.25)

    figure_numbers = {
        "mass": "05",
        "friction": "06",
        "observation_noise": "07",
        "action_disturbance": "08",
    }
    number = figure_numbers[name]
    save_figure(fig, f"{number}_{name}_completion")


def main():
    datasets = load_data()

    # Protect existing figures before generating anything.
    expected_names = [
        "01_baseline_success",
        "02_baseline_completion_steps",
        "03_baseline_max_tilt",
        "04_baseline_grasp_contact",
        *[
            f"{number}_{name}_completion"
            for number, name in (
                ("05", "mass"),
                ("06", "friction"),
                ("07", "observation_noise"),
                ("08", "action_disturbance"),
            )
        ],
    ]

    for name in expected_names:
        for extension in ("png", "pdf"):
            destination = OUTPUT / f"{name}.{extension}"
            if destination.exists():
                raise FileExistsError(
                    f"Existing figure: {destination}"
                )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    baseline_figures(datasets["mass"])

    for name, (_, parameter, levels, xlabel) in DATASETS.items():
        robustness_figure(
            name,
            datasets[name],
            parameter,
            levels,
            xlabel,
        )

    print("\nPASS: Generated eight figures in PNG and PDF formats.")


if __name__ == "__main__":
    main()
