"""Plot the learning curves of a 24 game RLCD run.

Figure 1 is generation level: the controller's dev success rate against the measured
random baselines, the frozen-policy event rates the labels come from, calibration, and
the paired-PG diagnostics with the policy entropy.

Figure 2 is step level: loss, gradient norm, advantage, collision rate and the
prediction-versus-observation tracking, with the generation boundaries marked.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

RANDOM_PLAY = 0.0060  # uniform random three-step play, measured on the dev split
RANDOM_FIRST_SOLVABLE = 0.2340  # chance level of keeping the puzzle solvable
RANDOM_CONDITIONAL = 0.0880  # solve rate after a solvable-preserving first step


def load(run):
    generations = json.loads((Path(run) / "generations.json").read_text())
    steps = [
        json.loads(line)
        for line in (Path(run) / "training.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return generations, steps


def series(generations, source, *path):
    values = []
    for summary in generations:
        value = summary[source]
        for key in path:
            value = value[key]
        values.append(float(value))
    return np.array(values)


def rolling(values, window):
    if len(values) < window:
        return np.array([]), np.array([])
    kernel = np.ones(window) / window
    return np.arange(window - 1, len(values)), np.convolve(values, kernel, mode="valid")


def generation_figure(run, generations, out_path):
    figure = Figure(figsize=(16, 9))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3)
    gen = series(generations, "rollout", "generation")

    ax = axes[0, 0]
    controller = series(generations, "dev_controller", "solved_rate")
    first_ok = series(generations, "dev_controller", "first_action_solvable_rate")
    conditional = np.divide(controller, first_ok, out=np.zeros_like(controller), where=first_ok > 0)
    ax.plot(gen, controller, "-", color="tab:blue", label="greedy controller solves")
    ax.plot(gen, first_ok, "--", color="tab:cyan", label="first action keeps solvable")
    ax.plot(gen, conditional, "-.", color="tab:green", label="solves | first action ok")
    ax.axhline(RANDOM_PLAY, color="gray", ls=":", label=f"random play {RANDOM_PLAY:.4f}")
    ax.axhline(RANDOM_FIRST_SOLVABLE, color="gray", ls="-.", label=f"chance first step {RANDOM_FIRST_SOLVABLE:.3f}")
    ax.axhline(RANDOM_CONDITIONAL / RANDOM_FIRST_SOLVABLE, color="gray", ls="--", label="random | first action ok")
    ax.set(title="Controller on the 204 dev puzzles", xlabel="generation", ylabel="rate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[0, 1]
    ax.plot(gen, series(generations, "rollout", "solved_rate"), "-", color="tab:blue", label="collection event rate")
    ax.plot(gen, series(generations, "frozen_dev_rollout", "solved_rate"), "--", color="tab:orange", label="frozen policy on dev")
    ax.plot(gen, series(generations, "rollout", "mean_p_yes"), ":", color="tab:green", label="model prediction (collection)")
    ax.plot(gen, series(generations, "dev_calibration", "predicted_success_probability"), "-.", color="tab:red", label="model prediction (dev)")
    ax.set(title="Frozen-policy success event", xlabel="generation", ylabel="rate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    rate = series(generations, "dev_calibration", "yes_rate")
    ax.plot(gen, series(generations, "dev_calibration", "observed_nll"), "-", color="tab:blue", label="NLL")
    ax.plot(gen, series(generations, "dev_calibration", "binary_brier"), "--", color="tab:orange", label="binary Brier")
    ax.plot(gen, series(generations, "dev_calibration", "ece_ge_correct"), "-.", color="tab:green", label="ECE")
    ax.plot(gen, rate * (1 - rate), ":", color="black", label="constant-predictor Brier")
    ax.set(title="Dev calibration (moves with the event rate)", xlabel="generation", ylabel="value")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(gen, series(generations, "mean_gradient_norm"), "-", color="tab:blue", label="gradient norm")
    ax.set(title="Paired-PG signal per generation", xlabel="generation", ylabel="gradient norm")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(gen, series(generations, "rollout", "mean_action_entropy"), "--", color="tab:orange", label="action entropy")
    ax2.set_ylabel("action entropy")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], fontsize=8)

    ax = axes[1, 1]
    ax.plot(gen, series(generations, "rollout", "row_yes_rate"), "-", color="tab:blue", label="observed event rate (train)")
    ax.plot(gen, series(generations, "rollout", "mean_p_yes"), "--", color="tab:orange", label="prediction (train)")
    ax.plot(gen, series(generations, "dev_calibration", "yes_rate"), "-.", color="tab:green", label="observed (dev)")
    ax.plot(gen, series(generations, "dev_calibration", "predicted_success_probability"), ":", color="tab:red", label="prediction (dev)")
    ax.set(title="Calibration tracking", xlabel="generation", ylabel="rate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.bar(gen - 0.2, series(generations, "rollout", "unique_state_actions"), width=0.4, label="unique (state, action)")
    ax.bar(gen + 0.2, series(generations, "rollout", "repeated_state_action_events"), width=0.4, label="repeated observations")
    ax.set(title="Training rows per generation", xlabel="generation", ylabel="rows")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(out_path, dpi=200)
    return dict(gen=gen, controller=controller, first_ok=first_ok, conditional=conditional)


def step_figure(run, steps, out_path):
    figure = Figure(figsize=(16, 9))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3)
    index = np.array([record["step"] for record in steps])
    train = [record["train"] for record in steps]
    curves = [
        (axes[0, 0], "loss", np.array([record["loss"] for record in steps])),
        (axes[0, 1], "gradient norm", np.array([record["gradient_norm"] for record in steps])),
        (axes[0, 2], "advantage |mean|", np.array([row["advantage_abs_mean"] for row in train])),
        (axes[1, 0], "collision rate", np.array([row["sampled_collision_rate"] for row in train])),
    ]
    for axis, title, values in curves:
        axis.plot(index, values, lw=0.7, alpha=0.4, color="tab:blue")
        smoothed_index, smoothed = rolling(values, 30)
        if len(smoothed):
            axis.plot(index[smoothed_index], smoothed, lw=2, color="tab:red", label="30-step mean")
        axis.set(title=title, xlabel="optimizer step")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    axis = axes[1, 1]
    axis.plot(index, [row["predicted_success_probability"] for row in train], lw=0.7, alpha=0.5, label="predicted")
    axis.plot(index, [row["observed_success_rate"] for row in train], lw=0.7, alpha=0.5, label="observed")
    axis.set(title="predicted vs observed event rate", xlabel="optimizer step", ylabel="rate")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)

    axis = axes[1, 2]
    axis.plot(index, [row["top1_accuracy"] for row in train], lw=0.7, alpha=0.6, label="top1 accuracy")
    axis.plot(index, [row["entropy"] for row in train], lw=0.7, alpha=0.6, label="prediction entropy")
    axis.set(title="top1 accuracy and prediction entropy", xlabel="optimizer step", ylabel="value")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)

    generations = sorted({record["generation"] for record in steps})
    for generation in generations[1:]:
        boundary = min(record["step"] for record in steps if record["generation"] == generation)
        for axis, _, _ in curves:
            axis.axvline(boundary, color="gray", alpha=0.4, lw=0.8)
        for axis in (axes[1, 1], axes[1, 2]):
            axis.axvline(boundary, color="gray", alpha=0.4, lw=0.8)
    figure.tight_layout()
    figure.savefig(out_path, dpi=200)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/paired_pg_24game_seed17")
    parser.add_argument("--out-dir", default="results/plots")
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    name = Path(args.run).name
    generations, steps = load(args.run)
    generation_path = Path(args.out_dir) / f"{name}_generations.png"
    step_path = Path(args.out_dir) / f"{name}_steps.png"
    curves = generation_figure(args.run, generations, generation_path)
    if steps:
        step_figure(args.run, steps, step_path)
    rows = [
        dict(
            generation=int(generation),
            controller=float(controller),
            first_action_solvable=float(first_ok),
            solves_given_first_ok=float(conditional),
            collect_rate=float(summary["rollout"]["solved_rate"]),
            frozen_dev=float(summary["frozen_dev_rollout"]["solved_rate"]),
            nll=float(summary["dev_calibration"]["observed_nll"]),
            binary_brier=float(summary["dev_calibration"]["binary_brier"]),
            ece=float(summary["dev_calibration"]["ece_ge_correct"]),
            action_entropy=float(summary["rollout"]["mean_action_entropy"]),
            gradient_norm=float(summary["mean_gradient_norm"]),
        )
        for summary, generation, controller, first_ok, conditional in zip(
            generations,
            curves["gen"],
            curves["controller"],
            curves["first_ok"],
            curves["conditional"],
        )
    ]
    curves_path = Path(args.out_dir) / f"{name}_curves.json"
    curves_path.write_text(
        json.dumps(
            dict(run=args.run, generations=len(generations), steps=len(steps), curves=rows),
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            dict(figures=[str(generation_path), str(step_path)], data=str(curves_path), rows=rows),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
