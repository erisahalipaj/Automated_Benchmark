import argparse
import json
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — safe for all terminals
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
#  CONFIGURATION & DATA LOADING
# ============================================================

# Define base directory path
BASE_PATH = Path(__file__).parent

# Load models and device profiles from JSON
with open(BASE_PATH / "models.json", "r") as f:
    models = json.load(f)

with open(BASE_PATH / "device_profiles.json", "r") as f:
    devices = json.load(f)

# Output directory for charts
ANALYTICS_DIR = BASE_PATH / "analytics"
ANALYTICS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
#  ENTROPY WEIGHT HELPER
# ------------------------------------------------------------


def compute_entropy_weights(matrix: np.ndarray) -> np.ndarray:
    """
    Compute Entropy Weight Method weights for a 2-D matrix
    where rows = alternatives, columns = criteria.
    Returns a 1-D weight array summing to 1.
    """
    eps = 1e-12
    P = matrix / (matrix.sum(axis=0) + eps)
    E = -np.nansum(P * np.log(P + eps), axis=0) / np.log(len(matrix))
    d = 1 - E
    return d / d.sum()


# ============================================================
#  CORE FUNCTIONS
# ============================================================


def is_feasible(model: dict, device: dict) -> bool:
    """
    Check whether a model satisfies device feasibility constraints.
    Energy is excluded from feasibility checks — not measurable on all devices.
    """
    return all(
        [
            model["mem_req"] <= device["RAM"],
            model["size"] <= device["STO"],
            model["cpu_req"] <= device["CPU"],
        ]
    )


def compute_selection_score(feasible: list[dict]) -> list[float]:
    """
    Compute the pre-selection suitability score (S_pre) for all feasible models
    in one pass, using entropy-based weights derived across the whole set.
    Returns scores in the same order as the input list.
    """
    matrix = np.array([[1 / m["mem_req"], 1 / m["cpu_req"], m["quality"], 1 / m["size"]] for m in feasible])
    weights = compute_entropy_weights(matrix)
    keys = ["mem", "spd", "qual", "foot"]
    print("[INFO] Pre-selection weights (EWM):", {k: round(float(w), 3) for k, w in zip(keys, weights)})
    scores = matrix @ weights
    return [round(float(s), 3) for s in scores]


def normalize_metric(values, reverse=False):
    """
    Normalize a list of metric values to 0–1 range.
    If reverse=True, smaller values are better (e.g. latency).
    """
    v_min, v_max = min(values), max(values)
    if v_max == v_min:
        return [1 for _ in values]  # Prevent division by zero
    if reverse:
        return [1 - ((v - v_min) / (v_max - v_min)) for v in values]
    return [(v - v_min) / (v_max - v_min) for v in values]


def compute_ess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Edge Suitability Score (ESS) for all models
    using normalized performance metrics (latency, throughput, memory, quality),
    and automatically calculate weights using the Entropy Weight Method.
    """
    # Normalize metrics
    df["L'"] = normalize_metric(df["latency"], reverse=True)
    df["R'"] = normalize_metric(df["throughput"], reverse=False)
    df["M'"] = normalize_metric(df["memory"], reverse=True)
    df["Q'"] = normalize_metric(df["quality"], reverse=False)

    # Entropy Weight Method — compute once across all feasible models
    norm_metrics = ["L'", "R'", "M'", "Q'"]
    X = df[norm_metrics].values
    weights = compute_entropy_weights(X)
    WEIGHTS_ESS = dict(zip(["L", "R", "M", "Q"], weights))
    print("[INFO] ESS weights (EWM):", {k: round(float(w), 3) for k, w in WEIGHTS_ESS.items()})

    # Compute ESS
    df["ESS"] = np.round(X @ weights, 3)

    # Z-score classification
    ess_mean = df["ESS"].mean()
    ess_std = df["ESS"].std(ddof=0)

    def classify_ess(val):
        if ess_std == 0:
            return "partially suitable"
        z = (val - ess_mean) / ess_std
        if z >= 0.75:
            return "suitable"
        elif z <= -0.75:
            return "unsuitable"
        return "partially suitable"

    df["ESS_class"] = df["ESS"].apply(classify_ess)

    # Per-metric contribution columns  (weight_i × normalised_value_i)
    _METRIC_NAMES = {
        "c_lat": "inference speed",
        "c_thr": "throughput",
        "c_mem": "memory efficiency",
        "c_qua": "quality",
    }
    df["c_lat"] = np.round(df["L'"] * weights[0], 3)
    df["c_thr"] = np.round(df["R'"] * weights[1], 3)
    df["c_mem"] = np.round(df["M'"] * weights[2], 3)
    df["c_qua"] = np.round(df["Q'"] * weights[3], 3)

    # Human-readable reason
    def make_reason(row) -> str:
        contribs = {
            "inference speed": row["c_lat"],
            "throughput": row["c_thr"],
            "memory efficiency": row["c_mem"],
            "quality": row["c_qua"],
        }
        ranked = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
        top2 = [name for name, _ in ranked[:2]]
        if row["ESS_class"] == "suitable":
            return f"Strengths: {top2[0]} & {top2[1]}"
        elif row["ESS_class"] == "unsuitable":
            worst = ranked[-1][0]
            return f"Limitation: {worst}"
        else:
            return f"Balanced ({top2[0]}, {top2[1]})"

    df["reason"] = df.apply(make_reason, axis=1)
    return df


def run_benchmark(device_name: str) -> pd.DataFrame:
    """
    Simulate the complete benchmarking process for a given device:
    - Filter feasible models
    - Compute pre-selection score
    - Generate mock benchmark metrics
    - Calculate ESS ranking
    """
    device = devices[device_name]
    feasible_models = [m for m in models if is_feasible(m, device)]

    print(f"\nDevice: {device_name.upper()}")
    print(f"Feasible Models: {[m['name'] for m in feasible_models]}")

    if not feasible_models:
        print("  No feasible models for this device. Skipping.")
        return pd.DataFrame()

    # Compute pre-selection scores in one pass
    scores = compute_selection_score(feasible_models)
    for m, s in zip(feasible_models, scores):
        m["selection_score"] = s

    # Load real measured metrics from models.json.
    # Falls back to mem_req if live measurements are not yet available
    # (i.e., collect.py has not been run yet).
    results = []
    for m in feasible_models:
        results.append(
            {
                "model": m["name"],
                "latency": m.get("latency", m["mem_req"] * 1.5),
                "throughput": m.get("throughput", 10.0),
                "memory": m.get("mem_req", m["mem_req"]),
                "quality": m["quality"],
                "selection_score": m["selection_score"],
            }
        )

    df = pd.DataFrame(results)
    df = compute_ess(df)
    df = df.sort_values(by="ESS", ascending=False).reset_index(drop=True)
    return df


def plot_results(df: pd.DataFrame, device_name: str):
    """
    Create a single combined figure per device:
      left  — ESS bar chart with class labels and reason text
      right — stacked metric contribution breakdown
    Saved as one PNG to analytics/.
    """
    if df.empty:
        return

    colour_map = {"suitable": "#2ecc71", "partially suitable": "steelblue", "unsuitable": "#e74c3c"}
    colours = [colour_map.get(c, "steelblue") for c in df["ESS_class"]]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"ESS Analysis — {device_name.upper()}", fontsize=14, fontweight="bold")

    # ── Left: ESS bar with reason annotation ────────────────────────
    bars = ax.bar(df["model"], df["ESS"], color=colours, zorder=3)

    for bar, row in zip(bars, df.itertuples()):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.015,
            row.ESS_class,
            ha="center",
            va="bottom",
            fontsize=8,
            color="dimgray",
            fontweight="bold",
        )

    ax.set_title("Edge Suitability Score", fontsize=11)
    ax.set_ylabel("ESS (0–1)")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    # ── Right: Stacked metric contribution breakdown ─────────────────
    contrib_cols = ["c_lat", "c_thr", "c_mem", "c_qua"]
    segment_labels = ["Inference Speed", "Throughput", "Memory Efficiency", "Quality"]
    segment_colours = ["#3498db", "#e67e22", "#9b59b6", "#1abc9c"]

    bottoms = np.zeros(len(df))
    x = np.arange(len(df))

    for col, label, colour in zip(contrib_cols, segment_labels, segment_colours):
        vals = df[col].values.astype(float)
        ax2.bar(x, vals, bottom=bottoms, label=label, color=colour, alpha=0.88)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.025:
                ax2.text(i, b + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white")
        bottoms += vals

    for i, row in enumerate(df.itertuples()):
        total = float(df.iloc[i][contrib_cols].sum())
        badge_colour = colour_map.get(row.ESS_class, "gray")
        ax2.text(
            i,
            total + 0.01,
            row.ESS_class,
            ha="center",
            va="bottom",
            fontsize=8,
            color=badge_colour,
            fontweight="bold",
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(df["model"])
    ax2.set_title("Metric Contribution Breakdown", fontsize=11)
    ax2.set_ylabel("Weighted Metric Contribution")
    ax2.set_xlabel("Model")
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = ANALYTICS_DIR / f"{device_name}_analysis.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[CHART] Saved: {out_path}")


# ============================================================
#  MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Benchmark Pipeline")
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Run data collection first: detect device, download models, measure metrics.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help="Subset of model names to collect (e.g. --models TinyLlama Qwen2).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per prompt during collection (default: 128).",
    )
    parser.add_argument(
        "--prompt-set",
        choices=["all", "simple", "complex"],
        default="all",
        help="Prompt set to use during collection (default: all).",
    )
    parser.add_argument(
        "--track-warmup",
        action="store_true",
        help="During collection, capture cold-start vs warm-state latency.",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="Optional override for the multi-config storage key in models.json.",
    )
    parser.add_argument(
        "--analyze",
        choices=[
            "scaling",
            "cohort",
            "pareto",
            "warmup",
            "prompt_complexity",
            "parameter_scaling",
            "weight_sensitivity",
            "all",
        ],
        default=None,
        help="Run a multi-configuration analysis after the standard pipeline.",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Skip the per-device ESS pipeline; only run --analyze (if specified).",
    )
    args = parser.parse_args()

    if args.collect:
        from collect import run_collection

        run_collection(
            model_names=args.models,
            max_tokens=args.max_tokens,
            prompt_set=args.prompt_set,
            track_warmup=args.track_warmup,
            config_name=args.config_name,
        )
        # Reload models and devices after collection updates the JSON files
        with open(BASE_PATH / "models.json", "r") as f:
            models = json.load(f)
        with open(BASE_PATH / "device_profiles.json", "r") as f:
            devices = json.load(f)

    if not args.skip_pipeline:
        for dev in devices.keys():
            df_results = run_benchmark(dev)
            if df_results.empty:
                continue
            print(f"\nResults for {dev.upper()}:")
            print(
                df_results[["model", "ESS", "ESS_class", "reason", "latency", "throughput", "memory"]].to_string(
                    index=False
                )
            )
            plot_results(df_results, dev)

    if args.analyze:
        from analysis import run_analysis

        run_analysis(args.analyze, models)

    print("\nBenchmark pipeline completed successfully!")
