"""
analysis.py
============

Multi-configuration analysis module for the Automated Benchmark pipeline.

This module reads the multi-configuration results stored in
models.json under ``model["benchmarks"][config_key]`` and produces
the four secondary analyses required by the thesis:

  1. Token-scaling analysis     (analyze_token_scaling)
  2. Cohort comparison          (analyze_cohort)
  3. Pareto frontier            (analyze_pareto)
  4. Cold-start vs warm-state   (analyze_warmup)

All analyses save their figures (and CSV summaries) under the
``analytics/`` directory next to the existing pipeline plots.

The module exposes a single dispatch entry point:

    run_analysis(name, models)

where ``name`` is one of: ``scaling``, ``cohort``, ``pareto``,
``warmup``, ``all``.

Design notes
------------
* The functions are defensive: if no multi-configuration data is
  present, they print a helpful message and return without raising,
  so the main pipeline never crashes because analyses are missing.
* Cohort assignment follows the thesis convention:
    Baseline (2023):    TinyLlama, Yi, Llama3, Mistral
    Contemporary (2024): Qwen2, Gemma, SmolLM2, Qwen2.5, Phi3.5, Llama3.2
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # headless backend, consistent with pipeline.py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau
from scipy.stats import kendalltau

# ------------------------------------------------------------
#  Paths
# ------------------------------------------------------------

BASE_PATH = Path(__file__).parent
ANALYTICS_DIR = BASE_PATH / "analytics"
ANALYTICS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
#  Cohort definitions (must mirror the thesis)
# ------------------------------------------------------------

BASELINE_2023 = {"TinyLlama", "Yi", "Llama3", "Mistral"}
CONTEMPORARY_2024 = {"Qwen2", "Gemma", "SmolLM2", "Qwen2.5", "Phi3.5", "Llama3.2"}

# Parameter tier definitions for scaling analysis
PARAM_TIERS = {
    "Tier 1 (1-2B)": ["TinyLlama", "Qwen2", "SmolLM2"],
    "Tier 2 (2-4B)": ["Gemma", "Qwen2.5", "Phi3.5", "Llama3.2"],
    "Tier 3 (6-8B)": ["Yi", "Mistral", "Llama3"],
}


def _cohort_of(model_name: str) -> str | None:
    if model_name in BASELINE_2023:
        return "baseline_2023"
    if model_name in CONTEMPORARY_2024:
        return "contemporary_2024"
    return None


def _compute_entropy_weights(matrix: np.ndarray) -> np.ndarray:
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


def _normalize_metric(values, reverse=False):
    """
    Normalize a list of metric values to 0–1 range.
    If reverse=True, smaller values are better (e.g. latency).
    """
    v_min, v_max = min(values), max(values)
    if v_max == v_min:
        return [1.0 for _ in values]
    if reverse:
        return [1.0 - ((v - v_min) / (v_max - v_min)) for v in values]
    return [(v - v_min) / (v_max - v_min) for v in values]


def _compute_ess_scores(latencies, throughputs, memories, qualities, custom_weights=None):
    """
    Compute ESS scores for a set of models using entropy weighting or custom weights.
    Returns (scores, weights_used).
    """
    # Normalize metrics
    L_prime = _normalize_metric(latencies, reverse=True)
    R_prime = _normalize_metric(throughputs, reverse=False)
    M_prime = _normalize_metric(memories, reverse=True)
    Q_prime = _normalize_metric(qualities, reverse=False)

    # Build matrix and compute weights
    X = np.column_stack([L_prime, R_prime, M_prime, Q_prime])
    if custom_weights is None:
        weights = _compute_entropy_weights(X)
    else:
        weights = np.array(custom_weights)

    scores = X @ weights
    return scores, weights


# ------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------


def _iter_benchmarks(models: list[dict]) -> Iterable[tuple[dict, str, dict]]:
    """
    Yield (model_entry, config_key, measurements) for every multi-config
    entry stored under model['benchmarks']. Models without a 'benchmarks'
    dict are skipped.
    """
    for m in models:
        bench = m.get("benchmarks")
        if not isinstance(bench, dict):
            continue
        for config_key, measurements in bench.items():
            if isinstance(measurements, dict):
                yield m, config_key, measurements


def _save_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[CSV]   Saved: {path}")


def _save_fig(fig, name: str) -> None:
    out = ANALYTICS_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[CHART] Saved: {out}")


# ============================================================
#  1. TOKEN SCALING ANALYSIS
# ============================================================


def analyze_token_scaling(models: list[dict]) -> None:
    """
    For each model that has multiple ``max_tokens`` configurations,
    plot latency, throughput, and memory as a function of max_tokens
    and fit a simple linear regression to the latency curve.
    """
    # Group: model_name -> list of (max_tokens, latency, throughput, memory)
    raw_data: dict[str, list[tuple[int, float, float, float]]] = {}

    for model, _key, meas in _iter_benchmarks(models):
        mt = meas.get("max_tokens")
        if mt is None:
            continue
        # Prefer "all" prompt_set to avoid duplicates from simple/complex splits
        prompt_set = meas.get("prompt_set", "all")
        if prompt_set != "all":
            continue
        raw_data.setdefault(model["name"], []).append(
            (
                int(mt),
                float(meas.get("latency", 0.0)),
                float(meas.get("throughput", 0.0)),
                float(meas.get("memory", 0.0)),
            )
        )

    # Deduplicate: if multiple entries for same max_tokens, average them
    data: dict[str, list[tuple[int, float, float, float]]] = {}
    for model_name, points in raw_data.items():
        # Group by max_tokens
        by_tokens: dict[int, list[tuple[float, float, float]]] = {}
        for mt, lat, thr, mem in points:
            by_tokens.setdefault(mt, []).append((lat, thr, mem))

        # Average each group
        averaged = []
        for mt, measurements in sorted(by_tokens.items()):
            avg_lat = sum(m[0] for m in measurements) / len(measurements)
            avg_thr = sum(m[1] for m in measurements) / len(measurements)
            avg_mem = sum(m[2] for m in measurements) / len(measurements)
            averaged.append((mt, avg_lat, avg_thr, avg_mem))

        data[model_name] = averaged

    # Keep only models with at least 2 distinct token settings
    data = {k: v for k, v in data.items() if len(v) >= 2}

    if not data:
        print(
            "[ANALYSIS] token_scaling: no multi-token data found. "
            "Run collection with several --max-tokens values first."
        )
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    csv_rows: list[list] = []

    for name, points in sorted(data.items()):
        tokens = np.array([p[0] for p in points], dtype=float)
        lat = np.array([p[1] for p in points], dtype=float)
        thr = np.array([p[2] for p in points], dtype=float)
        mem = np.array([p[3] for p in points], dtype=float)

        axes[0].plot(tokens, lat, "o-", label=name)
        axes[1].plot(tokens, thr, "o-", label=name)
        axes[2].plot(tokens, mem, "o-", label=name)

        # Linear fit on latency vs tokens (only meaningful for >=2 points)
        slope, intercept = np.polyfit(tokens, lat, 1)
        for t, l, th, mb in points:
            csv_rows.append([name, t, l, th, mb, round(slope, 5), round(intercept, 4)])

    axes[0].set(title="Latency vs max_tokens", xlabel="max_tokens", ylabel="seconds / prompt")
    axes[1].set(title="Throughput vs max_tokens", xlabel="max_tokens", ylabel="tokens / s")
    axes[2].set(title="Peak memory vs max_tokens", xlabel="max_tokens", ylabel="GB")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Token-scaling analysis", fontsize=14, fontweight="bold")
    _save_fig(fig, "analysis_token_scaling.png")
    _save_csv(
        ANALYTICS_DIR / "analysis_token_scaling.csv",
        ["model", "max_tokens", "latency_s", "throughput_tps", "memory_gb", "lat_slope_s_per_token", "lat_intercept_s"],
        csv_rows,
    )


# ============================================================
#  2. COHORT COMPARISON ANALYSIS
# ============================================================


def analyze_cohort(models: list[dict]) -> None:
    """
    Compare baseline (2023) vs contemporary (2024) cohorts on
    parameter-normalised latency and throughput. Uses the legacy
    top-level fields when no multi-config data is present, so it
    works even before any extra collection runs.
    """
    rows: list[tuple[str, str, float, float, float]] = []
    for m in models:
        cohort = _cohort_of(m["name"])
        if cohort is None:
            continue
        params = float(m.get("params", 0.0)) or 1.0
        latency = float(m.get("latency", 0.0))
        throughput = float(m.get("throughput", 0.0))
        rows.append((m["name"], cohort, params, latency, throughput))

    if not rows:
        print("[ANALYSIS] cohort: no model entries with cohort assignment.")
        return

    base = [r for r in rows if r[1] == "baseline_2023"]
    cont = [r for r in rows if r[1] == "contemporary_2024"]

    def _summary(group: list[tuple[str, str, float, float, float]]) -> dict:
        if not group:
            return {"n": 0}
        lat_per_b = np.array([r[3] / r[2] for r in group])  # s per Billion params
        thr_per_b = np.array([r[4] / r[2] for r in group])
        return {
            "n": len(group),
            "lat_per_B_mean": float(np.mean(lat_per_b)),
            "lat_per_B_std": float(np.std(lat_per_b, ddof=1)) if len(group) > 1 else 0.0,
            "thr_per_B_mean": float(np.mean(thr_per_b)),
            "thr_per_B_std": float(np.std(thr_per_b, ddof=1)) if len(group) > 1 else 0.0,
        }

    summary = {"baseline_2023": _summary(base), "contemporary_2024": _summary(cont)}

    # Welch's t-test (manual, no scipy dep)
    def _welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
        if len(a) < 2 or len(b) < 2:
            return float("nan"), 0
        va, vb = a.var(ddof=1), b.var(ddof=1)
        na, nb = len(a), len(b)
        denom = np.sqrt(va / na + vb / nb)
        if denom == 0:
            return float("nan"), 0
        t = (a.mean() - b.mean()) / denom
        # Welch-Satterthwaite df, rounded to int
        df_num = (va / na + vb / nb) ** 2
        df_den = (va**2) / ((na**2) * (na - 1)) + (vb**2) / ((nb**2) * (nb - 1))
        df = int(df_num / df_den) if df_den > 0 else 0
        return float(t), df

    base_lat = np.array([r[3] / r[2] for r in base])
    cont_lat = np.array([r[3] / r[2] for r in cont])
    t_stat, df_used = _welch_t(base_lat, cont_lat)

    print("\n[ANALYSIS] Cohort comparison (params-normalised):")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  Welch's t (latency/B params): t={t_stat:.3f}  df={df_used}")

    # Single grouped-bar chart: latency-per-B and throughput-per-B
    # for each cohort, with the t-statistic in the subtitle so the
    # reader knows whether the difference is statistically meaningful.
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Latency", "Throughput"]
    base_vals = [
        summary["baseline_2023"].get("lat_per_B_mean", 0.0),
        summary["baseline_2023"].get("thr_per_B_mean", 0.0),
    ]
    cont_vals = [
        summary["contemporary_2024"].get("lat_per_B_mean", 0.0),
        summary["contemporary_2024"].get("thr_per_B_mean", 0.0),
    ]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(
        x - w / 2,
        base_vals,
        w,
        label=f"Baseline 2023 (n={summary['baseline_2023']['n']})",
        color="#888888",
    )
    ax.bar(
        x + w / 2,
        cont_vals,
        w,
        label=f"Contemporary 2024 (n={summary['contemporary_2024']['n']})",
        color="#1f77b4",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Per billion parameters (s for latency, tps for throughput)")
    ax.set_title("Cohort comparison")
    ax.text(
        0.5,
        1.01,
        f"Welch's t = {t_stat:.2f},  df = {df_used}",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "analysis_cohort_comparison.png")

    # CSV
    csv_rows = [[r[0], r[1], r[2], r[3], r[4], r[3] / r[2], r[4] / r[2]] for r in rows]
    _save_csv(
        ANALYTICS_DIR / "analysis_cohort_comparison.csv",
        ["model", "cohort", "params_B", "latency_s", "throughput_tps", "lat_per_B", "thr_per_B"],
        csv_rows,
    )

    # Persist summary as JSON for the thesis appendix
    summary_path = ANALYTICS_DIR / "analysis_cohort_summary.json"
    summary_payload = {**summary, "welch_t_latency_per_B": t_stat, "df": df_used}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"[JSON]  Saved: {summary_path}")


# ============================================================
#  3. PARETO FRONTIER ANALYSIS
# ============================================================


def analyze_pareto(models: list[dict]) -> None:
    """
    Identify the Pareto frontier on (latency vs quality) and
    (memory vs quality). A point is Pareto-optimal if no other
    point has both lower cost AND higher (or equal) quality.
    """
    pts: list[tuple[str, float, float, float]] = []  # (name, latency, memory, quality)
    for m in models:
        lat = float(m.get("latency", 0.0))
        mem = float(m.get("mem_req", 0.0))
        qua = float(m.get("quality", 0.0))
        if lat <= 0 or qua <= 0:
            continue
        pts.append((m["name"], lat, mem, qua))

    if not pts:
        print("[ANALYSIS] pareto: no usable model data.")
        return

    def _pareto(cost_idx: int) -> set[str]:
        """Names of Pareto-optimal models for (cost = pts[i][cost_idx], quality = pts[i][3])."""
        front: set[str] = set()
        for i, (ni, *vals_i) in enumerate(pts):
            ci, qi = vals_i[cost_idx - 1], vals_i[2]
            dominated = False
            for j, (nj, *vals_j) in enumerate(pts):
                if i == j:
                    continue
                cj, qj = vals_j[cost_idx - 1], vals_j[2]
                if cj <= ci and qj >= qi and (cj < ci or qj > qi):
                    dominated = True
                    break
            if not dominated:
                front.add(ni)
        return front

    # Use memory as the cost dimension (memory is typically the harder constraint on edge devices)
    front_mem = _pareto(2)  # memory

    fig, ax = plt.subplots(figsize=(9, 6))

    # Sort points so labels for the frontier draw on top.
    ordered = sorted(pts, key=lambda p: p[0] in front_mem)

    # Plot non-frontier as grey dots, frontier as red, larger.
    for name, lat, mem, qua in ordered:
        on_front = name in front_mem
        ax.scatter(
            mem,
            qua,
            s=130 if on_front else 55,
            c="#d62728" if on_front else "#bbbbbb",
            edgecolors="black",
            linewidths=0.8,
            zorder=3 if on_front else 2,
        )

    # Annotate ONLY frontier models in-plot.
    for name, lat, mem, qua in pts:
        if name not in front_mem:
            continue
        ax.annotate(
            name,
            (mem, qua),
            fontsize=9,
            fontweight="bold",
            xytext=(8, 6),
            textcoords="offset points",
            color="#8b1a1a",
        )

    # Simple legend without listing dominated models
    legend_handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color="#d62728",
            markeredgecolor="black",
            markersize=10,
            label="Pareto-optimal",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color="#bbbbbb",
            markeredgecolor="black",
            markersize=8,
            label="Dominated",
        ),
    ]

    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.9)

    ax.set_xlabel("Memory requirement (GB)")
    ax.set_ylabel("Quality")
    ax.set_title("Pareto frontier")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, "analysis_pareto.png")

    # CSV still includes both latency and memory frontiers for completeness
    front_lat = _pareto(1)
    csv_rows = [[n, lat, mem, qua, n in front_lat, n in front_mem] for n, lat, mem, qua in pts]
    _save_csv(
        ANALYTICS_DIR / "analysis_pareto.csv",
        ["model", "latency_s", "memory_gb", "quality", "pareto_latency", "pareto_memory"],
        csv_rows,
    )


# ============================================================
#  4. WARMUP / COLD-START ANALYSIS
# ============================================================


def analyze_warmup(models: list[dict]) -> None:
    """
    Compare cold-start latency (first prompt) vs warm-state latency
    (mean of prompts 10-19). Requires at least one config collected
    with --track-warmup.
    """
    rows: list[tuple[str, str, float, float, float]] = []
    for model, key, meas in _iter_benchmarks(models):
        cold = meas.get("cold_start_latency")
        warm = meas.get("warm_latency")
        if cold is None or warm is None:
            continue
        cold_f, warm_f = float(cold), float(warm)
        overhead_pct = ((cold_f - warm_f) / warm_f * 100.0) if warm_f > 0 else 0.0
        rows.append((model["name"], key, cold_f, warm_f, overhead_pct))

    if not rows:
        print("[ANALYSIS] warmup: no cold/warm data found. " "Run collection with --track-warmup first.")
        return

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(rows) + 4), 5))
    x = np.arange(len(rows))
    w = 0.4
    ax.bar(x - w / 2, [r[2] for r in rows], w, label="cold start (1st prompt)", color="#d62728")
    ax.bar(x + w / 2, [r[3] for r in rows], w, label="warm (prompts 10-19)", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r[0]}\n{r[1]}" for r in rows], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("latency (s)")
    ax.set_title("Cold-start vs warm-state latency")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, "analysis_warmup.png")

    _save_csv(
        ANALYTICS_DIR / "analysis_warmup.csv",
        ["model", "config", "cold_start_latency_s", "warm_latency_s", "overhead_percent"],
        [[r[0], r[1], r[2], r[3], round(r[4], 2)] for r in rows],
    )


# ============================================================
#  5. PROMPT COMPLEXITY STRATIFICATION
# ============================================================


def analyze_prompt_complexity(models: list[dict]) -> None:
    """
    Compare ESS rankings between simple and complex prompt sets.
    Checks if model rankings change across prompt complexity levels.
    """
    # Extract data for simple and complex configurations
    simple_data = {}
    complex_data = {}

    for model in models:
        name = model["name"]
        benchmarks = model.get("benchmarks", {})

        # Find simple and complex configurations (look for psimple, pcomplex in config_key)
        for config_key, meas in benchmarks.items():
            if "psimple" in config_key:
                simple_data[name] = meas
            elif "pcomplex" in config_key:
                complex_data[name] = meas

    # Check if we have data for both
    common_models = set(simple_data.keys()) & set(complex_data.keys())
    if len(common_models) < 2:
        print(
            "[ANALYSIS] prompt_complexity: insufficient data. "
            "Run collection with --prompt-set simple and --prompt-set complex first."
        )
        return

    # Build lists for ESS calculation
    model_names = sorted(common_models)
    simple_lats = [simple_data[m]["latency"] for m in model_names]
    simple_thrs = [simple_data[m]["throughput"] for m in model_names]
    simple_mems = [simple_data[m]["memory"] for m in model_names]
    complex_lats = [complex_data[m]["latency"] for m in model_names]
    complex_thrs = [complex_data[m]["throughput"] for m in model_names]
    complex_mems = [complex_data[m]["memory"] for m in model_names]

    # Get quality scores from top-level model data
    qualities = [next(m["quality"] for m in models if m["name"] == name) for name in model_names]

    # Compute ESS for both prompt sets
    simple_ess, simple_weights = _compute_ess_scores(simple_lats, simple_thrs, simple_mems, qualities)
    complex_ess, complex_weights = _compute_ess_scores(complex_lats, complex_thrs, complex_mems, qualities)

    # Compute rankings (higher ESS = better rank)
    simple_ranks = [sorted(simple_ess, reverse=True).index(s) + 1 for s in simple_ess]
    complex_ranks = [sorted(complex_ess, reverse=True).index(s) + 1 for s in complex_ess]

    # Spearman correlation
    from scipy.stats import spearmanr

    corr, p_value = spearmanr(simple_ranks, complex_ranks)

    # Plot: side-by-side bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.35

    ax.bar(x - width / 2, simple_ess, width, label="Simple prompts", color="#4CAF50", alpha=0.8)
    ax.bar(x + width / 2, complex_ess, width, label="Complex prompts", color="#FF9800", alpha=0.8)

    ax.set_xlabel("Model")
    ax.set_ylabel("ESS Score")
    ax.set_title(f"Prompt Complexity Comparison (Spearman ρ = {corr:.3f}, p = {p_value:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "analysis_prompt_complexity.png")

    # CSV output
    csv_rows = [
        [
            model_names[i],
            round(simple_ess[i], 3),
            simple_ranks[i],
            round(complex_ess[i], 3),
            complex_ranks[i],
            abs(simple_ranks[i] - complex_ranks[i]),
        ]
        for i in range(len(model_names))
    ]
    _save_csv(
        ANALYTICS_DIR / "analysis_prompt_complexity.csv",
        ["model", "simple_ess", "simple_rank", "complex_ess", "complex_rank", "rank_change"],
        csv_rows,
    )

    # Summary JSON
    summary = {
        "spearman_correlation": round(corr, 3),
        "p_value": round(p_value, 4),
        "interpretation": (
            "High correlation (>0.8) = stable rankings" if corr > 0.8 else "Rankings change with prompt type"
        ),
    }
    summary_path = ANALYTICS_DIR / "analysis_prompt_complexity_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[JSON]  Saved: {summary_path}")


# ============================================================
#  6. PARAMETER SCALING STUDY
# ============================================================


def analyze_parameter_scaling(models: list[dict]) -> None:
    """
    Group models by parameter tiers and analyze scaling efficiency.
    Plots latency and memory vs parameter count, calculates efficiency ratios.
    """
    # Collect data: name, params, latency, mem_req, quality
    data = []
    for m in models:
        # Use top-level benchmark data (mem_req for memory requirement)
        if all(k in m for k in ["name", "params", "latency", "mem_req", "quality"]):
            data.append(
                {
                    "name": m["name"],
                    "params": m["params"],
                    "latency": m["latency"],
                    "memory": m["mem_req"],  # Use mem_req as memory proxy
                    "quality": m["quality"],
                }
            )

    if len(data) < 2:
        print("[ANALYSIS] parameter_scaling: insufficient baseline data.")
        return

    # Sort by parameter count
    data.sort(key=lambda x: x["params"])

    # Extract arrays
    names = [d["name"] for d in data]
    params = np.array([d["params"] for d in data])
    latencies = np.array([d["latency"] for d in data])
    memories = np.array([d["memory"] for d in data])
    qualities = np.array([d["quality"] for d in data])

    # Calculate efficiency ratios
    lat_efficiency = qualities / latencies  # Higher = better (quality per second)
    mem_efficiency = qualities / memories  # Higher = better (quality per GB)

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Latency vs Params
    axes[0, 0].scatter(params, latencies, s=100, alpha=0.7, color="#2196F3")
    for i, name in enumerate(names):
        axes[0, 0].annotate(name, (params[i], latencies[i]), fontsize=8, ha="right", alpha=0.7)
    axes[0, 0].set_xlabel("Parameters (B)")
    axes[0, 0].set_ylabel("Latency (s)")
    axes[0, 0].set_title("Latency vs Parameter Count")
    axes[0, 0].grid(alpha=0.3)

    # Top-right: Memory vs Params
    axes[0, 1].scatter(params, memories, s=100, alpha=0.7, color="#FF5722")
    for i, name in enumerate(names):
        axes[0, 1].annotate(name, (params[i], memories[i]), fontsize=8, ha="right", alpha=0.7)
    axes[0, 1].set_xlabel("Parameters (B)")
    axes[0, 1].set_ylabel("Memory (GB)")
    axes[0, 1].set_title("Memory vs Parameter Count")
    axes[0, 1].grid(alpha=0.3)

    # Bottom-left: Latency Efficiency
    axes[1, 0].bar(range(len(names)), lat_efficiency, color="#4CAF50", alpha=0.8)
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_xticklabels(names, rotation=45, ha="right")
    axes[1, 0].set_ylabel("Quality / Latency (score/s)")
    axes[1, 0].set_title("Latency Efficiency Ratio")
    axes[1, 0].grid(axis="y", alpha=0.3)

    # Bottom-right: Memory Efficiency
    axes[1, 1].bar(range(len(names)), mem_efficiency, color="#9C27B0", alpha=0.8)
    axes[1, 1].set_xticks(range(len(names)))
    axes[1, 1].set_xticklabels(names, rotation=45, ha="right")
    axes[1, 1].set_ylabel("Quality / Memory (score/GB)")
    axes[1, 1].set_title("Memory Efficiency Ratio")
    axes[1, 1].grid(axis="y", alpha=0.3)

    fig.suptitle("Parameter Scaling Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_fig(fig, "analysis_parameter_scaling.png")

    # CSV output
    csv_rows = [
        [
            names[i],
            params[i],
            round(latencies[i], 3),
            round(memories[i], 3),
            round(qualities[i], 3),
            round(lat_efficiency[i], 4),
            round(mem_efficiency[i], 4),
        ]
        for i in range(len(names))
    ]
    _save_csv(
        ANALYTICS_DIR / "analysis_parameter_scaling.csv",
        ["model", "params_B", "latency_s", "memory_GB", "quality", "qual_per_latency", "qual_per_memory"],
        csv_rows,
    )


# ============================================================
#  7. WEIGHT SENSITIVITY ANALYSIS
# ============================================================


def analyze_weight_sensitivity(models: list[dict]) -> None:
    """
    Test ESS ranking stability across different weighting schemes.
    Compares data-driven EWM weights vs equal/latency-focused/quality-focused.
    """
    # Collect baseline data
    data = []
    for m in models:
        if all(k in m for k in ["name", "latency", "throughput", "mem_req", "quality"]):
            data.append(m)

    if len(data) < 2:
        print("[ANALYSIS] weight_sensitivity: insufficient baseline data.")
        return

    names = [m["name"] for m in data]
    latencies = [m["latency"] for m in data]
    throughputs = [m["throughput"] for m in data]
    memories = [m["mem_req"] for m in data]  # Use mem_req field
    qualities = [m["quality"] for m in data]

    # Define 4 weighting schemes
    schemes = {
        "EWM (data-driven)": None,  # Will compute entropy weights
        "Equal weights": [0.25, 0.25, 0.25, 0.25],
        "Latency-focused": [0.5, 0.2, 0.15, 0.15],
        "Quality-focused": [0.15, 0.15, 0.2, 0.5],
    }

    # Compute ESS for each scheme
    all_ess = {}
    all_ranks = {}
    for scheme_name, weights in schemes.items():
        ess, _ = _compute_ess_scores(latencies, throughputs, memories, qualities, custom_weights=weights)
        all_ess[scheme_name] = ess
        # Ranks: 1 = best (highest ESS)
        all_ranks[scheme_name] = [sorted(ess, reverse=True).index(s) + 1 for s in ess]

    # Compute pairwise Kendall's tau
    scheme_names = list(schemes.keys())
    tau_matrix = np.zeros((len(scheme_names), len(scheme_names)))
    for i, s1 in enumerate(scheme_names):
        for j, s2 in enumerate(scheme_names):
            tau, _ = kendalltau(all_ranks[s1], all_ranks[s2])
            tau_matrix[i, j] = tau

    # Plot 1: ESS scores for each scheme (grouped bar chart)
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    x = np.arange(len(names))
    width = 0.2
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    for i, scheme_name in enumerate(scheme_names):
        offset = (i - 1.5) * width
        ax1.bar(x + offset, all_ess[scheme_name], width, label=scheme_name, color=colors[i], alpha=0.8)

    ax1.set_xlabel("Model")
    ax1.set_ylabel("ESS Score")
    ax1.set_title("ESS Sensitivity to Weight Scheme")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    fig1.tight_layout()
    _save_fig(fig1, "analysis_weight_sensitivity.png")

    # Plot 2: Kendall's tau heatmap
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im = ax2.imshow(tau_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(scheme_names)))
    ax2.set_yticks(range(len(scheme_names)))
    ax2.set_xticklabels(scheme_names, rotation=45, ha="right")
    ax2.set_yticklabels(scheme_names)
    ax2.set_title("Kendall's Tau: Ranking Concordance")

    # Annotate cells
    for i in range(len(scheme_names)):
        for j in range(len(scheme_names)):
            text = ax2.text(j, i, f"{tau_matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=10)

    fig2.colorbar(im, ax=ax2, label="Kendall's τ")
    fig2.tight_layout()
    _save_fig(fig2, "analysis_weight_sensitivity_heatmap.png")

    # CSV output: ESS scores per scheme
    csv_rows = []
    for i, name in enumerate(names):
        row = [name]
        for scheme_name in scheme_names:
            row.append(round(all_ess[scheme_name][i], 3))
        csv_rows.append(row)

    _save_csv(
        ANALYTICS_DIR / "analysis_weight_sensitivity.csv",
        ["model"] + scheme_names,
        csv_rows,
    )

    # CSV output: Kendall's tau matrix
    tau_csv_rows = [
        [scheme_names[i]] + [round(tau_matrix[i, j], 3) for j in range(len(scheme_names))]
        for i in range(len(scheme_names))
    ]
    _save_csv(
        ANALYTICS_DIR / "analysis_weight_sensitivity_tau.csv",
        ["scheme"] + scheme_names,
        tau_csv_rows,
    )

    # Summary JSON
    avg_tau = np.mean([tau_matrix[i, j] for i in range(len(scheme_names)) for j in range(i + 1, len(scheme_names))])
    summary = {
        "average_kendall_tau": round(avg_tau, 3),
        "interpretation": (
            "High stability (τ > 0.8)" if avg_tau > 0.8 else "Moderate stability" if avg_tau > 0.6 else "Low stability"
        ),
    }
    summary_path = ANALYTICS_DIR / "analysis_weight_sensitivity_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[JSON]  Saved: {summary_path}")


# ============================================================
#  DISPATCH ENTRY POINT
# ============================================================

_DISPATCH = {
    "scaling": analyze_token_scaling,
    "cohort": analyze_cohort,
    "pareto": analyze_pareto,
    "warmup": analyze_warmup,
    "prompt_complexity": analyze_prompt_complexity,
    "parameter_scaling": analyze_parameter_scaling,
    "weight_sensitivity": analyze_weight_sensitivity,
}


def run_analysis(name: str, models: list[dict]) -> None:
    """
    Dispatch to one (or all) of the analysis functions.

    Args:
        name:   One of 'scaling', 'cohort', 'pareto', 'warmup', 'all'.
        models: The list loaded from models.json.
    """
    print(f"\n[ANALYSIS] === Running '{name}' analysis ===")
    if name == "all":
        for fn in _DISPATCH.values():
            fn(models)
    elif name in _DISPATCH:
        _DISPATCH[name](models)
    else:
        raise ValueError(f"Unknown analysis '{name}'. " f"Choices: {list(_DISPATCH.keys()) + ['all']}")


# Allow standalone execution: python analysis.py [name]
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run multi-configuration analyses.")
    p.add_argument(
        "name",
        nargs="?",
        default="all",
        choices=list(_DISPATCH.keys()) + ["all"],
        help="Which analysis to run (default: all).",
    )
    args = p.parse_args()

    with open(BASE_PATH / "models.json", "r", encoding="utf-8") as f:
        _models = json.load(f)
    run_analysis(args.name, _models)
