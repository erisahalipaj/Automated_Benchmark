"""
collect.py
==========
Data collection module for the Automated Benchmark Pipeline.

Responsibilities:
  1. Auto-detect the current device's hardware profile and update device_profiles.json.
  2. Download quantized GGUF models from HuggingFace (if not already cached).
  3. Run each model against the full prompt set and measure:
       - Average latency (seconds/prompt)
       - Average throughput (tokens/second)
       - Peak memory usage (GB)
  4. Persist real measurements back into models.json so the pipeline
     uses actual, device-measured values instead of mock data.

Usage (standalone):
    python collect.py

Usage (integrated, called from pipeline.py with --collect flag):
    python pipeline.py --collect
"""

import json
import os
import platform
import socket
import time
from pathlib import Path

import numpy as np
import psutil
from huggingface_hub import hf_hub_download

# ============================================================
#  PATHS
# ============================================================

BASE_PATH = Path(__file__).parent
MODELS_JSON = BASE_PATH / "models.json"
DEVICES_JSON = BASE_PATH / "device_profiles.json"
PROMPTS_FILE = BASE_PATH / "prompts.txt"
PROMPTS_SIMPLE_FILE = BASE_PATH / "prompts_simple.txt"
PROMPTS_COMPLEX_FILE = BASE_PATH / "prompts_complex.txt"
MODEL_CACHE_DIR = BASE_PATH / "model_cache"

MODEL_CACHE_DIR.mkdir(exist_ok=True)

# Map prompt-set identifiers to their source files
PROMPT_SETS = {
    "all": PROMPTS_FILE,
    "simple": PROMPTS_SIMPLE_FILE,
    "complex": PROMPTS_COMPLEX_FILE,
}

# ============================================================
#  HuggingFace GGUF Model Registry
#  Maps the model name (as in models.json) to its HuggingFace
#  repo, filename (Q4_K_M quantization), and known fixed
#  quality score (from public benchmarks e.g. MMLU, HellaSwag).
# ============================================================

GGUF_REGISTRY: dict[str, dict] = {
    "TinyLlama": {
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "params": 1.1,
        "quality": 0.80,
    },
    "Qwen2": {
        "repo_id": "Qwen/Qwen2-1.5B-Instruct-GGUF",
        "filename": "qwen2-1_5b-instruct-q4_k_m.gguf",
        "params": 1.5,
        "quality": 0.82,
    },
    "Yi": {
        "repo_id": "bartowski/Yi-1.5-6B-Chat-GGUF",
        "filename": "Yi-1.5-6B-Chat-Q4_K_M.gguf",
        "params": 6.0,
        "quality": 0.79,
    },
    "Gemma": {
        "repo_id": "bartowski/gemma-2-2b-it-GGUF",
        "filename": "gemma-2-2b-it-Q4_K_M.gguf",
        "params": 2.6,
        "quality": 0.85,
    },
    "Llama3": {
        "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "params": 8.0,
        "quality": 0.88,
    },
    "Mistral": {
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "params": 7.2,
        "quality": 0.91,
    },
    "SmolLM2": {
        "repo_id": "bartowski/SmolLM2-1.7B-Instruct-GGUF",
        "filename": "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        "params": 1.7,
        "quality": 0.81,
    },
    "Qwen2.5": {
        "repo_id": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "filename": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "params": 3.0,
        "quality": 0.86,
    },
    "Phi3.5": {
        "repo_id": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "params": 3.8,
        "quality": 0.89,
    },
    "Llama3.2": {
        "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "params": 3.2,
        "quality": 0.87,
    },
}


# ============================================================
#  STEP 1: Auto-detect device profile
# ============================================================


def detect_device_profile() -> dict:
    """
    Reads current hardware specs using psutil and platform.

    Returns a dict with:
        RAM  - total RAM in GB
        CPU  - number of logical CPU cores
        STO  - available disk space in GB (on the drive where the project lives)
    """
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    cpu_cores = psutil.cpu_count(logical=True)
    disk_usage = psutil.disk_usage(str(BASE_PATH))
    sto_gb = round(disk_usage.free / (1024**3), 1)

    profile = {
        "RAM": ram_gb,
        "CPU": cpu_cores,
        "STO": sto_gb,
    }

    print(f"[DEVICE] Detected profile: {profile}")
    return profile


def get_device_key() -> str:
    """
    Returns a simple identifier for the current device based on hostname.
    Falls back to 'laptop' if hostname is not informative.
    """
    hostname = socket.gethostname().lower()
    if "pi" in hostname or "raspberry" in hostname:
        return "raspberry_pi"
    if "jetson" in hostname or "nano" in hostname:
        return "jetson_nano"
    return "laptop"


def update_device_profiles(profile: dict, device_key: str) -> None:
    """Writes the detected profile into device_profiles.json."""
    with open(DEVICES_JSON, "r") as f:
        devices = json.load(f)

    devices[device_key] = profile

    with open(DEVICES_JSON, "w") as f:
        json.dump(devices, f, indent=4)

    print(f"[DEVICE] Saved profile as '{device_key}' in device_profiles.json")


# ============================================================
#  STEP 2: Download quantized model
# ============================================================


def download_model(model_name: str) -> Path:
    """
    Downloads the GGUF model from HuggingFace Hub if not already cached.

    Returns:
        Path to the local .gguf file.
    """
    if model_name not in GGUF_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in GGUF_REGISTRY.")

    reg = GGUF_REGISTRY[model_name]
    local_path = MODEL_CACHE_DIR / reg["filename"]

    if local_path.exists():
        print(f"[DOWNLOAD] {model_name}: already cached at {local_path}")
        return local_path

    print(f"[DOWNLOAD] {model_name}: downloading from {reg['repo_id']} ...")
    downloaded = hf_hub_download(
        repo_id=reg["repo_id"],
        filename=reg["filename"],
        local_dir=str(MODEL_CACHE_DIR),
    )
    print(f"[DOWNLOAD] {model_name}: saved to {downloaded}")
    return Path(downloaded)


# ============================================================
#  STEP 3: Load prompts
# ============================================================


def load_prompts(prompt_set: str = "all") -> list[str]:
    """
    Reads prompts from the requested prompt set file.
    Skips blank lines and comment lines starting with '#'.

    Args:
        prompt_set: One of 'all', 'simple', 'complex'.
    """
    if prompt_set not in PROMPT_SETS:
        raise ValueError(f"Unknown prompt set '{prompt_set}'. " f"Valid options: {list(PROMPT_SETS.keys())}")

    prompts_file = PROMPT_SETS[prompt_set]
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompts_file}")

    prompts = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)

    if not prompts:
        raise ValueError(f"{prompts_file.name} is empty or has no valid prompts.")

    print(f"[PROMPTS] Loaded {len(prompts)} prompts from {prompts_file.name}")
    return prompts


# ============================================================
#  STEP 4: Run inference and measure metrics
# ============================================================


def benchmark_model(
    model_path: Path,
    prompts: list[str],
    max_tokens: int = 128,
    track_warmup: bool = False,
) -> dict:
    """
    Loads a GGUF model using llama-cpp-python, runs all prompts,
    and records per-prompt latency, throughput, and peak memory.

    Args:
        model_path:    Path to the GGUF model file.
        prompts:       List of prompts to run.
        max_tokens:    Maximum tokens to generate per prompt (default 128).
        track_warmup:  If True, also reports cold_start_latency (first prompt)
                       and warm_latency (mean of prompts 10-19) separately.

    Metrics returned:
        latency    - mean seconds per prompt (over all prompts)
        throughput - mean tokens per second
        memory     - peak RAM usage in GB during inference
        size       - model file size on disk in GB
        max_tokens - the max_tokens setting used
        n_prompts  - number of prompts processed
        cold_start_latency (optional) - latency of the first prompt
        warm_latency (optional)       - mean latency of prompts 10-19
    """
    # Import here so the module remains usable even if llama-cpp is not yet installed
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError("llama-cpp-python is not installed. " "Run: pip install llama-cpp-python")

    model_size_gb = round(model_path.stat().st_size / (1024**3), 2)

    print(f"[BENCH] Loading {model_path.name} ({model_size_gb} GB) " f"max_tokens={max_tokens} prompts={len(prompts)}")
    llm = Llama(
        model_path=str(model_path),
        n_ctx=512,  # context window
        n_threads=psutil.cpu_count(logical=False) or 4,
        verbose=False,
    )

    latencies: list[float] = []
    throughputs: list[float] = []
    mem_samples: list[float] = []

    process = psutil.Process(os.getpid())

    for i, prompt in enumerate(prompts):
        t_start = time.perf_counter()
        response = llm(
            prompt,
            max_tokens=max_tokens,
            echo=False,
        )
        t_end = time.perf_counter()

        elapsed = t_end - t_start
        tokens_generated = response["usage"]["completion_tokens"]
        tps = tokens_generated / elapsed if elapsed > 0 else 0.0

        mem_after = process.memory_info().rss / (1024**3)

        latencies.append(elapsed)
        throughputs.append(tps)
        mem_samples.append(mem_after)

        if (i + 1) % 10 == 0 or (i + 1) == len(prompts):
            print(f"  [{i+1}/{len(prompts)}] " f"lat={elapsed:.2f}s  tps={tps:.1f}  mem={mem_after:.2f}GB")

    # Free model from memory before returning
    del llm

    result = {
        "latency": round(float(np.mean(latencies)), 3),
        "throughput": round(float(np.mean(throughputs)), 2),
        "memory": round(float(np.max(mem_samples)), 2),
        "size": model_size_gb,
        "max_tokens": max_tokens,
        "n_prompts": len(prompts),
    }

    # Cold start and warm-state metrics: only meaningful when we have enough prompts.
    if track_warmup and len(latencies) >= 1:
        result["cold_start_latency"] = round(float(latencies[0]), 3)
        # Warm window = prompts 10..19 (indices 10..19 inclusive); fall back if shorter.
        warm_start = min(10, len(latencies) - 1)
        warm_end = min(20, len(latencies))
        warm_window = latencies[warm_start:warm_end]
        if warm_window:
            result["warm_latency"] = round(float(np.mean(warm_window)), 3)

    return result


# ============================================================
#  STEP 5: Persist measured values to models.json
# ============================================================


def update_models_json(
    model_name: str,
    measured: dict,
    config_key: str | None = None,
    update_top_level: bool = True,
) -> None:
    """
    Updates the matching model entry in models.json with measured values.

    Args:
        model_name:       The model name to update.
        measured:         Dict from benchmark_model().
        config_key:       Optional configuration identifier. If provided, the
                          measurements are also stored under
                          model["benchmarks"][config_key] for multi-configuration
                          analysis (token scaling, prompt-set comparison, etc.).
        update_top_level: If True (default), also writes latency/throughput/memory
                          to the top-level model fields. This preserves backward
                          compatibility with the existing pipeline.py reader.
    """
    with open(MODELS_JSON, "r") as f:
        models = json.load(f)

    for m in models:
        if m["name"] == model_name:
            if update_top_level:
                m["mem_req"] = measured["memory"]
                m["size"] = measured["size"]
                m["latency"] = measured["latency"]
                m["throughput"] = measured["throughput"]
                # cpu_req: approximate from observed throughput (lower = more cores needed)
                m["cpu_req"] = round(max(1.0, 10.0 / (measured["throughput"] + 1e-6)), 2)

            if config_key is not None:
                if "benchmarks" not in m or not isinstance(m.get("benchmarks"), dict):
                    m["benchmarks"] = {}
                m["benchmarks"][config_key] = measured
            break

    with open(MODELS_JSON, "w") as f:
        json.dump(models, f, indent=4)

    suffix = f" [{config_key}]" if config_key else ""
    print(f"[SAVE] Updated models.json for {model_name}{suffix}")


# ============================================================
#  MAIN COLLECTION RUNNER
# ============================================================


def build_config_key(device_key: str, max_tokens: int, prompt_set: str) -> str:
    """
    Build a deterministic config identifier used as the key under
    model['benchmarks'] in models.json. Example: 'laptop_t128_pall'.
    """
    return f"{device_key}_t{max_tokens}_p{prompt_set}"


def run_collection(
    model_names: list[str] | None = None,
    max_tokens: int = 128,
    prompt_set: str = "all",
    track_warmup: bool = False,
    config_name: str | None = None,
) -> None:
    """
    Orchestrates the full data collection process:
      1. Detect and save device profile.
      2. Load prompt set.
      3. For each model: download → benchmark → save.

    Args:
        model_names:  Optional list of model names to collect.
                      If None, all models in GGUF_REGISTRY are used.
        max_tokens:   Maximum tokens to generate per prompt.
        prompt_set:   Which prompt set to use ('all', 'simple', 'complex').
        track_warmup: If True, also captures cold/warm latency separately.
        config_name:  Optional override for the multi-config storage key.
                      If None, an auto-key is built from device + tokens + set.

    Backward compatibility: when called with defaults (max_tokens=128,
    prompt_set='all', config_name=None), the run still updates the legacy
    top-level fields (latency, throughput, memory, etc.) in models.json so
    the existing pipeline.py reader continues to work unchanged.
    """
    print("\n" + "=" * 60)
    print("  AUTOMATED BENCHMARK — DATA COLLECTION")
    print(f"  max_tokens={max_tokens}  prompt_set={prompt_set}  track_warmup={track_warmup}")
    print("=" * 60)

    # --- Step 1: Device profile ---
    profile = detect_device_profile()
    device_key = get_device_key()
    update_device_profiles(profile, device_key)

    # --- Step 2: Prompts ---
    prompts = load_prompts(prompt_set=prompt_set)

    # --- Step 3: Models ---
    targets = model_names or list(GGUF_REGISTRY.keys())

    # Determine the config key for multi-configuration storage.
    config_key = config_name or build_config_key(device_key, max_tokens, prompt_set)

    # Default-config runs (max_tokens=128, prompt_set='all') keep the legacy
    # top-level fields up to date so pipeline.py continues to read real values.
    is_default_config = max_tokens == 128 and prompt_set == "all" and not track_warmup
    update_top_level = is_default_config

    for model_name in targets:
        print(f"\n{'─' * 50}")
        print(f"  Model: {model_name}  config={config_key}")
        print(f"{'─' * 50}")

        try:
            model_path = download_model(model_name)
            metrics = benchmark_model(
                model_path,
                prompts,
                max_tokens=max_tokens,
                track_warmup=track_warmup,
            )
            # Annotate with provenance so analysis scripts can group results.
            metrics["device"] = device_key
            metrics["prompt_set"] = prompt_set
            update_models_json(
                model_name,
                metrics,
                config_key=config_key,
                update_top_level=update_top_level,
            )
            print(f"[OK] {model_name} — {metrics}")

        except Exception as exc:
            print(f"[ERROR] {model_name} failed: {exc}")
            print("  Skipping this model. Existing values in models.json are preserved.")

    print("\n" + "=" * 60)
    print("  Data collection complete.")
    print(f"  Config key: {config_key}")
    print(f"  models.json and device_profiles.json have been updated.")
    print("=" * 60 + "\n")


# ============================================================
#  STANDALONE ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Automated Benchmark — Data Collection")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help="Subset of model names to collect (default: all in GGUF_REGISTRY).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per prompt (default: 128).",
    )
    parser.add_argument(
        "--prompt-set",
        choices=list(PROMPT_SETS.keys()),
        default="all",
        help="Which prompt set to use (default: all).",
    )
    parser.add_argument(
        "--track-warmup",
        action="store_true",
        help="Capture cold-start vs warm-state latency separately.",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="Optional override for the multi-config storage key in models.json.",
    )
    args = parser.parse_args()

    run_collection(
        model_names=args.models,
        max_tokens=args.max_tokens,
        prompt_set=args.prompt_set,
        track_warmup=args.track_warmup,
        config_name=args.config_name,
    )
