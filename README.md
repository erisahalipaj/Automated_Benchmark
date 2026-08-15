# Automated Benchmark Pipeline for Quantized LLMs on Edge Devices

An automated, device-aware benchmarking pipeline that evaluates quantized
open-source large language models (LLMs) on resource-constrained edge hardware.
The pipeline characterizes the host device, downloads quantized **GGUF** models,
runs them against a standardized prompt workload, and ranks them with a composite
**Edge Suitability Score (ESS)** whose metric weights are derived automatically
via the **Entropy Weight Method (EWM)**. A suite of secondary analyses
(token scaling, cohort comparison, Pareto frontier, cold-start, prompt
complexity, parameter scaling, and weight sensitivity) is also included.

> This repository contains the **code, prompt sets, and requirements** only.
> Large model files, generated results, and the machine-specific data files
> are intentionally excluded (see [Data files](#data-files-not-tracked)).

---

## Repository structure

```
.
├── collect.py            # Device characterization, GGUF download, metric collection
├── pipeline.py           # Feasibility filter, ESS scoring, ranking, plots (entry point)
├── analysis.py           # Seven multi-configuration secondary analyses
├── clean_analytics.py    # Utility to reset/clean the analytics output
├── prompts.txt           # Full 99-prompt workload
├── prompts_simple.txt    # Simple subset (factual / short-answer)
├── prompts_complex.txt   # Complex subset (coding / multi-step reasoning)
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

The GGUF **model registry** (HuggingFace repo IDs, filenames, parameter counts,
and published quality scores) is defined directly in `collect.py`
(`GGUF_REGISTRY`), so the set of benchmarked models is part of the code.

---

## Requirements

- **Python 3.12**
- The packages listed in `requirements.txt`:
  `pandas`, `numpy`, `matplotlib`, `scipy`, `psutil`, `huggingface_hub`,
  `llama-cpp-python`.

> `llama-cpp-python` compiles a native backend. On some platforms you may need a
> C/C++ toolchain (e.g. build-essential / Xcode command-line tools / MSVC).
> See the [llama-cpp-python installation notes](https://github.com/abetlen/llama-cpp-python)
> if the wheel does not install directly.

### Setup

```bash
# 1. Clone
git clone https://github.com/erisahalipaj/Automated_Benchmark.git
cd Automated_Benchmark

# 2. Create and activate a virtual environment
python -m venv venv
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Data files (not tracked)

Two JSON files are required at runtime but are **git-ignored** because they hold
generated results and machine-specific data. Create them once before the first
run.

### `device_profiles.json`

An empty JSON object is enough; the collection stage fills it in automatically
by characterizing the current device:

```json
{}
```

After a run it will contain, for example:

```json
{
  "laptop": { "RAM": 15.2, "CPU": 12, "STO": 251.2 }
}
```

### `models.json`

A JSON array with one object per model. Seed it with the model `name` and its
static `quality` score (from public benchmarks); the collection stage overwrites
`mem_req`, `size`, `cpu_req`, `latency`, and `throughput` with real measurements.
The seed below matches the registry in `collect.py`:

```json
[
  { "name": "TinyLlama", "params": 1.1, "quality": 0.80, "mem_req": 2, "size": 1, "cpu_req": 2 },
  { "name": "Qwen2",     "params": 1.5, "quality": 0.82, "mem_req": 2, "size": 1, "cpu_req": 2 },
  { "name": "Yi",        "params": 6.0, "quality": 0.79, "mem_req": 6, "size": 4, "cpu_req": 4 },
  { "name": "Gemma",     "params": 2.6, "quality": 0.85, "mem_req": 3, "size": 2, "cpu_req": 2 },
  { "name": "Llama3",    "params": 8.0, "quality": 0.88, "mem_req": 6, "size": 5, "cpu_req": 4 },
  { "name": "Mistral",   "params": 7.2, "quality": 0.91, "mem_req": 6, "size": 5, "cpu_req": 4 },
  { "name": "SmolLM2",   "params": 1.7, "quality": 0.81, "mem_req": 2, "size": 1, "cpu_req": 2 },
  { "name": "Qwen2.5",   "params": 3.0, "quality": 0.86, "mem_req": 3, "size": 2, "cpu_req": 2 },
  { "name": "Phi3.5",    "params": 3.8, "quality": 0.89, "mem_req": 4, "size": 3, "cpu_req": 2 },
  { "name": "Llama3.2",  "params": 3.2, "quality": 0.87, "mem_req": 4, "size": 2, "cpu_req": 2 }
]
```

The `mem_req`, `size`, and `cpu_req` seed values are only placeholders used by
the feasibility filter before real measurements exist; running `--collect`
replaces them with measured values.

---

## Usage

The main entry point is `pipeline.py`.

```bash
# Full run: characterize device, download models, measure metrics, then score
python pipeline.py --collect

# Score only, using data already in models.json
python pipeline.py

# Collect a subset of models
python pipeline.py --collect --models TinyLlama Qwen2

# Multi-configuration collection (token scaling / prompt subset)
python pipeline.py --collect --max-tokens 200 --prompt-set complex

# Run all seven secondary analyses without re-scoring
python pipeline.py --skip-pipeline --analyze all

# Run a single analysis
python analysis.py scaling
```

### Command-line arguments

| Argument | Effect |
| --- | --- |
| `--collect` | Characterize the device, download models, and measure metrics before scoring. |
| `--models NAMES` | Restrict collection to the listed models (e.g. `TinyLlama Qwen2`). |
| `--max-tokens N` | Maximum output length per prompt (default: 128). |
| `--prompt-set {all,simple,complex}` | Select the prompt subset. |
| `--track-warmup` | Capture cold-start vs warm-state latency. |
| `--analyze NAME` | Run one of the seven analyses (or `all`). |
| `--skip-pipeline` | Skip ESS scoring and run only the requested analysis. |

---

## Reproducing the experiment

1. Complete the [Setup](#setup) steps.
2. Create `device_profiles.json` (`{}`) and `models.json` (seed above).
3. Run the full baseline collection and scoring:
   ```bash
   python pipeline.py --collect
   ```
   Models are downloaded to `model_cache/` on first use and cached thereafter.
4. (Optional) Collect the additional configurations used by the secondary
   analyses:
   ```bash
   python pipeline.py --collect --max-tokens 50  --prompt-set all
   python pipeline.py --collect --max-tokens 100 --prompt-set all
   python pipeline.py --collect --max-tokens 200 --prompt-set all
   python pipeline.py --collect --prompt-set simple
   python pipeline.py --collect --prompt-set complex
   python pipeline.py --collect --track-warmup
   ```
5. Generate all analyses and plots:
   ```bash
   python pipeline.py --skip-pipeline --analyze all
   ```

### Outputs

Per-device ESS charts and all analysis figures/CSVs are written to the
`analytics/` directory (git-ignored). Results are keyed per configuration inside
`models.json` under a nested `benchmarks` dictionary
(e.g. `laptop_t128_pall`).

---

## Notes

- Inference is **CPU-only** via `llama-cpp-python`; no GPU offloading is used.
- All models are benchmarked at **Q4_K_M** quantization.
- `model_cache/`, `analytics/`, `models.json`, and `device_profiles.json` are
  regenerated locally and are not tracked in git.
