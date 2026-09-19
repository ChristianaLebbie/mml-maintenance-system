# Synthetic Demonstration Track

## What this is, and why it exists

The real research dataset (`Data Repo/`, a Limble CMMS export from Marampa
Mines -- see `docs/dataset_mapping.md`) has no continuous sensor telemetry
and no populated failure/breakdown log. That's why the thesis's primary,
real-data-backed deliverable is **CMMS-based maintenance-risk
classification** (see README "Primary ML task"). Several parts of the
original project design -- lag/rolling/trend features, near-term
failure-horizon labels, chronological validation, and an LSTM/GRU sequence
model -- genuinely require repeated per-machine time-series observations,
which the real data doesn't have.

Rather than fabricate those observations *inside* the real Data Repo
(which would mean presenting invented values as genuine mine records --
an academic-integrity violation this project explicitly avoids, see
project memory), this track adds a **separate, clearly-labeled synthetic
dataset** whose only purpose is to demonstrate that the rest of the
project's methodology (the parts the real CMMS snapshot cannot exercise)
is implemented correctly. It is never merged with, compared against, or
presented as the real Data Repo results.

**Every artifact from this track must be labeled synthetic.** The dataset
lives under `data/raw/synthetic/` with its own `README.md` disclaimer; the
generator script's docstring says so; this doc says so.

## How it's generated

`scripts/generate_synthetic_dataset.py` simulates 25 machines over 120
days of hourly sensor readings (voltage, rotation, pressure, vibration),
each with 2-4 randomly-timed degradation episodes: sensors drift from a
healthy baseline over a random 48-168 hour ramp, ending in a labeled
failure event, after which the machine resets to a fresh healthy baseline
(simulating a repair). This produces genuine, learnable pre-failure signal
rather than pure noise, with a fixed random seed for reproducibility.

```powershell
python scripts/generate_synthetic_dataset.py
```

## What it demonstrates

| Original design element | Real CMMS data | Synthetic demo data |
|---|---|---|
| Lag/rolling features (`src/features/lag_features.py`) | Honest no-op (single snapshot per machine) | Real: per-machine, chronologically-sorted lags and rolling stats |
| Trend features (`src/features/temporal_features.py:build_trend_features`) | N/A | Real: diff/pct-change/EWMA per machine |
| Near-term failure label (`src/labels/failure_labels.py:construct_failure_within_horizon_label`) | Not used (no failure log) | Real: label = 1 if failure occurs within `(t, t+H]` for that machine |
| Chronological split (`src/training/validation.py:chronological_split`) | Not used (no time order between rows) | Real: sorted by timestamp, train strictly precedes val precedes test |
| LSTM/GRU (`src/training/sequence_model.py`) | Not implemented (no sequence exists) | Real: trained on sliding windows of past-only observations |

## Running the full demo track

```powershell
python scripts/generate_synthetic_dataset.py
python scripts/train_sequence_model.py 24 24
```

The second command trains both the LSTM sequence model and a flat Random
Forest baseline (using the same lag/rolling/trend features, flattened per
row) on an identical chronological split, so the comparison between a
"general"/flat model and a genuine sequence model (section 52) is
apples-to-apples. Metrics are saved to
`reports/experiments/synthetic_sequence_demo/`; the trained LSTM to
`models/trained/lstm_synthetic_demo.keras`.

## Actual results from a real run (24h horizon, 24h sequence length)

From `python scripts/train_sequence_model.py 24 24` against the generated
dataset (fixed seed -- reproducible, not invented):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Random Forest (flat, lag/rolling/trend features) | 0.985 | 0.729 | 0.651 | 0.688 | 0.992 | 0.796 |
| LSTM (sequence) | 0.973 | 0.477 | 0.948 | 0.634 | 0.993 | 0.778 |

A genuine, realistic tradeoff: the LSTM catches far more of the actual
failures (recall 0.948 vs 0.651) at the cost of more false alarms
(precision 0.477 vs 0.729). Both are strong (ROC-AUC ~0.99) because the
synthetic degradation signal is intentionally clean -- do not read these
numbers as evidence about real equipment reliability (see below).

## What is NOT claimed

- This is not evidence about the real mine's equipment reliability --
  the failure events are simulated, not observed.
- No thesis conclusion about Marampa Mines' actual assets should cite
  numbers from this track.
- It exists solely to prove the codebase correctly implements
  time-series-dependent methodology that the real data cannot exercise.
