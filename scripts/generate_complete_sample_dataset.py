"""Generate a SYNTHETIC, COMPLETE mining maintenance dataset covering every
canonical field the pipeline understands.

IMPORTANT -- this is fabricated demonstration data, not real mine data. The
two existing sample datasets each cover one half of the schema:
data/raw/sample_test/cmms_sample_dataset.csv (CMMS compliance snapshot,
cross-sectional -- see scripts/generate_cmms_sample_dataset.py) and
data/raw/synthetic/synthetic_sensor_telemetry.csv (sensor time series --
see scripts/generate_synthetic_dataset.py). This script produces one
dataset with BOTH: for each synthetic asset, an hourly sensor time series
(voltage, rotation, pressure, vibration, timestamped failure events) *and*
a current CMMS compliance state (overdue PMs/WOs, completion history,
downtime, cost, last-completed dates) attached to every row for that
asset -- so every field in src/ingestion/mapper.py's INTERNAL_FIELDS is
fully populated with zero missingness, and every supported task
(binary_failure_classification, intervention_priority_classification,
anomaly_detection_across_machines, time_series_analysis) can be exercised
from a single upload. Saved under data/raw/complete_sample/ -- never
merged with or presented as the real Data Repo.

Simulation design: each asset gets a latent "maintenance discipline" score
(0-1, see scripts/generate_cmms_sample_dataset.py for the compliance-side
rationale) that drives both its CMMS compliance numbers AND how often/
severely its sensor readings degrade toward a failure event -- so a
poorly-disciplined asset (high overdue PMs/WOs, more downtime) genuinely
tends to fail more often in the simulated sensor stream too. This gives a
thesis-relevant, non-trivial correlation between the CMMS-based
intervention-priority score and real failure occurrence, without being a
deterministic/fabricated
link -- both signals still carry their own independent noise.

Usage:
    python scripts/generate_complete_sample_dataset.py
"""

from __future__ import annotations

import sys
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config.settings import PATHS, RANDOM_SEED

N_MACHINES = 25
DAYS = 120
HOURS_PER_DAY = 24
REFERENCE_DATE = datetime.date(2026, 8, 26)

# (category, 2-letter code, downtime severity mult, cost mult, sensor baseline profile)
# sensor baseline: (voltage_mean, voltage_std, rotation_mean, rotation_std,
#                    pressure_mean, pressure_std, vibration_mean, vibration_std)
CATEGORIES = [
    ("Gyratory Crusher", "CR", 2.2, 2.5, (415.0, 6.0, 380.0, 8.0, 7.5, 0.4, 3.2, 0.5)),
    ("SAG Mill", "SM", 2.5, 3.0, (420.0, 6.5, 12.0, 0.5, 8.0, 0.5, 3.5, 0.6)),
    ("Ball Mill", "BM", 2.3, 2.8, (418.0, 6.0, 16.0, 0.6, 7.8, 0.5, 3.4, 0.55)),
    ("Conveyor", "CV", 1.0, 1.0, (400.0, 5.0, 1480.0, 15.0, 5.0, 0.3, 2.2, 0.35)),
    ("Pump", "PU", 1.2, 1.1, (400.0, 5.0, 1750.0, 18.0, 6.5, 0.35, 2.6, 0.4)),
    ("Motor", "MO", 1.1, 1.0, (400.0, 5.0, 1480.0, 15.0, 0.0, 0.0, 2.3, 0.35)),
    ("Transformer", "TR", 1.6, 1.8, (11000.0, 80.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2)),
    ("Compressor", "CP", 1.3, 1.2, (400.0, 5.0, 2950.0, 20.0, 9.0, 0.5, 3.0, 0.45)),
    ("Screen", "SC", 1.0, 0.9, (400.0, 5.0, 950.0, 12.0, 0.0, 0.0, 4.0, 0.6)),
    ("Cyclone", "CY", 0.8, 0.7, (0.0, 0.0, 0.0, 0.0, 4.5, 0.3, 1.8, 0.3)),
    ("Level Transmitter", "LT", 0.4, 0.3, (24.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.3, 0.1)),
    ("Pressure Transmitter", "PT", 0.4, 0.3, (24.0, 0.5, 0.0, 0.0, 6.0, 0.3, 0.3, 0.1)),
    ("Flow Transmitter", "FT", 0.4, 0.3, (24.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.3, 0.1)),
]
AREA_CODES = ["4212", "4310", "4415", "4520", "4630"]
MANUFACTURERS = [
    "Metso Outotec", "FLSmidth", "Sandvik", "Weir Minerals", "ABB",
    "Siemens", "Schneider Electric", "SKF", "Grundfos", "Gardner Denver",
]
CRITICALITIES = ["Critical", "High", "Medium", "Low"]
CRITICALITY_WEIGHTS = [0.15, 0.30, 0.35, 0.20]


def _simulate_sensor_stream(
    discipline: float, baseline: tuple[float, ...], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_hours = DAYS * HOURS_PER_DAY
    v_mean, v_std, r_mean, r_std, p_mean, p_std, vib_mean, vib_std = baseline
    timestamps = pd.date_range("2025-01-01", periods=n_hours, freq="h")

    voltage = rng.normal(v_mean, v_std, size=n_hours)
    rotation = rng.normal(r_mean, r_std, size=n_hours)
    pressure = rng.normal(p_mean, p_std, size=n_hours)
    vibration = rng.normal(vib_mean, vib_std, size=n_hours)
    failure_event = np.zeros(n_hours, dtype=int)

    # Poorer discipline -> more frequent, more severe degradation episodes.
    n_episodes = int(rng.integers(1, 3) + round((1 - discipline) * 3))
    episode_starts = sorted(rng.integers(200, n_hours - 200, size=max(n_episodes, 1)))
    severity = 0.5 + (1 - discipline) * 1.5
    for start in episode_starts:
        ramp_len = int(rng.integers(48, 168))
        end = min(start + ramp_len, n_hours - 1)
        ramp = np.linspace(0, 1, end - start)
        vibration[start:end] += ramp * rng.uniform(4.0, 8.0) * severity
        pressure[start:end] += ramp * rng.uniform(1.0, 2.5) * severity
        rotation[start:end] -= ramp * rng.uniform(50, 120) * severity
        failure_event[end] = 1

    return timestamps.values, voltage, rotation, pressure, vibration, failure_event


def _simulate_machine(idx: int, rng: np.random.Generator) -> pd.DataFrame:
    category, code, downtime_mult, cost_mult, baseline = CATEGORIES[idx % len(CATEGORIES)]
    area = AREA_CODES[idx % len(AREA_CODES)]
    asset_id = f"{area}-{code}-{idx:03d}"

    manufacturer = MANUFACTURERS[rng.integers(0, len(MANUFACTURERS))]
    criticality = rng.choice(CRITICALITIES, p=CRITICALITY_WEIGHTS)

    discipline = float(np.clip(rng.normal(0.55, 0.22), 0.02, 0.98))

    total_completed_pms = int(rng.poisson(4 + discipline * 30))
    total_completed_wos = int(rng.poisson(2 + discipline * 14))
    overdue_pms = int(rng.poisson(max(0.1, (1 - discipline) * 9)))
    overdue_wos = int(rng.poisson(max(0.1, (1 - discipline) * 5)))

    days_since_pm = int(np.clip(rng.exponential(20 + (1 - discipline) * 160), 0, 700))
    days_since_wo = int(np.clip(rng.exponential(25 + (1 - discipline) * 180), 0, 700))
    last_completed_pm = (REFERENCE_DATE - datetime.timedelta(days=days_since_pm)).isoformat()
    last_completed_wo = (REFERENCE_DATE - datetime.timedelta(days=days_since_wo)).isoformat()

    base_downtime = rng.gamma(shape=1.5, scale=40) * (1 - discipline + 0.15)
    downtime_minutes = round(float(base_downtime * downtime_mult), 1)

    base_cost = rng.gamma(shape=2.0, scale=180) * (1 - discipline + 0.2)
    total_cost = round(float(base_cost * cost_mult), 2)

    timestamps, voltage, rotation, pressure, vibration, failure_event = _simulate_sensor_stream(
        discipline, baseline, rng
    )
    n_hours = len(timestamps)

    return pd.DataFrame(
        {
            "Asset ID": asset_id,
            "Equipment Category": category,
            "Manufacturer": manufacturer,
            "Criticality": criticality,
            "Overdue PMs": overdue_pms,
            "Overdue WOs": overdue_wos,
            "Total Completed PMs": total_completed_pms,
            "Total Completed WOs": total_completed_wos,
            "Downtime": downtime_minutes,
            "Total Cost": total_cost,
            "Last Completed PM": last_completed_pm,
            "Last Completed WO": last_completed_wo,
            "Reading Time": timestamps,
            "Voltage": voltage,
            "Rotation": rotation,
            "Pressure": pressure,
            "Vibration": vibration,
            "Failure Event": failure_event,
        }
    )


def generate(seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = [_simulate_machine(i, rng) for i in range(N_MACHINES)]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    out_dir = PATHS["raw_excel"].parent / "complete_sample"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate()
    out_path = out_dir / "complete_mining_dataset.csv"
    df.to_csv(out_path, index=False)

    readme_path = out_dir / "README.md"
    readme_path.write_text(
        "# SYNTHETIC COMPLETE SAMPLE DATA -- NOT REAL MINE DATA\n\n"
        "Generated by scripts/generate_complete_sample_dataset.py (fixed "
        f"random seed={RANDOM_SEED} for reproducibility). Combines both "
        "halves of the pipeline's schema in one file -- CMMS compliance "
        "state (Overdue PMs/WOs, Total Completed PMs/WOs, Downtime, Total "
        "Cost, Last Completed PM/WO) and hourly sensor telemetry (Voltage, "
        "Rotation, Pressure, Vibration) with real timestamped failure "
        "events -- across 25 synthetic assets styled after the real "
        "Marampa Mines equipment fleet (crushers, mills, conveyors, "
        "pumps, motors, transformers, compressors, screens, cyclones, "
        "transmitters). Every field is fully populated (zero missingness) "
        "so every supported modelling task (binary failure classification, "
        "intervention-priority classification, cross-machine anomaly detection, "
        "time-series analysis) can be exercised from one upload. Entirely "
        "fabricated -- never merged with or presented as the real Marampa "
        "Mines Data Repo. See docs/dataset_mapping.md for the schema this "
        "mirrors.\n\n"
        "To import:\n\n"
        "    python scripts/prepare_dataset.py "
        '"data/raw/complete_sample/complete_mining_dataset.csv" complete_sample_test\n',
        encoding="utf-8",
    )

    print(f"Generated {len(df)} rows across {N_MACHINES} synthetic machines.")
    print(f"Failure events: {df['Failure Event'].sum()}")
    print(f"Missing values: {int(df.isnull().sum().sum())}")
    print(f"Saved to: {out_path}")
    print(f"Disclaimer written to: {readme_path}")


if __name__ == "__main__":
    main()
