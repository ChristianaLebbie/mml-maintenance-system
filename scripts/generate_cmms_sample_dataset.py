"""Generate a SYNTHETIC, COMPLETE CMMS sample/test dataset.

IMPORTANT -- this is fabricated demonstration data, not real mine data. It
is a stand-in for the real Data Repo's PM/WO compliance snapshot (e.g.
"PM Spot Check.xlsx", see docs/dataset_mapping.md) with the same column
shape (Asset ID, Equipment Category, Manufacturer, Criticality, Overdue
PMs/WOs, Total Completed PMs/WOs, Downtime, Total Cost, Last Completed
PM/WO), but every row is fully populated -- no blanks, no formatting-bleed
artifacts. Its purpose is to let the full pipeline (ingestion -> mapping ->
feasibility -> feature engineering -> intervention-priority labeling ->
training -> SHAP) be
exercised and tested end-to-end without depending on the real, partially
messy Excel exports. Saved under data/raw/sample_test/ -- never merged with
or presented as the real Data Repo.

Simulation design: each synthetic asset gets a latent "maintenance
discipline" score (0-1, independent of criticality). Higher discipline
drives more completed PMs/WOs and more recent last-completed dates; lower
discipline drives more overdue PMs/WOs and more downtime. This gives the
downstream intervention-priority classification model genuine, learnable
structure (asset history predicts compliance risk) rather than pure noise,
while keeping the model's feature set (equipment_category, manufacturer,
criticality, total_completed_*, days_since_last_completed_*) fully
independent of the intervention-priority label's own inputs (overdue_pms,
overdue_wos, downtime_minutes) -- see project memory "modeling design
decisions".

Usage:
    python scripts/generate_cmms_sample_dataset.py
"""

from __future__ import annotations

import sys
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config.settings import PATHS, RANDOM_SEED

N_ASSETS = 200
REFERENCE_DATE = datetime.date(2026, 8, 26)

# (category, 2-letter code, downtime severity multiplier, cost multiplier)
CATEGORIES = [
    ("Gyratory Crusher", "CR", 2.2, 2.5),
    ("SAG Mill", "SM", 2.5, 3.0),
    ("Ball Mill", "BM", 2.3, 2.8),
    ("Conveyor", "CV", 1.0, 1.0),
    ("Pump", "PU", 1.2, 1.1),
    ("Motor", "MO", 1.1, 1.0),
    ("Transformer", "TR", 1.6, 1.8),
    ("Compressor", "CP", 1.3, 1.2),
    ("Screen", "SC", 1.0, 0.9),
    ("Cyclone", "CY", 0.8, 0.7),
    ("Level Transmitter", "LT", 0.4, 0.3),
    ("Pressure Transmitter", "PT", 0.4, 0.3),
    ("Flow Transmitter", "FT", 0.4, 0.3),
]
AREA_CODES = ["4212", "4310", "4415", "4520", "4630"]
MANUFACTURERS = [
    "Metso Outotec", "FLSmidth", "Sandvik", "Weir Minerals", "ABB",
    "Siemens", "Schneider Electric", "SKF", "Grundfos", "Gardner Denver",
]
CRITICALITIES = ["Critical", "High", "Medium", "Low"]
CRITICALITY_WEIGHTS = [0.15, 0.30, 0.35, 0.20]


def _simulate_asset(idx: int, rng: np.random.Generator) -> dict:
    category, code, downtime_mult, cost_mult = CATEGORIES[idx % len(CATEGORIES)]
    area = AREA_CODES[idx % len(AREA_CODES)]
    asset_id = f"{area}-{code}-{idx:03d}"

    manufacturer = MANUFACTURERS[rng.integers(0, len(MANUFACTURERS))]
    criticality = rng.choice(CRITICALITIES, p=CRITICALITY_WEIGHTS)

    # Latent maintenance-discipline score: 0 = badly neglected, 1 = pristine.
    discipline = float(np.clip(rng.normal(0.55, 0.22), 0.02, 0.98))

    total_completed_pms = int(rng.poisson(4 + discipline * 30))
    total_completed_wos = int(rng.poisson(2 + discipline * 14))
    overdue_pms = int(rng.poisson(max(0.1, (1 - discipline) * 9)))
    overdue_wos = int(rng.poisson(max(0.1, (1 - discipline) * 5)))

    days_since_pm = int(np.clip(rng.exponential(20 + (1 - discipline) * 160), 0, 700))
    days_since_wo = int(np.clip(rng.exponential(25 + (1 - discipline) * 180), 0, 700))
    last_completed_pm = REFERENCE_DATE - datetime.timedelta(days=days_since_pm)
    last_completed_wo = REFERENCE_DATE - datetime.timedelta(days=days_since_wo)

    base_downtime = rng.gamma(shape=1.5, scale=40) * (1 - discipline + 0.15)
    downtime_minutes = round(float(base_downtime * downtime_mult), 1)

    base_cost = rng.gamma(shape=2.0, scale=180) * (1 - discipline + 0.2)
    total_cost = round(float(base_cost * cost_mult), 2)

    return {
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
        "Last Completed PM": last_completed_pm.isoformat(),
        "Last Completed WO": last_completed_wo.isoformat(),
    }


def generate(seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [_simulate_asset(i, rng) for i in range(N_ASSETS)]
    return pd.DataFrame(rows)


def main() -> None:
    out_dir = PATHS["raw_excel"].parent / "sample_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate()
    out_path = out_dir / "cmms_sample_dataset.csv"
    df.to_csv(out_path, index=False)

    readme_path = out_dir / "README.md"
    readme_path.write_text(
        "# SYNTHETIC SAMPLE/TEST DATA -- NOT REAL MINE DATA\n\n"
        "Generated by scripts/generate_cmms_sample_dataset.py (fixed random "
        f"seed={RANDOM_SEED} for reproducibility). Shaped exactly like the "
        "real Data Repo's PM/WO compliance snapshot (Asset ID, Equipment "
        "Category, Manufacturer, Criticality, Overdue PMs/WOs, Total "
        "Completed PMs/WOs, Downtime, Total Cost, Last Completed PM/WO) but "
        "entirely fabricated and fully populated (no blanks), so the "
        "complete pipeline -- ingestion, mapping, feasibility, feature "
        "engineering, intervention-priority labeling, training, SHAP -- can be run and "
        "tested end-to-end without the real, partially messy Excel "
        "exports. Never merged with or presented as the real Marampa Mines "
        "Data Repo. See docs/dataset_mapping.md for the real schema this "
        "mirrors.\n\n"
        "To import:\n\n"
        "    python scripts/prepare_dataset.py "
        '"data/raw/sample_test/cmms_sample_dataset.csv" cmms_sample_test\n',
        encoding="utf-8",
    )

    print(f"Generated {len(df)} synthetic assets across {len(CATEGORIES)} equipment categories.")
    print(f"Criticality distribution:\n{df['Criticality'].value_counts()}")
    print(f"Saved to: {out_path}")
    print(f"Disclaimer written to: {readme_path}")


if __name__ == "__main__":
    main()
