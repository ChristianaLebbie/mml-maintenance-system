"""Frequency-domain feature group (FFT energy, dominant frequency, etc.).

Not applicable: these require high-frequency vibration waveform data (e.g.
PHM 2012 bearing dataset), which this project's CMMS export does not
contain. Per the master project's own rule ("do not use FFT on hourly
Azure-style telemetry merely because FFT exists in the thesis"), this
module stays a no-op until a genuine vibration-waveform source is
integrated.
"""

from __future__ import annotations

import pandas as pd


def build_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(index=df.index)
