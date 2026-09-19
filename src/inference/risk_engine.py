"""Maps a model's output probability to a risk level using thresholds that
were fit during training (never invented at inference time -- section 32)."""

from __future__ import annotations


def classify_risk(probability: float, watch_threshold: float, high_threshold: float) -> str:
    if probability >= high_threshold:
        return "High Risk"
    if probability >= watch_threshold:
        return "Watch"
    return "Normal"
