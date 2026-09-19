"""Phase 13 tests: risk-level classification from a probability + fitted
thresholds."""

from __future__ import annotations

from src.inference.risk_engine import classify_risk


def test_classify_risk_normal_below_watch_threshold():
    assert classify_risk(0.1, watch_threshold=0.3, high_threshold=0.7) == "Normal"


def test_classify_risk_watch_between_thresholds():
    assert classify_risk(0.5, watch_threshold=0.3, high_threshold=0.7) == "Watch"


def test_classify_risk_high_at_or_above_high_threshold():
    assert classify_risk(0.7, watch_threshold=0.3, high_threshold=0.7) == "High Risk"
    assert classify_risk(0.95, watch_threshold=0.3, high_threshold=0.7) == "High Risk"


def test_classify_risk_boundary_is_inclusive_for_watch():
    assert classify_risk(0.3, watch_threshold=0.3, high_threshold=0.7) == "Watch"
