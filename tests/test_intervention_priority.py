"""Phase 13 tests: intervention-priority classification from a probability +
fitted thresholds."""

from __future__ import annotations

from src.inference.priority_engine import classify_intervention_priority


def test_classify_intervention_priority_normal_below_watch_threshold():
    assert classify_intervention_priority(0.1, watch_threshold=0.3, high_threshold=0.7) == "Normal"


def test_classify_intervention_priority_watch_between_thresholds():
    assert classify_intervention_priority(0.5, watch_threshold=0.3, high_threshold=0.7) == "Watch"


def test_classify_intervention_priority_high_at_or_above_high_threshold():
    assert classify_intervention_priority(0.7, watch_threshold=0.3, high_threshold=0.7) == "High Priority"
    assert classify_intervention_priority(0.95, watch_threshold=0.3, high_threshold=0.7) == "High Priority"


def test_classify_intervention_priority_boundary_is_inclusive_for_watch():
    assert classify_intervention_priority(0.3, watch_threshold=0.3, high_threshold=0.7) == "Watch"
