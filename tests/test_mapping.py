"""Phase 4 tests: column-mapping suggestion, validation, application, and
persistence to config/datasets.yaml."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion import mapper


@pytest.fixture()
def sample_df():
    return pd.DataFrame(
        {
            "Asset ID": ["4212-CR-001", "4212-PU-002"],
            "Equipment Category": ["Crusher", "Pump"],
            "Overdue PMs": [2, 0],
            "Overdue WOs": [1, 0],
            "Last Completed PM": ["2025/01/03", "2025/01/10"],
            "Irrelevant Notes": ["a", "b"],
        }
    )


def test_suggest_mapping_finds_known_aliases(sample_df):
    mapping = mapper.suggest_mapping(list(sample_df.columns))
    assert mapping["machine_id"] == "Asset ID"
    assert mapping["equipment_category"] == "Equipment Category"
    assert mapping["overdue_pms"] == "Overdue PMs"
    assert mapping["overdue_wos"] == "Overdue WOs"
    assert mapping["last_completed_pm"] == "Last Completed PM"


def test_suggest_mapping_leaves_unmatched_fields_none(sample_df):
    mapping = mapper.suggest_mapping(list(sample_df.columns))
    assert mapping["manufacturer"] is None
    assert mapping["failure_label"] is None


def test_suggest_mapping_matches_underscored_column_names():
    # Real bug: "machine_id" (underscore) previously failed to match the
    # "machine id" (space) alias because normalization didn't unify them.
    mapping = mapper.suggest_mapping(["machine_id", "voltage", "failure_event"])
    assert mapping["machine_id"] == "machine_id"
    assert mapping["voltage"] == "voltage"
    assert mapping["failure_label"] == "failure_event"


def test_suggest_mapping_does_not_reuse_a_column_twice():
    # "Overdue PMs" and "Overdue WOs" must not both match the same column.
    columns = ["Asset ID", "Overdue PMs", "Overdue WOs"]
    mapping = mapper.suggest_mapping(columns)
    assert mapping["overdue_pms"] != mapping["overdue_wos"]


def test_validate_mapping_flags_missing_required_field():
    problems = mapper.validate_mapping({"machine_id": None}, ["Asset ID"])
    assert any("machine_id" in p and "not mapped" in p for p in problems)


def test_validate_mapping_flags_column_not_in_dataset():
    mapping = {"machine_id": "Asset ID", "equipment_category": "Nonexistent Column"}
    problems = mapper.validate_mapping(mapping, ["Asset ID"])
    assert any("Nonexistent Column" in p for p in problems)


def test_validate_mapping_passes_for_good_mapping():
    mapping = {"machine_id": "Asset ID"}
    assert mapper.validate_mapping(mapping, ["Asset ID"]) == []


def test_apply_mapping_produces_canonical_columns(sample_df):
    mapping = mapper.suggest_mapping(list(sample_df.columns))
    standardized = mapper.apply_mapping(sample_df, mapping, source_dataset="test-source")

    assert list(standardized["machine_id"]) == ["4212-CR-001", "4212-PU-002"]
    assert list(standardized["equipment_category"]) == ["Crusher", "Pump"]
    assert (standardized["source_dataset"] == "test-source").all()
    assert standardized["manufacturer"].isna().all()


def test_apply_mapping_carries_through_unmapped_columns_as_extra(sample_df):
    mapping = mapper.suggest_mapping(list(sample_df.columns))
    standardized = mapper.apply_mapping(sample_df, mapping, source_dataset="test-source")

    # "Irrelevant Notes" doesn't match any known field's aliases.
    assert "extra__irrelevant_notes" in standardized.columns
    assert list(standardized["extra__irrelevant_notes"]) == ["a", "b"]


def test_apply_mapping_include_extra_false_drops_unmapped_columns(sample_df):
    mapping = mapper.suggest_mapping(list(sample_df.columns))
    standardized = mapper.apply_mapping(
        sample_df, mapping, source_dataset="test-source", include_extra=False
    )
    assert not any(c.startswith("extra__") for c in standardized.columns)


def test_apply_mapping_sanitizes_extra_column_names():
    df = pd.DataFrame({"Asset ID": ["A1"], "Weird Column!! (mm)": [5]})
    mapping = {"machine_id": "Asset ID"}
    standardized = mapper.apply_mapping(df, mapping, source_dataset="x")
    assert "extra__weird_column_mm" in standardized.columns


def test_apply_mapping_never_drops_data_for_an_unknown_schema():
    # Simulates a genuinely different mining dataset with no columns
    # matching any known CMMS alias except the machine identifier.
    df = pd.DataFrame(
        {
            "Equip No": ["X1", "X2"],
            "Shift Utilization Hours": [7.5, 8.0],
            "Spare Parts Count": [3, 1],
        }
    )
    mapping = mapper.suggest_mapping(list(df.columns))
    mapping["machine_id"] = "Equip No"  # only one auto-mappable field
    standardized = mapper.apply_mapping(df, mapping, source_dataset="other-mine.csv")

    assert list(standardized["machine_id"]) == ["X1", "X2"]
    assert "extra__shift_utilization_hours" in standardized.columns
    assert "extra__spare_parts_count" in standardized.columns
    assert list(standardized["extra__spare_parts_count"]) == [3, 1]


def test_apply_mapping_coerces_numeric_field_sentinel_strings_to_nan():
    # Real bug: the peer-adjusted-analysis columns (e.g.
    # Global_Anomaly_Score) mix real numbers with a text sentinel like
    # "Not Applicable (...)" for rows the analysis doesn't apply to --
    # that must become NaN, not silently produce a non-numeric column
    # that later crashes writing the dataset to Parquet.
    df = pd.DataFrame(
        {
            "Asset ID": ["A1", "A2"],
            "Global Anomaly Score": [0.32, "Not Applicable (area/system node)"],
        }
    )
    mapping = {"machine_id": "Asset ID", "global_anomaly_score": "Global Anomaly Score"}
    standardized = mapper.apply_mapping(df, mapping, source_dataset="x")

    assert pd.api.types.is_numeric_dtype(standardized["global_anomaly_score"])
    assert standardized["global_anomaly_score"].iloc[0] == 0.32
    assert pd.isna(standardized["global_anomaly_score"].iloc[1])


def test_apply_mapping_normalizes_mixed_type_extra_column_to_string():
    # Real bug: an extra__ pass-through column that mixes str and int
    # (e.g. a "Serial Number" column where some values were entered as
    # bare numbers) also crashed writing the dataset to Parquet.
    df = pd.DataFrame({"Asset ID": ["A1", "A2", "A3"], "Serial Number": ["SN-001", 4021, None]})
    mapping = {"machine_id": "Asset ID"}
    standardized = mapper.apply_mapping(df, mapping, source_dataset="x")

    col = standardized["extra__serial_number"]
    non_null = col.dropna()
    assert all(isinstance(v, str) for v in non_null)
    assert list(non_null) == ["SN-001", "4021"]
    assert col.isna().sum() == 1


def test_apply_mapping_leaves_uniformly_typed_extra_column_untouched(sample_df):
    # A homogeneous numeric extra column must stay numeric (and therefore
    # still eligible for src.features.raw_features.build_extra_numeric_features)
    # -- the mixed-type normalization must not over-apply.
    df = pd.DataFrame({"Asset ID": ["A1", "A2"], "Spare Parts Count": [3, 1]})
    mapping = {"machine_id": "Asset ID"}
    standardized = mapper.apply_mapping(df, mapping, source_dataset="x")
    assert pd.api.types.is_numeric_dtype(standardized["extra__spare_parts_count"])


def test_apply_mapping_raises_when_required_field_unmapped(sample_df):
    mapping = mapper.suggest_mapping(list(sample_df.columns))
    mapping["machine_id"] = None
    with pytest.raises(ValueError):
        mapper.apply_mapping(sample_df, mapping, source_dataset="test-source")


def test_save_and_load_mapping_roundtrip(tmp_path, monkeypatch):
    fake_config_path = tmp_path / "datasets.yaml"
    monkeypatch.setattr(mapper, "CONFIG_PATH", fake_config_path)

    mapping = {"machine_id": "Asset ID", "equipment_category": "Equipment Category"}
    mapper.save_mapping("my-dataset", "excel", "assets.xlsx", mapping)

    loaded = mapper.load_mapping("my-dataset")
    assert loaded["source_type"] == "excel"
    assert loaded["source_filename"] == "assets.xlsx"
    assert loaded["mapping"] == mapping


def test_load_mapping_returns_none_for_unknown_dataset(tmp_path, monkeypatch):
    fake_config_path = tmp_path / "datasets.yaml"
    monkeypatch.setattr(mapper, "CONFIG_PATH", fake_config_path)
    assert mapper.load_mapping("nonexistent") is None


def test_delete_mapping_removes_saved_mapping(tmp_path, monkeypatch):
    fake_config_path = tmp_path / "datasets.yaml"
    monkeypatch.setattr(mapper, "CONFIG_PATH", fake_config_path)

    mapper.save_mapping("my-dataset", "excel", "assets.xlsx", {"machine_id": "Asset ID"})
    assert mapper.load_mapping("my-dataset") is not None

    removed = mapper.delete_mapping("my-dataset")

    assert removed is True
    assert mapper.load_mapping("my-dataset") is None


def test_delete_mapping_returns_false_when_nothing_to_remove(tmp_path, monkeypatch):
    fake_config_path = tmp_path / "datasets.yaml"
    monkeypatch.setattr(mapper, "CONFIG_PATH", fake_config_path)
    assert mapper.delete_mapping("never-existed") is False


def test_delete_mapping_leaves_other_datasets_untouched(tmp_path, monkeypatch):
    fake_config_path = tmp_path / "datasets.yaml"
    monkeypatch.setattr(mapper, "CONFIG_PATH", fake_config_path)

    mapper.save_mapping("dataset-a", "excel", "a.xlsx", {"machine_id": "Asset ID"})
    mapper.save_mapping("dataset-b", "excel", "b.xlsx", {"machine_id": "Asset ID"})

    mapper.delete_mapping("dataset-a")

    assert mapper.load_mapping("dataset-a") is None
    assert mapper.load_mapping("dataset-b") is not None
