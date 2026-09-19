"""Shared chart color palette for every Plotly chart in the Streamlit app.

Values are the dataviz skill's validated reference palette, used unchanged
(no new hex values introduced, so no re-validation is needed). Rules
applied throughout the app's charts:
- Categorical hues are assigned in this fixed order, never cycled or
  assigned by rank (see MODEL_COLORS: a color always means the same model).
- Intervention-priority levels use the reserved status palette
  (good/warning/critical), never the categorical slots, so a priority color
  never impersonates a series.
- Magnitude (missingness %, confusion-matrix counts) uses the single-hue
  sequential blue ramp, light -> dark -- never a rainbow.
"""

from __future__ import annotations

CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# Sequential blue ramp, step 100 (near-surface/low) -> step 700 (dark/high).
SEQUENTIAL_BLUE = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]

MUTED_INK = "#898781"

# Fixed model order/coloring -- a model's color never changes based on
# which other models are shown alongside it (color follows the entity).
MODEL_ORDER = ["logistic_regression", "decision_tree", "random_forest", "xgboost", "lstm"]
MODEL_COLORS = {name: CATEGORICAL[i] for i, name in enumerate(MODEL_ORDER)}

INTERVENTION_PRIORITY_ORDER = ["Normal", "Watch", "High Priority"]
INTERVENTION_PRIORITY_COLORS = {
    "Normal": STATUS["good"],
    "Watch": STATUS["warning"],
    "High Priority": STATUS["critical"],
}


def model_color(name: str) -> str:
    return MODEL_COLORS.get(name, MUTED_INK)


def priority_color(level: str) -> str:
    return INTERVENTION_PRIORITY_COLORS.get(level, MUTED_INK)


def sequential_blue_scale() -> list[list]:
    """A Plotly-compatible continuous colorscale (list of [position, hex])
    built from the sequential blue ramp."""
    n = len(SEQUENTIAL_BLUE)
    return [[i / (n - 1), hex_value] for i, hex_value in enumerate(SEQUENTIAL_BLUE)]
