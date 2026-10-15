
"""Illustrative rainfall suitability rules for crops in Uganda."""

import numpy as np


class CropRule:
    """Classify monthly rainfall using crop-specific thresholds."""

    def __init__(
        self,
        name: str,
        minimum_rainfall: float,
        maximum_rainfall: float
    ):
        """Store and validate rainfall thresholds in mm/month."""

        if not name or not name.strip():
            raise ValueError("Crop name cannot be empty.")

        if not (
            np.isfinite(minimum_rainfall)
            and np.isfinite(maximum_rainfall)
        ):
            raise ValueError("Rainfall thresholds must be finite.")

        if minimum_rainfall < 0:
            raise ValueError("Minimum rainfall cannot be negative.")

        if maximum_rainfall <= minimum_rainfall:
            raise ValueError(
                "Maximum rainfall must exceed minimum rainfall."
            )

        self.name = name
        self.minimum_rainfall = float(minimum_rainfall)
        self.maximum_rainfall = float(maximum_rainfall)

    def classify(self, rainfall: float) -> str:
        """Classify rainfall for one month."""

        if not np.isfinite(rainfall) or rainfall < 0:
            raise ValueError(
                "Rainfall must be finite and non-negative."
            )

        if rainfall < self.minimum_rainfall:
            return "Drought risk"

        if rainfall > self.maximum_rainfall:
            return "Excess rainfall risk"

        return "Suitable rainfall"

    def classify_year(self, rainfall_values) -> list:
        """Classify all 12 months of rainfall."""

        values = np.asarray(rainfall_values, dtype=float)

        if values.shape != (12,):
            raise ValueError(
                "Exactly 12 monthly rainfall values are required."
            )

        return [
            self.classify(value)
            for value in values
        ]

    def __repr__(self) -> str:
        """Return a readable description of the crop rule."""

        return (
            f"CropRule(name={self.name!r}, "
            f"range={self.minimum_rainfall}-"
            f"{self.maximum_rainfall} mm)"
        )


# Simplified screening thresholds for the assignment.
# These are modelling assumptions, not official FAO thresholds.
CROP_RULES = [
    CropRule("Maize", 100, 200),
    CropRule("Beans", 75, 150),
    CropRule("Coffee", 100, 200)
]
