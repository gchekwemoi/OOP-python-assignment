
"""Revenue risk assessment for a fish-export cooperative."""

import statistics as stats

import numpy as np


class RiskAssessor:
    """Measure revenue volatility and classify financial risk."""

    def __init__(
        self,
        low_threshold: float = 10.0,
        high_threshold: float = 25.0
    ):
        """Configure illustrative CV-based risk thresholds."""

        if not (
            np.isfinite(low_threshold)
            and np.isfinite(high_threshold)
        ):
            raise ValueError("Risk thresholds must be finite.")

        if not 0 <= low_threshold < high_threshold:
            raise ValueError("Risk thresholds must be increasing.")

        self.low_threshold = float(low_threshold)
        self.high_threshold = float(high_threshold)

    def summarise(self, revenues) -> dict:
        """Calculate statistics for weekly revenue observations."""

        values = np.asarray(revenues, dtype=float)

        if values.ndim != 1 or len(values) < 2:
            raise ValueError(
                "Provide at least two weekly revenue observations."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError("Revenues must be finite.")

        if np.any(values < 0):
            raise ValueError("Revenue cannot be negative.")

        revenue_list = values.tolist()

        mean = stats.mean(revenue_list)
        median = stats.median(revenue_list)
        variance = stats.variance(revenue_list)
        std_dev = stats.stdev(revenue_list)

        cv = (std_dev / mean * 100) if mean > 0 else 0.0

        return {
            "mean": mean,
            "median": median,
            "variance": variance,
            "standard_deviation": std_dev,
            "cv_percent": cv,
            "risk_class": self.classify(cv)
        }

    def classify(self, cv_percent: float) -> str:
        """Classify financial risk using the coefficient of variation."""

        if not np.isfinite(cv_percent) or cv_percent < 0:
            raise ValueError("CV must be finite and non-negative.")

        if cv_percent < self.low_threshold:
            return "Low"

        if cv_percent < self.high_threshold:
            return "Moderate"

        return "High"

    def value_at_risk(self, annual_revenues) -> dict:
        """Calculate the lower 5th percentile of annual revenue."""

        values = np.asarray(annual_revenues, dtype=float)

        if values.ndim != 1 or len(values) < 2:
            raise ValueError(
                "Provide at least two simulated annual revenues."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError("Annual revenues must be finite.")

        if np.any(values < 0):
            raise ValueError("Annual revenues cannot be negative.")

        threshold = float(np.percentile(values, 5))

        return {
            "revenue_5th_percentile": threshold,
            "mean_annual_revenue": float(np.mean(values)),
            "shortfall_from_mean": float(np.mean(values) - threshold)
        }

    def __repr__(self) -> str:
        """Return a readable risk assessor description."""

        return (
            f"RiskAssessor(low={self.low_threshold}%, "
            f"high={self.high_threshold}%)"
        )
