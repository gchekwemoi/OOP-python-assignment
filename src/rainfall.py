
"""Rainfall analysis for regions in Uganda."""

import numpy as np


class Region:
    """Store and analyse monthly rainfall data."""

    MONTHS = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]

    def __init__(self, name: str, rainfall):
        """Create a region with 12 monthly rainfall values."""

        self.name = name
        self.rainfall = np.asarray(rainfall, dtype=float)

        if self.rainfall.shape != (12,):
            raise ValueError(
                "Rainfall data must contain exactly 12 months."
            )

        if not np.all(np.isfinite(self.rainfall)):
            raise ValueError(
                "Rainfall values must be finite numbers."
            )

        if np.any(self.rainfall < 0):
            raise ValueError(
                "Rainfall cannot be negative."
            )

    def annual_total(self) -> float:
        """Calculate total annual rainfall in millimetres."""
        return float(np.sum(self.rainfall))

    def monthly_mean(self) -> float:
        """Calculate average monthly rainfall."""
        return float(np.mean(self.rainfall))

    def wettest_month(self) -> str:
        """Return the month with the highest rainfall."""
        return self.MONTHS[np.argmax(self.rainfall)]

    def driest_month(self) -> str:
        """Return the month with the lowest rainfall."""
        return self.MONTHS[np.argmin(self.rainfall)]

    def coefficient_of_variation(self) -> float:
        """Calculate monthly rainfall variability as a percentage."""

        mean = self.monthly_mean()

        if mean == 0:
            return 0.0

        return float(
            np.std(self.rainfall, ddof=1) / mean * 100
        )

    def __repr__(self) -> str:
        """Return a readable description of the region."""
        return f"Region(name={self.name!r}, annual_total={self.annual_total():.1f} mm)"
