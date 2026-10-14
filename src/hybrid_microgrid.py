
"""Hybrid solar, battery and diesel micro-grid model."""

import numpy as np
from scipy.linalg import solve

from src.microgrid import MicroGrid


class HybridMicroGrid(MicroGrid):
    """Extend MicroGrid with a diesel generator."""

    def __init__(self):
        super().__init__()

        # Illustrative third constraint including diesel (z)
        self.hybrid_matrix = np.array([
            [3, 2, 1],
            [4, 1, 2],
            [1, 1, 1]
        ], dtype=float)

    def solve_hybrid(
        self, d1: float, d2: float, d3: float
    ) -> dict:
        """Calculate solar, battery and diesel allocations."""

        try:
            demand = np.array([d1, d2, d3], dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("Demand must be numeric.") from error

        if not np.all(np.isfinite(demand)):
            raise ValueError("Demand must be finite.")

        if np.any(demand < 0):
            raise ValueError("Demand cannot be negative.")

        determinant = np.linalg.det(self.hybrid_matrix)

        if np.isclose(determinant, 0):
            raise ValueError(
                "The hybrid equations are linearly dependent."
            )

        solar, battery, diesel = solve(
            self.hybrid_matrix, demand
        )

        return {
            "solar_kwh": float(solar),
            "battery_kwh": float(battery),
            "diesel_kwh": float(diesel),
            "feasible": bool(
                np.all(np.array([solar, battery, diesel]) >= -1e-9)
            )
        }
