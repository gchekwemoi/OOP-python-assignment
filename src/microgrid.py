
"""Solar micro-grid energy allocation for a health centre in Kasese."""

import numpy as np
from scipy.linalg import solve


class MicroGrid:
    """Model solar and battery allocation using linear equations."""

    def __init__(self, coefficients=None):
        """Create and validate the coefficient matrix."""

        if coefficients is None:
            coefficients = [
                [3, 2],
                [4, 1]
            ]

        self.matrix = np.asarray(coefficients, dtype=float)

        if self.matrix.shape != (2, 2):
            raise ValueError(
                "The coefficient matrix must be 2 by 2."
            )

        if not np.all(np.isfinite(self.matrix)):
            raise ValueError(
                "Matrix values must be finite numbers."
            )

        self.determinant = float(np.linalg.det(self.matrix))
        self.condition_number = float(np.linalg.cond(self.matrix))

        if np.isclose(self.determinant, 0):
            raise ValueError(
                "The system is singular and cannot be solved uniquely."
            )

    def solve_day(self, d1: float, d2: float) -> dict:
        """Calculate solar and battery allocations for one day."""

        try:
            demand = np.array([d1, d2], dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Demand values must be numeric."
            ) from error

        if not np.all(np.isfinite(demand)):
            raise ValueError(
                "Demand values must be finite."
            )

        if np.any(demand < 0):
            raise ValueError(
                "Demand values cannot be negative."
            )

        solution = solve(self.matrix, demand)

        solar = float(solution[0])
        battery = float(solution[1])

        feasible = (
            solar >= -1e-9 and battery >= -1e-9
        )

        return {
            "solar_kwh": max(0.0, solar) if feasible else solar,
            "battery_kwh": max(0.0, battery) if feasible else battery,
            "feasible": feasible
        }

    def __repr__(self) -> str:
        """Return a readable description of the micro-grid."""

        return (
            f"MicroGrid(determinant={self.determinant:.2f}, "
            f"condition_number={self.condition_number:.2f})"
        )
