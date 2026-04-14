"""Common solver interface and result types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class SolverResult:
    """Result of a solver attempt.

    coords: (x, y) pixel location to click, in the coordinate system of the input image.
            None if the solver did not find a target.
    confidence: 0.0-1.0 self-reported score. Not calibrated across solvers; use as a
                relative signal within one solver type.
    meta: free-form per-solver diagnostics (mask names, candidate counts, timings).
    """

    coords: tuple[int, int] | None
    confidence: float = 0.0
    meta: dict[str, Any] | None = None

    @property
    def found(self) -> bool:
        return self.coords is not None


class CaptchaSolver(ABC):
    """Abstract base for a captcha solver.

    Subclasses take image bytes (PNG/JPEG) and return a SolverResult.
    Stateless; configuration is passed at construction.
    """

    @abstractmethod
    def solve(self, image: bytes, **kwargs: Any) -> SolverResult:
        """Solve one captcha. See SolverResult for return shape."""
        raise NotImplementedError
