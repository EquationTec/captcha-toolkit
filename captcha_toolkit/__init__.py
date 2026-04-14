"""captcha-toolkit: local-first captcha solvers using classical computer vision."""

from captcha_toolkit.base import CaptchaSolver, SolverResult
from captcha_toolkit.circle import BrokenCircleSolver

__version__ = "0.1.0"

__all__ = [
    "CaptchaSolver",
    "SolverResult",
    "BrokenCircleSolver",
]
