"""Synthetic tests for BrokenCircleSolver.

These render known broken-circle captchas in-memory and verify the solver
locates the target. They don't prove real-world robustness, but they catch
regressions in the core CV pipeline.
"""

from __future__ import annotations

import math

import pytest

from captcha_toolkit import BrokenCircleSolver
from tests.fixtures import make_broken_circle_image


def _within(a: tuple[int, int], b: tuple[int, int], tol: int) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= tol


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7])
def test_solver_finds_broken_circle(seed: int) -> None:
    # Tolerance is set to the ring radius so any predicted click inside the ring
    # counts as a solve — that's what passes a real-world click captcha.
    ring_radius = 40
    png, target = make_broken_circle_image(ring_radius=ring_radius, seed=seed)
    solver = BrokenCircleSolver()
    result = solver.solve(png)
    assert result.found, f"seed={seed}: expected a hit, got None ({result.meta})"
    assert _within(result.coords, target, tol=ring_radius), (
        f"seed={seed}: predicted {result.coords}, expected {target}"
    )


def test_solver_reports_confidence() -> None:
    png, _ = make_broken_circle_image(seed=0)
    result = BrokenCircleSolver().solve(png)
    assert 0.0 <= result.confidence <= 1.0


def test_solver_returns_none_on_empty_image() -> None:
    # Plain noisy background with no rings at all — solver must not hallucinate.
    import cv2
    import numpy as np

    rng = np.random.default_rng(0)
    img = rng.integers(40, 180, size=(420, 420, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (15, 15), 0)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    result = BrokenCircleSolver().solve(bytes(buf))
    assert not result.found
    assert result.meta is not None


def test_bbox_restriction() -> None:
    ring_radius = 40
    png, target = make_broken_circle_image(size=420, ring_radius=ring_radius, seed=0)
    x0 = max(0, target[0] - 80)
    y0 = max(0, target[1] - 80)
    bbox = (x0, y0, 160, 160)
    result = BrokenCircleSolver(bbox=bbox).solve(png)
    assert result.found
    assert _within(result.coords, target, tol=ring_radius)
