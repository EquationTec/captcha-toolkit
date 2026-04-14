"""Synthetic broken-circle captcha generator.

Used by tests and examples. Ships in the wheel so the `circle_demo` example
works after ``pip install captcha-toolkit``.
"""

from __future__ import annotations

import cv2
import numpy as np


def make_broken_circle_image(
    size: int = 420,
    ring_radius: int = 40,
    ring_thickness: int = 3,
    broken_gap_deg: float = 60.0,
    circles: int = 5,
    seed: int = 0,
) -> tuple[bytes, tuple[int, int]]:
    """Render a synthetic broken-circle captcha.

    Returns ``(png_bytes, (broken_x, broken_y))``: the target the solver
    should find.
    """
    rng = np.random.default_rng(seed)

    base = rng.integers(40, 180, size=(size, size, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (15, 15), 0)

    margin = ring_radius + 20
    positions: list[tuple[int, int]] = []
    attempts = 0
    while len(positions) < circles and attempts < 1000:
        x = int(rng.integers(margin, size - margin))
        y = int(rng.integers(margin, size - margin))
        if all((x - px) ** 2 + (y - py) ** 2 > (2.3 * ring_radius) ** 2 for px, py in positions):
            positions.append((x, y))
        attempts += 1

    if len(positions) < circles:
        raise RuntimeError("could not place all circles; reduce circles or grow size")

    broken_idx = int(rng.integers(0, circles))

    colors = [
        (
            int(rng.integers(120, 256)),
            int(rng.integers(0, 120)),
            int(rng.integers(0, 120)),
        )
        for _ in range(circles)
    ]
    # Rotate channels so each ring uses a different dominant color.
    for i, _ in enumerate(colors):
        shift = 1 + (i % 3)
        c = colors[i]
        colors[i] = (c[(0 + shift) % 3], c[(1 + shift) % 3], c[(2 + shift) % 3])

    for i, ((x, y), color) in enumerate(zip(positions, colors)):
        if i == broken_idx:
            start = float(rng.uniform(0, 360 - broken_gap_deg))
            end = start + (360 - broken_gap_deg)
            cv2.ellipse(
                base,
                (x, y),
                (ring_radius, ring_radius),
                0,
                start,
                end,
                color,
                ring_thickness,
            )
        else:
            cv2.circle(base, (x, y), ring_radius, color, ring_thickness)

    ok, buf = cv2.imencode(".png", base)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return bytes(buf), positions[broken_idx]
