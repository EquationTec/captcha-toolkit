"""Minimal demo: render a synthetic broken-circle captcha and solve it.

Run:
    python -m examples.circle_demo

This does not hit any third-party service; the image is generated locally.
"""

from __future__ import annotations

import sys
import time

from captcha_toolkit import BrokenCircleSolver
from captcha_toolkit._synth import make_broken_circle_image


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    png, target = make_broken_circle_image(seed=seed)

    solver = BrokenCircleSolver()
    t0 = time.perf_counter()
    result = solver.solve(png)
    dt_ms = (time.perf_counter() - t0) * 1000

    print(f"target:     {target}")
    print(f"predicted:  {result.coords}")
    print(f"confidence: {result.confidence:.3f}")
    print(f"meta:       {result.meta}")
    print(f"solve_time: {dt_ms:.1f} ms")

    if result.coords is None:
        return 1
    err = ((result.coords[0] - target[0]) ** 2 + (result.coords[1] - target[1]) ** 2) ** 0.5
    print(f"error_px:   {err:.1f}")
    return 0 if err < 20 else 2


if __name__ == "__main__":
    sys.exit(main())
