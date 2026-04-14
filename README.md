# captcha-toolkit

[![CI](https://github.com/EquationTec/captcha-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/EquationTec/captcha-toolkit/actions/workflows/ci.yml)

Local-first captcha solvers using classical computer vision. No ML, no GPU, no
paid API for core solvers. Just OpenCV and numpy.

## What's in here

| Solver               | Target                                      | Status  |
| -------------------- | ------------------------------------------- | ------- |
| `BrokenCircleSolver` | Find the one ring with a visible gap        | Ready   |
| `SilhouetteSolver`   | Image-grid "pick the odd one out"           | Planned |
| `ColorOutlierSolver` | Pick the color-saturation outlier in a grid | Planned |
| `SliderSolver`       | Puzzle-piece / slider captchas              | Planned |
| `RotateSolver`       | Rotate-to-upright image captcha             | Planned |
| `TwoCaptchaFallback` | Optional paid API wrapper                   | Planned |

All solvers share the `CaptchaSolver` interface and return a `SolverResult`
with click coordinates, a confidence score, and per-solver diagnostics.

## Install

```bash
pip install captcha-toolkit

# to run the tests:
pip install captcha-toolkit[dev]

# to use the (optional) 2Captcha fallback when added:
pip install captcha-toolkit[fallback]
```

## Quick start

```python
from captcha_toolkit import BrokenCircleSolver

with open("captcha.png", "rb") as f:
    image_bytes = f.read()

solver = BrokenCircleSolver()
result = solver.solve(image_bytes)

if result.found:
    x, y = result.coords
    print(f"click at ({x}, {y}), confidence={result.confidence:.2f}")
else:
    print(f"no broken circle found: {result.meta}")
```

For a captcha embedded in a larger screenshot, pass a bounding box:

```python
# bbox = (x, y, width, height) of the captcha region inside the full image
solver = BrokenCircleSolver(bbox=(430, 181, 420, 420))
result = solver.solve(page_screenshot_bytes)
```

If you can capture the captcha region before the overlay renders, pass it as
`reference` for a diff mask (usually boosts accuracy):

```python
result = solver.solve(image_bytes, reference=clean_backdrop_bytes)
```

## How BrokenCircleSolver works

The solver runs independent isolation masks, then looks for ring-shaped
features on each:

1. **Median-diff mask**: subtract a 7x7 median-blurred copy and threshold.
   Thin colored overlays survive; photograph content mostly does not.
2. **Saturation-anomaly mask**: subtract a Gaussian-blurred saturation
   channel and threshold. Catches rings when the photo median is too busy.
3. **OR of the first two**: catches rings either mask alone misses.
4. **Reference-diff mask**: if the caller supplied a clean backdrop, the
   raw pixel diff usually dominates.

Template matching at multiple radii locates ring candidates on each mask.
Candidates scored by arc coverage and largest contiguous gap (default
broken band: 25-160°), filtered by ring saturation, clustered across masks.
Cross-mask consensus drives the confidence score. A contour-based fallback
runs if template matching finds nothing.

## Known limitations

- On images with no broken circle, the solver still returns its best
  candidate; it doesn't decide "broken present vs none." Inspect
  `result.confidence` and `result.meta["mask_consensus"]` for a strict gate.
- Defaults are tuned to ~420x420px captchas with ring radius ~20-60px.
  For other sizes, pass `radii` and `bbox`.
- Heavy JPEG compression defeats the saturation mask; median-diff still
  usually works.
- Classical-CV only. Handles one captcha family, not a generic bypass.

## Benchmarks

Measured on a workstation (Ryzen 7, no GPU) with synthetic 420x420 images:

| Input                                | Mean solve time | Hit rate (tol=ring radius) |
| ------------------------------------ | --------------- | -------------------------- |
| 5 circles, 1 broken (seeds 0,1,2,3,7) | ~300-600 ms    | 5/5                        |

Real-world timing depends on source image format.

## Tests

```bash
pip install -e '.[dev]'
pytest -v
```

Synthetic fixtures only; no third-party captcha assets in repo. ~10s total.

## Roadmap

- [x] Broken-circle solver, synthetic tests, bbox support, optional reference
- [ ] Silhouette-hamming grid solver
- [ ] Color-saturation outlier solver
- [ ] Slider-puzzle solver
- [ ] Rotate-to-upright solver
- [ ] 2Captcha fallback wrapper with a unified interface
- [ ] Benchmark harness against public test images

## License

MIT. See [LICENSE](LICENSE).

## Use

Research, education, accessibility, and automating sites you're authorized
to automate. ToS and local law are your problem.
