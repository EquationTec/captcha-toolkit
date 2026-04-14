"""Broken-circle captcha solver.

Problem: the captcha shows several colored circles on a photographic
background. All but one are complete; click the one with a visible gap.

Approach:
    1. Isolate the colored overlay from photo background via median-diff and
       saturation-anomaly masks.
    2. Ring template matching at multiple radii on each mask.
    3. Score candidates by arc coverage and largest gap (degrees).
    4. Filter to the "broken" gap band (default 25-160°) and require ring-band
       saturation above a floor so photo edges don't pass.
    5. Cluster candidates across masks; the top cluster is the answer.

A contour-based fallback runs if template matching finds nothing.

Defaults tuned for ~420x420px captchas. For other sizes, pass ``radii`` and
``bbox`` explicitly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

from captcha_toolkit.base import CaptchaSolver, SolverResult

# Hard cap on decoded image dimensions. Prevents a malicious caller from
# passing a tiny PNG with an enormous IHDR header that balloons to GBs on
# decode. 4096x4096 is well above any real captcha.
MAX_IMAGE_DIM = 4096
# Per-radius detection cap. Adversarial near-uniform masks can produce
# millions of peaks above the threshold; bound the work before NMS.
MAX_DETECTIONS_PER_RADIUS = 2000


class BrokenCircleSolver(CaptchaSolver):
    """Find the one broken circle in a grid of colored-ring overlays.

    Args:
        bbox: Optional crop region as ``(x, y, w, h)``. If None, the whole image
            is scanned. Cropping speeds up scanning when the captcha is a known
            region of a page screenshot.
        radii: Iterable of candidate ring radii (in pixels) for template matching.
            Default: ``range(20, 65, 3)`` works for ~420px captchas.
        gap_range: ``(min_deg, max_deg)`` window that counts as "broken". Default
            ``(25, 160)``.
        min_saturation: Minimum mean HSV-S along the ring. Filters out photo edges
            that coincidentally match ring templates. Default 30.
    """

    def __init__(
        self,
        bbox: tuple[int, int, int, int] | None = None,
        radii: range | list[int] | None = None,
        gap_range: tuple[float, float] = (25.0, 160.0),
        min_saturation: float = 30.0,
    ) -> None:
        if cv2 is None:
            raise RuntimeError(
                "opencv-python is required for BrokenCircleSolver. "
                "Install with `pip install opencv-python`."
            )
        self.bbox = bbox
        self.radii = list(radii) if radii is not None else list(range(20, 65, 3))
        self.gap_range = gap_range
        self.min_saturation = min_saturation

    def solve(
        self,
        image: bytes,
        reference: bytes | None = None,
        **_kwargs: Any,
    ) -> SolverResult:
        """Solve one captcha.

        Args:
            image: PNG/JPEG bytes of the captcha image (or page screenshot if bbox set).
            reference: Optional reference background bytes. If supplied, a diff mask
                between reference and image is added to the isolation stack, which
                typically improves accuracy when a "clean" backdrop is available.

        Returns:
            SolverResult with coords in input-image pixel space.
        """
        page_img = _decode(image)
        if page_img is None:
            return SolverResult(coords=None, meta={"error": "decode_failed"})

        ph, pw = page_img.shape[:2]
        if ph > MAX_IMAGE_DIM or pw > MAX_IMAGE_DIM:
            return SolverResult(
                coords=None,
                meta={"error": "image_too_large", "shape": (ph, pw), "cap": MAX_IMAGE_DIM},
            )

        if self.bbox:
            bx, by, bw, bh = self.bbox
            cx = max(0, int(bx))
            cy = max(0, int(by))
            x_end = min(pw, cx + int(bw))
            y_end = min(ph, cy + int(bh))
            crop = page_img[cy:y_end, cx:x_end]
        else:
            cx, cy = 0, 0
            crop = page_img

        if crop.size == 0:
            return SolverResult(coords=None, meta={"error": "empty_crop"})
        # Re-read crop shape so a clipped bbox uses the actual dimensions,
        # not the user-requested ones.
        ch, cw = crop.shape[:2]

        masks = _build_masks(crop, reference, cw, ch)
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gap_lo, gap_hi = self.gap_range
        all_candidates: list[tuple[int, int, int, float, float, float, str]] = []

        for name, mask in masks:
            rings = _find_rings(mask, self.radii)
            if not rings:
                continue
            scored = []
            for rx, ry, rr, score in rings:
                coverage, gap = _arc_stats(mask, rx, ry, rr)
                sat = _ring_saturation(hsv_crop, rx, ry, rr)
                scored.append((rx, ry, rr, coverage, gap, score, sat))

            broken = [
                s
                for s in scored
                if gap_lo < s[4] < gap_hi and s[3] > 180 and s[6] > self.min_saturation
            ]
            for s in broken:
                all_candidates.append((s[0], s[1], s[2], s[3], s[4], s[6], name))

        if all_candidates:
            clusters = _cluster_candidates(all_candidates, dist=25)
            # Rank by mask consensus, then by average saturation.
            clusters.sort(
                key=lambda c: (len(c), float(np.mean([s[5] for s in c]))),
                reverse=True,
            )
            best = clusters[0]
            bx = int(np.mean([s[0] for s in best]))
            by = int(np.mean([s[1] for s in best]))
            avg_sat = float(np.mean([s[5] for s in best]))
            mask_names = sorted({s[6] for s in best})
            # `combined` is derived (OR of med20 + sat), not independent; exclude it
            # from both numerator and denominator of the consensus ratio.
            independent_masks = [m for m in masks if m[0] != "combined"]
            independent_hits = sum(1 for s in best if s[6] != "combined")
            confidence = min(
                1.0, independent_hits / max(1, len(independent_masks))
            ) * min(1.0, avg_sat / 100.0)
            return SolverResult(
                coords=(bx + cx, by + cy),
                confidence=confidence,
                meta={
                    "method": "ring_template",
                    "mask_consensus": len(best),
                    "mask_names": mask_names,
                    "avg_saturation": avg_sat,
                },
            )

        # Fallback: contour-based arc detection.
        for name, mask in masks:
            hit = _contour_arcs(mask)
            if hit:
                rx, ry, _rr = hit
                return SolverResult(
                    coords=(rx + cx, ry + cy),
                    confidence=0.3,
                    meta={"method": "contour_fallback", "mask": name},
                )

        return SolverResult(coords=None, meta={"method": "none"})


# --- internal helpers ---------------------------------------------------------


def _decode(png_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(png_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _build_masks(
    crop: np.ndarray, reference_png: bytes | None, cw: int, ch: int
) -> list[tuple[str, np.ndarray]]:
    """Build isolation masks using complementary techniques so at least one
    survives photograph content that defeats the others.
    """
    masks: list[tuple[str, np.ndarray]] = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Median-diff: overlay thin features pop out against a heavily smoothed base.
    median = cv2.medianBlur(crop, 7)
    diff = np.max(cv2.absdiff(crop, median), axis=2)
    for t in (20, 14):
        _, m = cv2.threshold(diff, t, 255, cv2.THRESH_BINARY)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        masks.append((f"med{t}", m))

    # Local saturation anomaly: colored overlays stand out vs. blurred saturation.
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    s_bg = cv2.GaussianBlur(s, (31, 31), 10)
    s_diff = np.clip(s - s_bg, 0, 255).astype(np.uint8)
    _, sm = cv2.threshold(s_diff, 12, 255, cv2.THRESH_BINARY)
    sm = cv2.morphologyEx(sm, cv2.MORPH_CLOSE, kernel)
    masks.append(("sat", sm))

    # Combined OR of first two — catches features either method alone misses.
    combined = cv2.bitwise_or(masks[0][1], sm)
    masks.append(("combined", combined))

    # Reference diff (if caller supplied a clean background for this captcha size).
    if reference_png:
        ref = _decode(reference_png)
        if ref is not None:
            # Guard against a resize buffer > MAX_IMAGE_DIM^2 when the crop
            # dimensions exceed the cap (refdiff requires resizing the ref to
            # match). Skip the refdiff mask in that case rather than allocate.
            if cw * ch > MAX_IMAGE_DIM * MAX_IMAGE_DIM:
                return masks
            if ref.shape[:2] != (ch, cw):
                ref = cv2.resize(ref, (cw, ch))
            ed = np.max(cv2.absdiff(crop, ref), axis=2)
            _, em = cv2.threshold(ed, 15, 255, cv2.THRESH_BINARY)
            em = cv2.morphologyEx(em, cv2.MORPH_CLOSE, kernel)
            # Insert first: usually the most robust mask when available.
            masks.insert(0, ("refdiff", em))

    return masks


def _find_rings(
    mask: np.ndarray, radii: list[int]
) -> list[tuple[int, int, int, float]]:
    """Template-match ring patterns at each radius, return NMS-filtered detections."""
    detections: list[tuple[int, int, int, float]] = []
    h, w = mask.shape[:2]

    # Skip degenerate masks: TM_CCOEFF_NORMED divides by variance, so
    # uniform masks produce NaN warnings and meaningless results.
    if mask.max() == 0:
        return []

    for r in radii:
        # Odd side length so the template has an integer center and no
        # half-pixel recovery bias.
        sz = 2 * r + 13
        if sz >= h or sz >= w:
            continue
        tmpl = np.zeros((sz, sz), dtype=np.uint8)
        c = sz // 2
        cv2.circle(tmpl, (c, c), r, 255, 2)

        with np.errstate(invalid="ignore", divide="ignore"):
            result = cv2.matchTemplate(mask, tmpl, cv2.TM_CCOEFF_NORMED)

        threshold = 0.25
        flat = result.ravel()
        # Cap work per radius against adversarial near-uniform masks.
        if flat.size > MAX_DETECTIONS_PER_RADIUS:
            # argpartition picks the top-K indices without sorting everything.
            top_idx = np.argpartition(flat, -MAX_DETECTIONS_PER_RADIUS)[
                -MAX_DETECTIONS_PER_RADIUS:
            ]
            idx = top_idx[flat[top_idx] >= threshold]
        else:
            idx = np.where(flat >= threshold)[0]

        if idx.size == 0:
            continue
        ys, xs = np.unravel_index(idx, result.shape)
        for py, px in zip(ys, xs):
            detections.append(
                (int(px + c), int(py + c), r, float(result[py, px]))
            )

    return _nms_rings(detections, dist=20)


def _nms_rings(
    detections: list[tuple[int, int, int, float]], dist: int = 20
) -> list[tuple[int, int, int, float]]:
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d[3], reverse=True)
    kept: list[tuple[int, int, int, float]] = []
    for d in detections:
        if any(
            (d[0] - k[0]) ** 2 + (d[1] - k[1]) ** 2 < dist * dist for k in kept
        ):
            continue
        kept.append(d)
    return kept


def _ring_saturation(
    hsv_crop: np.ndarray, cx: int, cy: int, radius: int, n: int = 72
) -> float:
    """Mean HSV-S along the ring (with 3-pixel radial tolerance)."""
    h, w = hsv_crop.shape[:2]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    sat_channel = hsv_crop[:, :, 1]

    collected: list[np.ndarray] = []
    for dr in (-1, 0, 1):
        r = radius + dr
        px = (cx + r * cos_a).astype(np.int64)
        py = (cy + r * sin_a).astype(np.int64)
        valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        if valid.any():
            collected.append(sat_channel[py[valid], px[valid]].astype(np.float64))
    if not collected:
        return 0.0
    vals = np.concatenate(collected)
    return float(vals.mean()) if vals.size else 0.0


def _arc_stats(
    mask: np.ndarray, cx: int, cy: int, radius: int, n: int = 360
) -> tuple[float, float]:
    """Return (coverage_deg, largest_gap_deg) for the ring at (cx, cy, radius).

    Uses a stricter ±1 radial band for coverage counting (so noisy backgrounds
    don't inflate coverage) and the looser ±2 band for gap detection (so a
    genuine ring isn't broken by 1-pixel jitter).
    """
    h, w = mask.shape[:2]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    has_edge_wide = np.zeros(n, dtype=bool)
    has_edge_tight = np.zeros(n, dtype=bool)

    for dr in (-2, -1, 0, 1, 2):
        r = radius + dr
        px = (cx + r * cos_a).astype(np.int64)
        py = (cy + r * sin_a).astype(np.int64)
        valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        if not valid.any():
            continue
        hit = np.zeros(n, dtype=bool)
        hit[valid] = mask[py[valid], px[valid]] > 0
        has_edge_wide |= hit
        if -1 <= dr <= 1:
            has_edge_tight |= hit

    coverage_deg = (np.sum(has_edge_tight) / n) * 360

    if np.all(has_edge_wide):
        return coverage_deg, 0.0
    if not np.any(has_edge_wide):
        return coverage_deg, 360.0

    # Largest contiguous False run, wrapping via doubling.
    not_edge = ~has_edge_wide
    doubled = np.concatenate([not_edge, not_edge])
    # Vectorized longest-run: find run boundaries then take the max length.
    # Pad with False so run-ends are always detected.
    padded = np.concatenate([[False], doubled, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if starts.size == 0:
        max_gap = 0
    else:
        max_gap = int((ends - starts).max())
    gap_deg = (min(max_gap, n) / n) * 360
    return coverage_deg, gap_deg


def _cluster_candidates(
    cands: list[tuple[int, int, int, float, float, float, str]], dist: int = 25
) -> list[list[tuple[int, int, int, float, float, float, str]]]:
    """Group candidate rings within `dist` pixels so multi-mask agreement stacks up.

    Compares each candidate against the running centroid of each cluster so
    membership is order-independent.
    """
    clusters: list[list[tuple[int, int, int, float, float, float, str]]] = []
    centroids: list[tuple[float, float]] = []
    for cand in cands:
        for i, (cxm, cym) in enumerate(centroids):
            dx = cand[0] - cxm
            dy = cand[1] - cym
            if dx * dx + dy * dy < dist * dist:
                clusters[i].append(cand)
                n = len(clusters[i])
                # Update running centroid mean.
                centroids[i] = (
                    cxm + (cand[0] - cxm) / n,
                    cym + (cand[1] - cym) / n,
                )
                break
        else:
            clusters.append([cand])
            centroids.append((float(cand[0]), float(cand[1])))
    return clusters


def _contour_arcs(mask: np.ndarray) -> tuple[int, int, int] | None:
    """Fallback: find broken arcs directly from mask contours."""
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    candidates: list[tuple[int, int, int, float, float]] = []
    for cnt in contours:
        if len(cnt) < 25:
            continue
        (ex, ey), er = cv2.minEnclosingCircle(cnt)
        if er < 18 or er > 70:
            continue
        pts = cnt[:, 0, :]
        dists = np.sqrt((pts[:, 0] - ex) ** 2 + (pts[:, 1] - ey) ** 2)
        if np.std(dists) > er * 0.18:
            continue
        if abs(np.mean(dists) - er) > er * 0.25:
            continue
        angles = np.arctan2(pts[:, 1] - ey, pts[:, 0] - ex)
        cov, gap = _angular_stats_contour(angles)
        if 200 < cov < 335:
            candidates.append((int(ex), int(ey), int(er), cov, gap))

    if not candidates:
        return None

    best = max(candidates, key=lambda c: c[3])
    return (best[0], best[1], best[2])


def _angular_stats_contour(angles: np.ndarray) -> tuple[float, float]:
    if len(angles) < 2:
        return 0.0, 360.0
    sa = np.sort(angles.flatten())
    diffs = np.diff(sa)
    wrap = (2 * np.pi - sa[-1]) + sa[0]
    gaps = np.append(diffs, wrap)
    gap_deg = float(np.degrees(np.max(gaps)))
    return 360.0 - gap_deg, gap_deg
