"""Re-export the synth generator so existing tests keep their import path."""

from captcha_toolkit._synth import make_broken_circle_image

__all__ = ["make_broken_circle_image"]
