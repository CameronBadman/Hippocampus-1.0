from __future__ import annotations


def rcparams() -> dict[str, object]:
    """Small dependency-free plot style shared by research artifacts."""

    return {
        "background": (250, 250, 248),
        "foreground": (37, 42, 52),
        "grid": (215, 218, 222),
        "kept": (37, 99, 235),
        "diagnostic": (124, 58, 237),
        "guard_violation": (220, 38, 38),
        "best": (5, 150, 105),
        "width": 900,
        "height": 520,
        "margin": 64,
    }
