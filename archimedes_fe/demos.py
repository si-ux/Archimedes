"""The four demo members."""
from __future__ import annotations

from .model import CONCRETE, STEEL, Load, Member, Model, Section, Support


def continuous_beam() -> Model:
    """Three equal 4 m spans, pinned + rollers, UDL plus a point load in the middle span."""
    return Model(
        "Continuous beam (3 × 4 m)",
        Member(Section("rect", 300, 500), 12000, "horizontal"),
        CONCRETE,
        [Support("pinned", 0.0), Support("roller", 1 / 3), Support("roller", 2 / 3), Support("roller", 1.0)],
        [Load("uniform", "z", -1, 0.0, 20.0, 1.0, 20.0), Load("point", "z", -1, 0.5, 60.0)],
    )


def cantilever() -> Model:
    """3 m steel cantilever: tip point load plus a triangular (trapezoidal 0 → 15 kN/m) load."""
    return Model(
        "Cantilever (3 m)",
        Member(Section("rect", 150, 300), 3000, "horizontal"),
        STEEL,
        [Support("fixed", 0.0)],
        [Load("point", "z", -1, 1.0, 30.0), Load("trapezoidal", "z", -1, 0.0, 0.0, 1.0, 15.0)],
    )


def long_column() -> Model:
    """Slender pinned-pinned steel column: Euler buckling governs."""
    return Model(
        "Long column (Ø150, 6 m)",
        Member(Section("circle", d=150), 6000, "vertical"),
        STEEL,
        [Support("pinned", 0.0), Support("pinned", 1.0)],
        [Load("point", "z", -1, 1.0, 200.0), Load("point", "x", 1, 0.5, 2.0)],
    )


def short_column() -> Model:
    """Stocky fixed-pinned concrete column: crushing governs."""
    return Model(
        "Short column (400 × 400, 2 m)",
        Member(Section("rect", 400, 400), 2000, "vertical"),
        CONCRETE,
        [Support("fixed", 0.0), Support("pinned", 1.0)],
        [Load("point", "z", -1, 1.0, 1500.0), Load("point", "x", 1, 0.5, 20.0)],
    )


DEMOS = {
    "continuous": continuous_beam,
    "cantilever": cantilever,
    "long_column": long_column,
    "short_column": short_column,
}
