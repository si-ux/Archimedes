"""Your own helpers, imported in the console:  from my_tools import self_weight

Modules reach the console commands through `api` (the console you are typing in).
"""
from archimedes_fe.scripting import api


def self_weight():
    """Add the member's self weight as a uniform load (beams: -z; columns: a point load at the base is not needed)."""
    m = api.model
    w = m.summary()["self_weight_kN_m"]
    if m.member.orientation == "vertical":
        print("columns: self weight acts along the member; added as an axial point load at the top")
        return api.point(round(w * m.member.length / 1000, 3), at=1, dir="-z")
    return api.uniform(round(w, 3), dir="-z")


def span_loads(w, spans):
    """The same UDL on each of `spans` equal spans, as separate loads (handy for pattern loading)."""
    for i in range(spans):
        api.uniform(w, start=i / spans, end=(i + 1) / spans, dir="-z")
