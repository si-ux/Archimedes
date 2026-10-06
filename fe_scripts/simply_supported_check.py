"""Simply supported steel beam under a UDL: FE mid-span deflection vs beam theory.

Run in the console:  run('simply_supported_check.py')
"""
L, b, h, w = 6000.0, 200.0, 400.0, 10.0  # mm, mm, mm, kN/m (= N/mm)

beam(b=b, h=h, L=L, name="SS beam check")
material("steel")
pinned(at=0)
roller(at=1)
uniform(w, dir="-z")
solve()

E, G = 200000.0, 200000.0 / 2.6
I, A = b * h ** 3 / 12, b * h
bending = 5 * w * L ** 4 / (384 * E * I)
shear = w * L ** 2 / (8 * 5 / 6 * G * A)
fe = -at(0.5, "U3")
print(f"mid-span deflection  FE {fe:.3f} mm   theory {bending + shear:.3f} mm "
      f"(bending {bending:.3f} + shear {shear:.3f})   diff {100 * (fe / (bending + shear) - 1):+.1f}%")
print(f"peak von Mises {peak('von_mises')['value']:.1f} MPa   M/Z = {w * L ** 2 / 8 / (b * h ** 2 / 6):.1f} MPa")
show("U3")
