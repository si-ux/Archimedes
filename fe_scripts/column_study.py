"""Parametric study: square concrete column, 2 m to 12 m high, 1500 kN axial + 20 kN lateral at mid-height.

For each height: slenderness, Euler load, utilisation P / P_cr, sway at mid-height and the first frequency.
Run in the console:  run('column_study.py')
"""
P = 1500.0  # kN
rows = []
for height in (2000, 6000, 12000):
    column(b=400, h=400, L=height, name=f"Column {height / 1000:g} m")
    material("concrete")
    fixed(at=0)
    pinned(at=1)
    point(P, at=1, dir="-z")
    point(20, at=0.5, dir="+x")
    solve()
    s = summary()
    rows.append((height, s["slenderness"], s["P_cr_kN"], s["column_class"], P / s["P_cr_kN"], at(0.5, "U1"), modes()[0]))

print(f"{'L (mm)':>7} {'KL/r':>6} {'P_cr (kN)':>10} {'class':>13} {'P/P_cr':>7} {'sway (mm)':>10} {'f1 (Hz)':>8}")
for r in rows:
    print(f"{r[0]:>7.0f} {r[1]:>6.1f} {r[2]:>10.0f} {r[3]:>13} {r[4]:>7.3f} {r[5]:>10.3f} {r[6]:>8.2f}")
