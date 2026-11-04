"""The energy filter of the density-matrix contour rule: weight f(E) that a state at energy E gets in rho.
Closed state (pole on the axis): f from the 24-point Gauss-Legendre arc rule (window Ec - 0.15 ... + 0.6 eV).
Broadened state (pole at E - i Gamma/2): exact fraction of its Lorentzian inside the window."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
mu1, mu2, n = -0.15, 0.6, 24                                  # E - Ec [eV]
c, r = 0.5 * (mu1 + mu2), 0.5 * (mu2 - mu1)
x, w = np.polynomial.legendre.leggauss(n)
th, wt = 0.5 * np.pi * (x + 1), 0.5 * np.pi * w
z = c + r * np.exp(1j * th); dz = -1j * r * np.exp(1j * th) * wt
E = np.linspace(-1.2, 1.6, 6001)
f = -np.imag((dz[None] / (z[None] - E[:, None])).sum(1)) / np.pi
feff = lambda G: (np.arctan((mu2 - E) / (G / 2)) - np.arctan((mu1 - E) / (G / 2))) / np.pi
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 7.6), sharex=True, height_ratios=[1, 1.5])
for a in (a1, a2):
    a.set_facecolor(SURF); a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.axvspan(mu1, mu2, color="#2a78d6", alpha=0.07, lw=0)
a1.plot(E, f, color=INK, lw=2, label="closed state, 24-point contour rule")
a1.plot(E, feff(0.1), color="#eb6834", lw=1.5, label="broadened state, Γ = 100 meV")
a1.set_ylabel("weight f(E) in ρ"); a1.set_ylim(-0.05, 1.1)
a1.text(c, 1.03, "window W", ha="center", fontsize=9, color="#2a78d6")
a1.legend(frameon=False, fontsize=8.5, loc="center right")
a1.set_title("Energy filter of the density-matrix contour (window Ec − 0.15 … + 0.6 eV)", loc="left", fontsize=10.5, color=INK)
a2.semilogy(E, np.abs(f) + 1e-18, color=INK, lw=2, label="closed state, 24-point contour rule")
for G, col in ((1e-3, "#1baf7a"), (1e-2, "#eda100"), (1e-1, "#eb6834")):
    a2.semilogy(E, feff(G), color=col, lw=1.5, label=f"broadened state, Γ = {1e3 * G:g} meV")
a2.set_ylim(1e-16, 3); a2.set_ylabel("|f(E)|"); a2.set_xlabel("energy of the state, E − Ec (eV)")
a2.legend(frameon=False, fontsize=8.5, loc="lower left")
fig.tight_layout()
fn = os.path.join(OUT, "fig11_contour_filter.png")
fig.savefig(fn, dpi=150, facecolor=SURF)
print("->", fn)
