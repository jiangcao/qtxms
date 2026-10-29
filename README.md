# Mode space as the coarse level of a two-grid transport solver

Prototype on an effective-mass rough Si nanosheet (5 × 3 nm, a = 0.25 nm, m* = 0.26, 3 eV oxide barrier,
top/bottom roughness rms 0.3 nm, Λ = 1.5 nm, uniform ideal leads). Reference: exact RGF.

## Method (`mgms/em.py`)
- Prolongation P = U (block-diagonal per-slab basis), restriction U†, Galerkin A_c = U† A U, A = E − H − Σ_L − Σ_R.
- Coarse-grid projector π = U A_c⁻¹ U† A (checked idempotent to ~1e-14). Mode-space solution = π ψ_exact.
- Right-preconditioned GMRES on A ψ = b (b = Γ_L^{1/2} injection columns, QTBM), started from the mode-space solution.
- Balancing two-level preconditioner M⁻¹ = P_c + (I − P_c A) S (I − A P_c), P_c = U A_c⁻¹ U†.
- Smoother S: Chebyshev polynomial for (−A)⁻¹ on [E_b − E, λ_max], E_b = energy up to which the basis was built.
  Valid in effective mass because the complement is (nearly) definite; TB will need a two-interval smoother.
- Fine level uses the exact lead Σ; only the coarse solve sees U†ΣU.

Bases (`mgms/em.py`):
- CMS M: lowest M eigenvectors of each slab H_ii (conventional coupled mode space).
- LDM E_b: local density-matrix basis. For slab i, diagonalize slabs i−3..i+3 (zero-flux ends), keep all states below
  e1 + E_b, form ρ_i restricted to slab i, keep eigenvectors with weight > 1e-3 · max.

## Results (`python -m mgms.run 20`, 80 slabs × 320 = 25 600 points, 6 energies 10–160 meV above the first subband, tol 1e-8)

| basis | N/slab | mode-space max T error | after GMRES | GMRES its (mean) | fine matvecs / RHS |
|---|---|---|---|---|---|
| none (smoother + GMRES only) | 0 | — | 7e-8 | 41.7 | 459 |
| CMS M=2 | 2 | 85% | 4e-8 | 15.8 | 175 |
| CMS M=16 | 16 | 32% | 1e-10 | 14.8 | 164 |
| LDM E_b=0.6 eV | 5.3 | 3.3% | 1e-10 | 8.9 | 99 |
| LDM E_b=1.0 eV | 10.0 | 1.2% | 2e-11 | 6.8 | 76 |

(Chebyshev degree 8. Degree 16 halves GMRES iterations but gives about the same matvec count.)

Device-length scaling (`python -m mgms.scaling`, Chebyshev degree 8, 3 energies):

| L | n | no coarse: its / matvecs | LDM E_b=1: N, its / matvecs |
|---|---|---|---|
| 10 nm | 12 800 | 32.5 / 342 | 383, 6.8 / 74 |
| 20 nm | 25 600 | 46.0 / 485 | 799, 7.0 / 78 |
| 40 nm | 51 200 | 76.2 / 789 | 1648, 7.0 / 78 |

## Takeaways
1. The coarse space makes the iteration count independent of device length (7 iterations from 10 to 40 nm);
   without it, iterations grow with the number of window states.
2. The basis no longer needs to be accurate on its own: a 1.2%-error basis converges to 1e-11 in 7 iterations.
3. What matters is how high in energy the basis covers (E_b), because that sets the smoother's condition number.
   CMS iteration counts barely improve with M because the smoother bound was kept at 0.3 eV for CMS.
4. Cost per energy (20 nm): ~78 sparse matvecs (nnz ≈ 3.8e5 incl. dense Σ blocks) ≈ 3e7 complex mults,
   vs RGF ≈ 80 × ~6.6e7 ≈ 5e9 (rough estimate). Python wall time 0.15 s vs 1.6 s/energy.
   The LDM build (76 s, one-off, unoptimized cluster eigh) is the dominant cost here.

## Not yet tested
- Tight binding: indefinite complement (valence states below E) → two-interval Chebyshev or a few inner GMRES steps;
  spurious coarse bands (monitor σ_min(A_c)).
- Your two-level basis as U (drop-in: pass its block-diagonal U to `TwoLevel`).
- Bias / Poisson potential, many energies (basis reused, A_c rebuilt from precomputed U†HU).

Files (in `mgms/`): `device.py` (model, leads, RGF), `em.py` (bases, projector, solver), `run.py` (sweep), `scaling.py`.

---

# Ab initio tight-binding input (`abtb.py`, `twolevel.py`, `run_tb.py`)

```
python -m mgms.run_tb --xyz device.xyz --H H.mtx --S S.mtx --norb "Si:10,H:1" --units eV \
                 --dE 0.02,0.08,0.15 --Eb 0.6 --basis ldm,slab
```

Input
- `--xyz`: standard XYZ (count, comment, `species x y z`). Atom order must match the matrix orbital order.
- `--H`, `--S`: `.npz` (scipy `save_npz`), `.mtx` (Matrix Market), or text triplets `i j Re [Im]`
  (0/1-based auto-detected, `#`/`%` comments). Upper-triangle-only files are completed to Hermitian.
  `--S` omitted = orthogonal TB. `--units eV|Ry|Ha` scales H.
- `--norb`: one integer, per species (`Si:10,H:1`), or a file with one integer per atom.
- Assumes a device-only matrix (no k-points), orbitals of an atom consecutive, and device ends that repeat
  the lead cell (checked and reported).

What it does
1. Sorts atoms along `--axis`, bins them into slabs of width = coupling range along transport (from the
   nonzero pattern of H and S), with bin edges between atomic planes; asserts block tridiagonality.
   Use `--slab-width` = lead unit-cell length (or a multiple) when the cell holds several planes.
2. Leads: Sancho–Rubio on A = (E+iη)S − H using the first/last two slabs; injection from Γ_L
   (channel count cross-checked against lead band crossings).
3. Lead bands → Ev, Ec (largest gap, or `--Ec`). Basis window = [mid-gap, Ec + Eb], so valence states are
   excluded from U and end up in the complement.
4. Coarse bases: `ldm` (local density matrix from generalized eigh of 2w+1-slab clusters) and `slab`
   (M lowest conduction eigenvectors of each (H_ii, S_ii), CMS analogue).
5. Smoother default = 10 GMRES steps (handles the indefinite complement: valence below E, conduction above);
   outer flexible GMRES. `--smoother cheb` only for definite cases. Reports σ_min(A_c) to flag spurious bands.
6. RGF reference with variable slab sizes (`--no-ref` to skip).

Tested on a synthetic non-orthogonal model (`python -m mgms.make_test_tb`: rough H-passivated wire, 2 orbitals/Si,
1 orbital/H, 1st+2nd-neighbour H, NN overlap, gap 3 eV, 5928 orbitals, shuffled atoms, upper-triangle text):

| E − Ec | channels | none: its / matvecs | ldm (N=221): MS err → final, its / matvecs | slab M=4 (N=240): MS err → final, its |
|---|---|---|---|---|
| 0.15 eV | 2 | 56 / 616 | 3.4e-3 → 5e-10, 8 / 105 | 2.5e-1 → 6e-10, 10 |
| 0.25 eV | 2 | 90 / 984 | 2.4e-4 → 6e-10, 9 / 118 | 2.2e-2 → 6e-10, 12 |

(At +0.02 and +0.08 eV T ≈ 0, tunnelling through constrictions.) `.npz`/`.mtx`/per-atom-file inputs give identical results.
Not yet tested on real ab initio files (e.g. Wannier/CP2K sp3d5s*-size orbital sets, d-orbitals, spin).
