# BUG: `ortho_lead` correction matrices are non-Hermitian at sampled k-points

- **Component:** `Gf_Module/Gf.py`, class `ortho_lead` (used by `read_overlap_data`)
- **Severity:** High — the device–electrode orthogonalization correction matrices
  `Sig0` / `Sig1` are non-Hermitian (and numerically wrong) for every k-point
  with a non-integer transverse component (see "Affected and unaffected cases").
- **Status:** Resolved (fixed in `ortho_lead.eval_HS_k`, 2026-10-23; verified)
- **Environment used for reproduction:** sisl 0.17.0rc2.dev42, numpy 2.5.2
  (2026-10-16)

## Resolution

**Resolved 2026-10-23.** Root cause: the transverse Bloch phase grid in
`ortho_lead.eval_HS_k` was built from `Nd1, Nd2` in cell-offset order
(`[-1, 0, +1, ...]`) while the stored blocks are in array order
(`[0, +1, ..., +n//2, -n//2, ..., -1]`), so each block was multiplied by the
phase of the wrong cell — invisible at Gamma, non-Hermitian otherwise.

Fix (applied in `eval_HS_k`): roll the two index arrays into storage order
before building `exp_ph`:

```python
Nd1 = np.roll(Nd1, -(len(Nd1)//2))   # [-1,0,1] -> [0,+1,-1] (storage order)
Nd2 = np.roll(Nd2, -(len(Nd2)//2))
```

Verified after the fix: `herm_err(s00) = 0`, and `herm_err(Sinv_ll)` /
`herm_err(S⁻¹HS⁻¹)` at machine precision (`<= 7e-18`) at k = (0,0,0),
(0,0.13,0) and (0,0.5,0), for leads `+a1`, `-a1`, `+a2` and `-a2`; Gamma
results are unchanged. The full original analysis, the reproduction script
and the alternative `take`-based fix are kept below for reference.

## Description

When k-points are sampled (any k with a non-integer component transverse to
the lead direction), the correction matrices produced through `ortho_lead` —
`Sinv_ll = calc_Sinv(k)` and `X = converge_Sinv_Hl_Sinv(k)`, and therefore
`Sig0`/`Sig1` in `read_overlap_data` — are **not Hermitian**. At the Gamma
point everything is Hermitian to machine precision, which is why the bug is
invisible in Gamma-only calculations.

The Hermiticity-breaking defect is **not** in the inversion machinery
(`SE_1`, `block_td.Invert`, `calc_Sinv`, `calc_Sinv_Hl_Sinv` all preserve
Hermiticity of their input). It enters at the first k-dependent step,
`ortho_lead.eval_HS_k`, which applies the transverse Bloch phase factors to
the wrong real-space blocks. As a result `S00(k)` (and `H00/H01/S01(k)`) are
wrong and non-Hermitian for non-integer transverse k, and everything
downstream inherits it.

## Affected and unaffected cases

The bug is a misassignment of Bloch phase factors, so it is invisible whenever
all transverse phase factors `exp(2πi k·R)` equal 1. Summary:

| Case | Affected? | Why |
|---|---|---|
| No k-point sampling (single k = [0,0,0], Gamma only) | **No** | All phases are 1; the misassignment is a no-op. Verified numerically: herm_err(s00) = 0, herm_err(Sinv_ll) = 4.4e-16, herm_err(Sinv_Hl_Sinv) = 1.1e-14 at Gamma, and `s00` matches the correct reference sum exactly. |
| k-point sampling — Gamma point in the grid | **No** (at that k-point) | Same as above; only the non-Gamma points in the grid are corrupted. |
| k-point sampling — any non-integer transverse k-component | **Yes** | Phases are nontrivial and get assigned to the wrong blocks; `s00(k)` is non-Hermitian and the corrections are non-Hermitian and wrong. |
| Half-integer transverse k (k_perp = 1/2) | **Yes** | Phases are ±1 (real) but still misassigned. |
| Integer transverse k (k_perp ≡ 0 mod 1, e.g. k_y = 1.0) | **No** | All phases are 1; equivalent to Gamma. |
| Nonzero k-component along the lead direction | **Irrelevant** | The lead component is popped (`kv.pop(ax)`) before the phases are built. |
| Electrode with transverse nsc = 1 | **No** | No transverse periodicity → no transverse phases exist (`Nd = [0]`). |

The exact criterion: with A/B/C the cell 0/+1/−1 blocks along a transverse
axis and θ = 2π k_perp, the anti-Hermitian part of the code's `s00(k)` is

```
A(e^{−iθ} − e^{iθ}) + B(1 − e^{−iθ}) + C(e^{iθ} − 1)
```

which vanishes for all blocks only when θ ≡ 0 (mod 2π), i.e. integer k_perp.
At θ = π (k_perp = 1/2) it equals 2(B − C) ≠ 0 in general. So the safe
points are exactly the integer transverse k-points (Gamma and its
equivalents), not just Gamma itself.

Practical statement: a TBtrans calculation **without k-point sampling** runs
at the single k-point [0,0,0] and is **not affected** — the correction
matrices are correct and Hermitian. The bug only appears when k-points are
sampled and at least one sampled k-point has a non-integer component
transverse to the lead direction (which additionally requires transverse
periodicity, i.e. transverse nsc > 1). The device-side blocks `H_D_l` /
`S_D_l` come from sisl's own `Hk`/`Sk` and are correct in all cases.

Transparency note: the boundary above (integer vs non-integer k_perp,
including k_perp = 1/2) follows from the exact analytic expression for the
anti-Hermitian part, together with the two non-Gamma k-points verified
numerically in the reproduction below. A numerical k_y sweep
(0, 0.25, 0.5, 0.75, 1.0) was attempted but blocked by the environment's
approval mechanism; the Gamma row is verified numerically.

## Affected code

| Location | Role |
|---|---|
| `Gf.py:1015-1031` | `ortho_lead.eval_HS_k` — builds `H00, H01, S00, S01` at k (**buggy**) |
| `Gf.py:1211-1238` | `sisl2array_min` — stores real-space blocks in `Hops`/`Sops` (storage convention that `eval_HS_k` misreads) |
| `Gf.py:1032-1043` | `construct_BTD` — builds the block-tridiagonal overlap matrix from `s00, s01` |
| `Gf.py:1044-1060` | `calc_Sinv_Hl_Sinv` — surface block of `S⁻¹ H S⁻¹` |
| `Gf.py:1079-1083` | `calc_Sinv` — returns `Sinv_ll` (surface block of `S⁻¹`) |
| `Gf.py:956-965` | `read_overlap_data` — assembles `sig0`/`sig1` from `Sinv_ll` and `X` |

## Root cause

**Index-convention mismatch between `sisl2array_min` and `eval_HS_k`.**

1. `sisl2array_min` (Gf.py:1225-1236) stores blocks at `Hops[I,J,K]` using
   raw cell indices `I ∈ {-(n//2), …, n//2}`. NumPy negative indexing wraps
   cell −1 into the **last** array slot. Along each axis, array index `i`
   therefore holds cell:

   ```
   i:      0    1    …  n//2   n//2+1  …  n-1
   cell:   0   +1    … +n//2  -n//2    …  -1
   ```

   (Verified numerically: `Hm[0,2,0]` equals `HS[..,(0,-1,0)]` for n=3.)

2. `eval_HS_k` (Gf.py:1022-1030) builds the phase array from the **arrays**
   `Nd1, Nd2` returned by `get_UC_inds`:
   `Nd1 = arange(-(n1//2), n1//2+1)`, `Nd2 = arange(-(n2//2), n2//2+1)`,
   i.e. NumPy arrays of transverse cell offsets such as `[−1, 0, +1]` — not
   scalars. It then multiplies them **positionally** with the stored blocks:

   ```python
   exp_ph = np.exp(2j*np.pi*np.add.outer(kv[0]*Nd1, kv[1]*Nd2))[:,:, None, None]
   def dosum(L):
       return [(exp_ph * l).sum(axis=(0,1)) for l in L]
   ```

   This assumes array index `i` ↔ cell `Nd[i]` (cell −1 at index 0) — the
   opposite of the actual storage. Concretely, for n=3 the on-site block
   (cell 0) gets phase `e^{−2πik}`, the +1 block gets `e^{0}`, and the −1
   block gets `e^{+2πik}`.

3. At k = 0 all phases are 1, so the scrambling is invisible and the sum is
   Hermitian. At k ≠ 0, Hermiticity of a Bloch sum relies on pairing block R
   with `e^{+ikR}` and block −R with `e^{−ikR}` (given real-space Hermiticity
   `S(R)† = S(−R)`). The scrambled pairing breaks that, so `s00(k)` is
   non-Hermitian.

**Propagation to the correction matrices:**

```
eval_HS_k:  s00(k) non-Hermitian
  → SE_1(1.0, 0, 0, s00, s01):  igb = s00 non-Hermitian → Sinv_sur non-Hermitian
  → construct_BTD:  diagonal blocks (s00 − Sinv_sur) and s00 non-Hermitian
    → block-tridiagonal overlap matrix non-Hermitian (off-diagonals s01 / s01† are fine)
  → block_td.Invert:  inverse of a non-Hermitian matrix is non-Hermitian
  → calc_Sinv:  Sinv_ll non-Hermitian
  → calc_Sinv_Hl_Sinv:  X = [S⁻¹ H S⁻¹] surface block non-Hermitian
  → read_overlap_data (Gf.py:959-965):
      sig1 = S_D_l @ Sinv_ll @ S_D_l†                       (inherits)
      sig0 = S_D_l @ X @ S_D_l† − S_D_l @ Sinv_ll @ H_D_l†
             − H_D_l @ Sinv_ll @ S_D_l†                     (inherits)
```

The `sig0`/`sig1` formulas themselves are Hermiticity-preserving; they only
inherit the defect from `Sinv_ll` and `X`.

Corroborating evidence: the commented-out `get_elec_overlap_inv` at the
bottom of `Gf.py` (~lines 1300-1322) performs the same Bloch sum **correctly**
using cell-indexed access `H00[i1,i2]` with `i1 ∈ Nd1` (negative indices wrap
to the right cell). The vectorized `eval_HS_k` is a regression of that code.

## Reproduction

Build a small sisl lead (2 orbitals/cell, square lattice, complex hoppings,
nsc=(3,3,1), lead along `+a1`), then check Hermiticity of `s00(k)`,
`calc_Sinv(k)` and `converge_Sinv_Hl_Sinv(k)` at Gamma and at sampled k.
Run with `python3 repro.py` (requires `sisl`, and `Gf_Module` +
`Block_matrices` importable):

```python
import numpy as np
import sisl
from Gf_Module.Gf import ortho_lead

rng = np.random.default_rng(42)
no = 2  # orbitals per cell

lat = sisl.Lattice(np.array([[1.,0,0],[0,1.,0],[0,0,4.]]), nsc=[3,3,1])
geom = sisl.Geometry(np.array([[0.0,0,0],[0.4,0.3,0.0]]), atoms=sisl.Atom(1), lattice=lat)
HS = sisl.Hamiltonian(geom, nsc=[3,3,1], orthogonal=False, dtype=np.complex128)

def set_hs(io, jo, R, h, s):
    HS[io, jo, R] = (h, s)
    HS[jo, io, tuple(-np.array(R))] = (np.conj(h), np.conj(s))

A = rng.normal(size=(no,no)) + 1j*rng.normal(size=(no,no)); A = A + A.conj().T
B = rng.normal(size=(no,no)) + 1j*rng.normal(size=(no,no)); B = B + B.conj().T
for io in range(no):
    for jo in range(no):
        HS[io, jo, (0,0,0)] = (A[io,jo], B[io,jo])

for R in [(1,0,0), (0,1,0)]:  # complex hoppings along lead (x) and transverse (y)
    T = (rng.normal(size=(no,no)) + 1j*rng.normal(size=(no,no))) * 0.3
    U = (rng.normal(size=(no,no)) + 1j*rng.normal(size=(no,no))) * 0.1
    for io in range(no):
        for jo in range(no):
            set_hs(io, jo, R, T[io,jo], U[io,jo])

ol = ortho_lead(HS, '+a1')

def herm_err(M):
    return float(np.abs(M - M.conj().T).max())

def correct_blocks(kvec):
    """Reference Bloch sum with cell-indexed access (negative indices wrap
    correctly), equivalent to the commented-out get_elec_overlap_inv."""
    Hm, Sm = ol.Hm, ol.Sm
    nx,ny,nz = Hm.shape[:3]
    Ny = np.arange(-(ny//2), ny//2+1); Nz = np.arange(-(nz//2), nz//2+1)
    kv = [kvec[1], kvec[2]]
    H00, H01 = Hm[0], Hm[1]
    S00, S01 = Sm[0], Sm[1]
    h00=h01=s00=s01=0.0
    for i1 in Ny:
        for i2 in Nz:
            ph = np.exp(2j*np.pi*(kv[0]*i1 + kv[1]*i2))
            h00 = h00 + H00[i1,i2]*ph
            h01 = h01 + H01[i1,i2]*ph
            s00 = s00 + S00[i1,i2]*ph
            s01 = s01 + S01[i1,i2]*ph
    return h00,h01,s00,s01

print("-- eval_HS_k vs correct Bloch sum, and Hermiticity of s00 --")
for k in [(0.0,0.0,0.0), (0.0,0.13,0.0), (0.0,0.31,0.0)]:
    k = np.array(k)
    h00, h01, s00, s01 = ol.eval_HS_k(k)
    ch00, ch01, cs00, cs01 = correct_blocks(k)
    print(f"k={tuple(k)}: |s00_code - s00_correct| = {np.abs(s00-cs00).max():.3e}   "
          f"herm_err(s00_code) = {herm_err(s00):.3e}   herm_err(s00_correct) = {herm_err(cs00):.3e}")

print("\n-- downstream correction-matrix Hermiticity --")
for k in [(0.0,0.0,0.0), (0.0,0.13,0.0), (0.0,0.31,0.0)]:
    k = np.array(k)
    Sinv_ll = ol.calc_Sinv(k, 10)
    X = ol.converge_Sinv_Hl_Sinv(k, tol=1e-9, maxcount=40, fail_on_break=False)
    print(f"k={tuple(k)}: herm_err(Sinv_ll) = {herm_err(Sinv_ll):.3e}   "
          f"herm_err(Sinv_Hl_Sinv) = {herm_err(X):.3e}")
```

### Observed output (unpatched code)

```
-- eval_HS_k vs correct Bloch sum, and Hermiticity of s00 --
k=(0.0, 0.0, 0.0):  |s00_code - s00_correct| = 0.000e+00   herm_err(s00_code) = 0.000e+00   herm_err(s00_correct) = 0.000e+00
k=(0.0, 0.13, 0.0): |s00_code - s00_correct| = 1.192e+00   herm_err(s00_code) = 2.220e+00   herm_err(s00_correct) = 0.000e+00
k=(0.0, 0.31, 0.0): |s00_code - s00_correct| = 2.466e+00   herm_err(s00_code) = 2.788e+00   herm_err(s00_correct) = 0.000e+00

-- downstream correction-matrix Hermiticity --
k=(0.0, 0.0, 0.0):  herm_err(Sinv_ll) = 4.441e-16   herm_err(Sinv_Hl_Sinv) = 1.066e-14
k=(0.0, 0.13, 0.0): herm_err(Sinv_ll) = 6.329e+00   herm_err(Sinv_Hl_Sinv) = 3.467e+01
k=(0.0, 0.31, 0.0): herm_err(Sinv_ll) = 3.160e+00   herm_err(Sinv_Hl_Sinv) = 1.693e+01
```

Exactly the reported symptom: Hermitian at Gamma, non-Hermitian (O(1)–O(10)
errors) at sampled k-points. The correct cell-indexed reference sum is
Hermitian at all k, and `eval_HS_k` disagrees with it by O(1) at non-Gamma k.

## Impact

- `Sig0`/`Sig1` (the device–electrode orthogonalization corrections used for
  the Zandpack workflow) are non-Hermitian at every sampled k-point with a
  nonzero transverse component.
- Because the phases also corrupt `h00/h01/s01`, the correction matrices are
  not merely non-Hermitian — their **values are wrong** at non-Gamma k.
- Gamma-only calculations are unaffected (the bug is a no-op at k = 0).

## How to fix (guide)

The fix must make the positional phase multiplication in `eval_HS_k` pair
each stored block with the phase of the cell it actually holds. Two options:

1. **Recommended (minimal, local):** reorder the transverse axes of each
   block with `np.take` before the positional sum. `np.take` wraps negative
   indices, so after `l.take(Nd1, axis=0).take(Nd2, axis=1)` position `i`
   holds cell `Nd[i]` and the existing `exp_ph` is correct. Note that after
   `Hm.take(0, axis=ax)` the two transverse axes are always axes 0 and 1 of
   the taken block, in the same order as `Nd1, Nd2` (because `Nx_i =
   [Nx,Ny,Nz]` with `ax` popped preserves order) — so the same two lines work
   for `ax = 0, 1, 2`.
2. **Alternative:** keep the blocks as stored and build `exp_ph` from the
   actual cell of each array index,
   `cell(i) = i if i <= n//2 else i - n`, instead of from `Nd`.

Do **not** fix this in `sisl2array_min` by shifting the storage index
(e.g. `Hops[I + n//2]`): the lead-axis indexing in `eval_HS_k`
(`Hm.take(0)` = cell 0, `Hm.take(d)` with `d = ±1`) relies on the current
layout (`take(-1)` = last slot = cell −1), so changing the storage convention
would require touching that as well. Fixing `eval_HS_k` is the minimal,
self-contained change.

## Recommended minimal fix

In `Gf_Module/Gf.py`, `ortho_lead.eval_HS_k` (Gf.py:1024-1025), insert the
reordering right after the blocks are taken:

```diff
         H00, H01 = Hm.take(0,axis=ax), Hm.take(d,axis=ax)
         S00, S01 = Sm.take(0,axis=ax), Sm.take(d,axis=ax)
+        # sisl2array_min stores cell -1 at the last array index along each
+        # axis (negative-index wraparound), so array index i holds cell
+        # [0, +1, ..., +n//2, -n//2, ..., -1][i]. Reorder the transverse axes
+        # so position i holds cell Nd[i] before applying exp_ph positionally;
+        # otherwise the Bloch phases are assigned to the wrong blocks and
+        # S00(k)/H00(k) are non-Hermitian for k != 0.
+        H00 = H00.take(Nd1, axis=0).take(Nd2, axis=1)
+        H01 = H01.take(Nd1, axis=0).take(Nd2, axis=1)
+        S00 = S00.take(Nd1, axis=0).take(Nd2, axis=1)
+        S01 = S01.take(Nd1, axis=0).take(Nd2, axis=1)
         exp_ph  =  np.exp(2j*np.pi*
                           np.add.outer(kv[0]*Nd1, 
                                        kv[1]*Nd2 ))[:,:, None, None]
```

`Nd1` and `Nd2` here are the **arrays** of transverse cell offsets built in
`get_UC_inds` (`np.arange(-(n//2), n//2+1)`, e.g. `[−1, 0, +1]`), not
scalars. That distinction governs the shapes: `np.take` with an **array** of
indices is a gather along that axis — the axis is kept, with length
`len(indices)` — whereas `np.take` with a **scalar** index (the
`Hm.take(0, axis=ax)` / `Hm.take(d, axis=ax)` calls above) removes the axis.
Because for odd `nsc` `len(Nd1) = n1` and `len(Nd2) = n2`, each of the two
chained takes rewrites one transverse axis **with the same length**, so the
block shape stays `(n1, n2, no, no)` throughout:

| step | shape |
|---|---|
| `Hm` / `Sm` | `(nx, ny, nz, no, no)` |
| `take(0, axis=ax)` / `take(d, axis=ax)` (scalar index) | `(n1, n2, no, no)` — lead axis removed |
| `take(Nd1, axis=0)` (array index) | `(n1, n2, no, no)` — axis kept, len = len(Nd1) = n1 |
| `take(Nd2, axis=1)` (array index) | `(n1, n2, no, no)` — axis kept, len = len(Nd2) = n2 |
| `exp_ph * l`, then `.sum(axis=(0,1))` | `(no, no)` |

In general, `a.take(ind, axis=k)` replaces axis `k` with `ind.shape`
(`a.shape[:k] + ind.shape + a.shape[k+1:]`), which is why a scalar index
loses the axis and an array index keeps it. Caveat: for even `nsc`,
`len(Nd) = n + 1`, so the take would grow the axis by one — but that case is
already corrupted in `sisl2array_min` (see Secondary observations).
Electrodes use odd `nsc`, and the lead axis is asserted to be 3.

## Verification

1. Apply the fix and rerun the reproduction script above. Expected:
   `herm_err(s00_code)` drops to 0 at all k, `|s00_code - s00_correct|`
   drops to 0 at all k, and `herm_err(Sinv_ll)` / `herm_err(Sinv_Hl_Sinv)`
   drop to machine precision (~1e-14) at all k.
2. Regression check: Gamma-point results must be unchanged (the reordering
   is a no-op at k = 0 because all phases are 1).
3. End-to-end: recompute `Sig0`/`Sig1` via `read_overlap_data` on a TBtrans
   dataset with k-sampling and assert `np.allclose(Sig, Sig.conj().T)` per
   k-point.

The reordering was validated numerically before writing this report: with
`l.take(Nd1, axis=0).take(Nd2, axis=1)` applied to `S00`, the result matches
the correct cell-indexed Bloch sum exactly and is Hermitian (error 0.0) at
k = (0,0,0), (0,0.13,0) and (0,0.31,0).

## Notes on sisl behavior and usage

Facts about sisl (0.17.0rc2) that this report relies on, verified
interactively while writing it. Useful when reproducing the bug or writing
tests around `ortho_lead`.

### Constructing a non-orthogonal Hamiltonian

- `sisl.Hamiltonian(geom, nsc=[nx,ny,nz], orthogonal=False, dtype=...)` —
  `orthogonal=False` must be passed **at construction**. Per the sisl
  docstring, matrix elements are then assigned as tuples
  `H[i,j,R] = (H_ij^R, S_ij^R)`. Without it, tuple assignment fails with
  `ValueError: shape mismatch: value array of shape (2,1) could not be
  broadcast to indexing result of shape (1,1)`.
- `nsc` sets the supercell sparsity pattern: valid cell offsets along each
  axis are `-(n//2) … n//2` (odd nsc). This is exactly the range that
  `sisl2array_min` and `get_UC_inds` iterate over. The code asserts
  nsc = 3 along the lead direction (`eval_HS_k`), which is what electrode
  TSHS files from TBtrans provide.

### Element access convention

- `H[io, jo, (I,J,K)]` is the matrix element between orbital `io` in the
  home cell and orbital `jo` in cell `(I,J,K)`. Negative offsets are valid
  and distinct from positive ones (`H[io,jo,(-1,0,0)]` ≠ `H[io,jo,(1,0,0)]`).
- The return layout depends on spin and orthogonality:
  - unpolarized, non-orthogonal: 2-element array `[H, S]`
  - polarized, non-orthogonal: 3-element array `[H_up, H_down, S]` — the
    overlap is spin-independent (assigning a 4-tuple fails with
    `ValueError: cannot reshape array of size 4 into shape (3)`)
  - orthogonal (`transform(orthogonal=True)`): a scalar `H` only
  - `sisl2array_min` relies on this layout: `hijR[0]` = H, `hijR[1]` = S
    (unpolarized), `hijR[2]` = S (polarized).
- sisl does **not** enforce Hermiticity on assignment. The relation to
  maintain manually is `H[io,jo,R] = conj(H[jo,io,-R])` (likewise for S).
  When building test Hamiltonians, always set both `(io,jo,R)` and
  `(jo,io,-R)`.
- `H.spin` compares equal to `sisl.spin.Spin("unpolarized")` /
  `sisl.spin.Spin("polarized")` — this is how `sisl2array_min` branches.

### `transform(orthogonal=...)`

- `HS.transform(orthogonal=False)` (called in `ortho_lead.__init__` before
  `sisl2array_min`) returns a **copy** with the same nsc and the same
  `(H, S)` matrix elements — i.e. it keeps the non-orthogonal storage that
  `sisl2array_min`'s tuple indexing requires.
- `HS.transform(orthogonal=True)` returns a Hamiltonian flagged
  `orthogonal=True` whose overlap is the identity (`Sk(k)` = I) and whose
  element access returns a scalar `H` — tuple indexing such as `hijR[1]`
  would break on it. In my test the on-site H value was unchanged, so do
  not rely on it to Löwdin-transform H; per the `transform` docstring it
  transforms by a matrix or converts spin, and the `orthogonal` flag must
  merely be consistent with the resulting matrix.

### Bloch sums: `Hk` / `Sk`

- `H.Hk(k=...)` / `H.Sk(k=...)` are sisl's own Bloch sums. For a Hermitian
  real-space input they are exactly Hermitian at every k (verified: error
  0.0 at k_y = 0, 0.13, 0.5). They faithfully reflect the stored elements:
  a non-Hermitian input yields a non-Hermitian `Hk` (observed error 1.0 for
  a complex on-site element), so sanity-check real-space Hermiticity first.
- `read_overlap_data` uses `Hd.Hk(k)` / `Sd.Sk(k)` for the device blocks
  `H_D_l` / `S_D_l` — these are correct. The buggy hand-rolled Bloch sum is
  only `ortho_lead.eval_HS_k`, used for the lead blocks.

### Practical checklist for reproducing / testing

- Minimal lead construction (as in the reproduction script):
  `sisl.Lattice(sc, nsc=[3,3,1])`,
  `sisl.Geometry(xyz, atoms=sisl.Atom(1), lattice=lat)` (note: the keyword
  is `atoms=`, not `atom=`),
  `sisl.Hamiltonian(geom, nsc=[3,3,1], orthogonal=False, dtype=np.complex128)`.
- Set Hermitian on-site blocks and hoppings in both ±R directions with
  conjugate values.
- Sanity checks before trusting any result: real-space Hermiticity
  `H[io,jo,R] == conj(H[jo,io,-R])`; per-k Hermiticity of `Hk`/`Sk`; and
  for `ortho_lead` outputs, `M == M†` at each k.
- In the real workflow the electrode Hamiltonian comes from
  `elec.read_TSHS()`; nsc and spin then come from the TSHS file, and
  `ortho_lead` is constructed with `elec.semi_inf` (e.g. `'+a1'` —
  `translate2` needs the sign, see Secondary observations).

## Secondary observations (out of scope for this fix)

- **Even transverse nsc corrupts data in `sisl2array_min`:** for even `n`,
  `arange(-(n//2), n//2+1)` has `n+1` entries and the negative-index
  wraparound makes two different cells write to the same array slot (e.g.
  n=2: cell −1 and cell +1 both land at index 1, the latter overwriting the
  former). Latent bug; typical electrode nsc is odd, but worth a separate
  guard or fix.
- **`translate2` requires a sign in `direc`:** a bare `'a1'` (no `+`/`-`)
  raises `UnboundLocalError` at Gf.py:1249. TBtrans `semi_inf` strings
  include the sign, so this only bites hand-constructed calls.
- The block orientation in `calc_Sinv_Hl_Sinv` (`Hblock`) is consistent with
  `construct_BTD` given `block_td`'s convention (B = subdiagonal,
  C = superdiagonal) — not related to this bug.

### Comments:
Low severity bug that was discovered thanks to the new tests. 
Only affected k-point sampled systems. 
Fix: commit 21dfbfb
Date of discovery: Oct 15 2026
Date of fix:       Oct 24 2026


