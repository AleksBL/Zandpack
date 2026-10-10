# 11_gram_basin_spread — perturbed-start basin spread (gram_cut decision round)

Implementation of project_handoff.md follow-up point 3: measure the
START-TO-START outcome distributions of `tr_solver='exact'` vs
`tr_solver='gram'` (cuts 1e-7 / 1e-9 / 0) so the gram_cut default and
the Ag3Au3 'exact' pin can be decided from DISTRIBUTIONS instead of
single-start comparisons. Two systems: Ag3Au3 lead1 (the system pinned
to 'exact') and VerySmallRSSE lead0 (the notch system, cheap fits).

Method: start c0 = the unjittered production init (the deterministic
`psd_dict_init` poles, passed via `fit(init_poles=...)`); starts
s1..s6 = every init-pole center AND width jittered by U(-1e-6, 1e-6) eV
(seed 20261009). Per run: the production `fit_boosted` with the
system's apply_method/compare_dos_t kwargs, plus Gamma metrics (sample
rms [RELATIVE, `frobenius_rms_error`], pole count, padded-grid min
eigenvalue, wall) and a T-level evaluation (fresh pickle, candidate
lead injected with the other lead FIXED, find_correction,
Renormalise_H, `Inspect_transmission_from_hilbert_transform` vs the
sampled-SE / TBtrans reference). Every run records min eig AND T rms
(the health gate).

## Scripts

| file | role |
|---|---|
| `spread_common.py` | solvers, jittered starts, metrics, JSONL resume |
| `ag3au3_spread.py` | Ag3Au3 lead1 sweep (lead0 fixed = exactera cache) |
| `rsse_spread.py` | RSSE lead0 sweep (lead1 fixed = fresh exact refit on the current 189-pt grid) |
| `analyze.py` | distribution tables + per-start table + `spread_*.png` |
| `sanity_teval.py` | T-eval validation: the exactera pair reproduces the stored T rms 0.0100 (measured 0.009958) |
| `diag_control.py` / `diag_backup.py` / `diag_package.py` | era forensics (below) |
| `diag_cache_data.py` / `diag_cache_attrs.py` / `diag_cache_blocks.py` / `diag_init_skrf.py` | exactera-cache forensics |

## Results (28 runs per system; OMP_NUM_THREADS=1)

### Ag3Au3 lead1 (451 pts, p=27; window/corr (-5,5), twin (-5,9), eta 1e-2)

| solver | wall mean | rel rms med (min/max) | T rms med (min/max) | min-eig range | healthy starts* |
|---|---|---|---|---|---|
| exact | 448 s | 0.016 (0.0045 / 0.105) | 0.020 (0.0075 / **3.44**) | -3.3e4 .. ~0 | 2-3 / 7 |
| gram cut=1e-7 | 101 s | 0.027 (0.0061 / 0.193) | 0.034 (0.029 / 0.324) | -166 .. ~0 | 1 / 7 |
| gram cut=1e-9 | 95 s | 0.021 (0.0045 / 0.193) | **0.014** (0.0097 / 0.196) | -67 .. ~0 | **4 / 7** |
| gram cut=0 | 94 s | identical to 1e-9 at every start | identical | identical | 4 / 7 |

*healthy = padded min-eig |.| <= 1 (the exactera healthy reference is
-0.13) AND T rms <= 0.03 (~3x the 0.0100 reference).

Key observations:
- **exact's own start-to-start spread is catastrophic**: the CONTROL
  start (the actual production configuration!) lands the worst basin of
  the whole study: rel rms 0.0258, min eig -236, T rms 3.44 (T_ref max
  is 1.53). Best exact starts reach T rms 0.0075. Single-start
  comparisons on this system are noise — the 0.0228-vs-0.0149
  gram-vs-exact gap that pinned Ag3Au3 to 'exact' is far INSIDE
  exact's own spread.
- **gram cut=1e-9 == cut=0 bit-identically in all 28 runs** (both
  systems): the 1e-9 truncation never fires on real data. cut=1e-7 is
  ACTIVELY HARMFUL on Ag3Au3 (distinct, worse basins: 1/7 healthy,
  catastrophic 0.118-0.193 rms starts that 1e-9 does not produce).
- gram@1e-9 has the best T-rms MEDIAN (0.014 vs exact 0.020), 4/7
  healthy vs exact 2-3/7, at 4.7x lower wall.

### VerySmallRSSE lead0 (189 pts, p=6; twin (-5,5) vs TBtrans)

| solver | wall mean | rel rms med | T rms med | healthy starts* |
|---|---|---|---|---|
| exact | 29.6 s | 0.0054 | 0.0064 | 5 / 7 |
| gram cut=1e-7 | 7.8 s | 0.0054 | 0.0056 | 6 / 7 |
| gram cut=1e-9 | 8.2 s | 0.0054 | 0.0056 | 6 / 7 |
| gram cut=0 | 8.2 s | identical to 1e-9 at every start | identical | 6 / 7 |

*healthy = min-eig >= -1 AND T rms <= 0.01 (production reference 0.0079).
All three gram cuts are bit-identical on RSSE (no direction is ever
truncated there — confirms the cut_deep_dive finding). exact's control
start (0.0114 / T 0.0064) is one of its POORER basins (jittered starts
reach 0.0037); the s5 basin (min eig -121/-131, both solvers) is the
known non-PSD trap: good sample rms 0.0037, T rms 0.016.

## Decisions (from the distributions)

1. **gram_cut default 1e-9 CONFIRMED.** It is equivalent to no
   truncation (cut=0) on real systems and 1e-7 is harmful. No change
   to the code.
2. **The Ag3Au3 'exact' pin is not supported at the distribution
   level** (gram@1e-9: better health rate 4/7 vs 2-3/7, better T-rms
   median, 4.7x faster). The pin's single-start rationale is inside
   the noise. Recommendation: drop the pin OR — better — treat
   basin-sensitive systems with a multi-start health gate (below);
   the solver choice then only affects speed (gram@1e-9).
3. **Production health gate (user-directed)**: select fits by padded
   min eigenvalue AND T-level (or split) error TOGETHER, never by
   sample rms alone. On Ag3Au3, sample rms 0.105 came with a PSD model
   and T rms 0.025 (bad), while min-eig -236 came with rms 0.026 and
   T rms 3.44 (catastrophic); only the joint gate separates them.
   For production: run 2-3 jittered starts (the harness's
   `make_starts` recipe), keep the fit that passes the gate.
4. **Default solver**: gram@1e-9 is the best speed/quality point on
   both systems; 'exact' retains no measured quality advantage — its
   only argument (bit-reproducibility of a given start) holds, but
   given the basin chaos a single reproducible start is worth little
   without the health gate.

## Era forensics (the Ag3Au3 exactera-cache control mismatch)

The control (c0, exact) did NOT reproduce the exactera cache
(rms 0.0171 / 94 poles vs 0.0258 / 108). Investigated exhaustively:

| test | result |
|---|---|
| seeded init_poles vs natural path (no seeding) | bit-identical (0.0258409/108 both) — harness valid |
| `NUMBA_DISABLE_JIT=1` (numpy J path) | bit-identical — numba kernel innocent |
| `backups/AI_fit_2026-10-08_pre_channelcomp.py` | bit-identical — compression round innocent |
| package-era code (`checkpoint_2026-10-08_lorentzfit`, the era that created the cache) | bit-identical — NO code-era regression anywhere in the lineage |
| skrf-relocation init instead of the psd dictionary | another basin (0.0487/65) — not the cache either |
| cache vs today's data (`frobenius_rms_error`, RELATIVE metric) | **bit-identical (lead0), 8 digits (lead1)** — same data |
| cache config fingerprint (min/max widths) | wmin 0.05000, wmax 0.50000 exactly — same eta_floor_mult=5, w_max=0.5 |

Conclusion: same data, same knobs, same code — the exactera fit is
simply ONE SAMPLE from the same chaotic basin distribution; its T rms
0.0100 is reproducible-in-kind and beatable (spread best: T rms 0.0075
at rel rms 0.0045). Nothing about the current code is stale or broken;
what is invalid is comparing single starts on this system (including
the historical 0.0228-vs-0.0149 gram-vs-exact gate and the
lsmr-regression assessment, both of which are within basin noise).

Metric trap documented: `frobenius_rms_error` is RELATIVE
(sqrt(sum|diff|^2 / sum|F|^2)) — comparing it to an absolute rms
produced a false "cache fits today's data better than its own rms"
artifact during the investigation.

## Production test (ag3au3_gram_production.py): gram in the full pipeline

Recipe: 3 starts per lead (c0/s1/s2 spread seeds), gram@1e-9, health
gate (padded min eig > -1 AND rms <= 0.05, best rms among healthy),
selected pair through inject + find_correction(-5,5) +
Inspect_transmission_from_hilbert_transform over (-5,9) at eta=1e-2.

| | T rms | T osc | pair min eig | poles l0/l1 | fit wall (6 fits) |
|---|---|---|---|---|---|
| gram@1e-9, health-gated 3-start | **0.00937** | 0.0041 | -0.33 | 107/98 | 583 s |
| exactera cached pair (stored) | 0.0100 (meas. 0.00996) | 0.0044 | -0.13 | 108/94 | (~6x slower with exact) |
| exact, single control start | 3.44 | 0.35 | -236 | 108/- | 417-854 s/fit |
| old 3.py pipeline (stored) | 0.0383 | 0.0072 | - | - | - |

The gate worked exactly as designed: lead0's control start was
rejected (min eig -1.38, just outside), both leads selected a healthy
start with the best rms (l0 s1: rms 0.00275 / -0.029; l1 s1: rms
0.00449 / -0.33). The gram production result BEATS the stored
exact-era number and the old pipeline at ~6x lower fit cost than
the exact-solver equivalent, and the per-start results reproduce the
spread-round records bit-consistently (l1 c0/s1/s2 identical to the
sweep values). The chosen fits are cached in
`ag3au3_gram_production_fits.npz` (reusable as the fresh Ag3Au3
production cache).

## How to rerun

```
OMP_NUM_THREADS=1 python3 rsse_spread.py      # ~8 min (28 runs)
OMP_NUM_THREADS=1 python3 ag3au3_spread.py    # ~2 h (28 runs)
OMP_NUM_THREADS=1 python3 ag3au3_gram_production.py  # ~12 min
OMP_NUM_THREADS=1 python3 analyze.py --plot
```
Both sweeps resume from their JSONLs (delete to rerun from scratch).
Full data: `ag3au3_results.jsonl`, `rsse_results.jsonl`,
`spread_ag3au3.png`, `spread_rsse.png`.
