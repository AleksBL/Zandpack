#!/usr/bin/env python3
"""Distribution of the default cap over ALL existing sweep caches.

For every Data2 system with npz fit caches and a usable pickle, run
cap_residues (default knobs, cap_mult=10) on every cache and record
violator/acceptance statistics. Gives a distribution rather than the
single Fig2Ag5 case (review 2026-10-13, point 4).

Mapping guard: each system's pickle is chosen as tmp_refined_deriv.gz
> newest tmp_refined*.gz > tmp.gz; a cache whose frobenius rms on the
extracted slice is > 20x its stored rec rms is flagged MISMATCHED
(wrong data) and excluded from the aggregates.

Run:  OMP_NUM_THREADS=1 python validation/cap_distribution.py
Out:  validation/cap_distribution.json
"""
import glob
import gzip
import json
import os
import pickle
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

ROOT = "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine"
DATA2 = os.path.join(ROOT, "Data2")
OUT = os.path.join(ROOT, "validation", "cap_distribution.json")

warnings.simplefilter("once", UserWarning)
T0 = time.time()


def log(msg):
    print(f"[capdist {time.time()-T0:6.0f}s] {msg}", flush=True)


def load_fit_module():
    """Repo AI_fit_base under BOTH flat aliases (new + old caches)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "AI_fit_base", os.path.join(ROOT, "AI_fit_base.py"))
    lf = importlib.util.module_from_spec(spec)
    sys.modules["AI_fit_base"] = lf
    spec.loader.exec_module(lf)
    # caches were written under THREE module names across eras:
    # lorentz_fit (package-era), AI_fit_base (flat alias), and
    # AI_fit (the old repo-root alias the Data2 drivers imported).
    # Register ALL - the first distribution run silently failed to
    # unpickle 534/600 caches (every system but Fig2Ag5) because
    # only the first two were registered.
    sys.modules.setdefault("lorentz_fit", lf)
    sys.modules.setdefault("AI_fit", lf)
    return lf


def system_pickle(sysdir):
    for pat in ("tmp_refined_deriv.gz", "tmp_refined*.gz", "tmp.gz"):
        cands = sorted(glob.glob(os.path.join(sysdir, pat)),
                       key=os.path.getmtime)
        if cands:
            return cands[-1]
    return None


def main():
    results = []
    systems = sorted(d for d in glob.glob(os.path.join(DATA2, "*"))
                     if os.path.isdir(d)
                     and glob.glob(os.path.join(d, "sweep*", "*.npz")))
    log(f"systems with caches: {[os.path.basename(s) for s in systems]}")
    for sysdir in systems:
        name_sys = os.path.basename(sysdir)
        pkl = system_pickle(sysdir)
        if pkl is None:
            log(f"{name_sys}: NO pickle - skipped")
            continue
        # pickle FIRST (imports Zandpack from the installed package),
        # then the fit module under its flat aliases
        _path0 = list(sys.path)
        with gzip.open(pkl, "rb") as f:
            R = pickle.load(f)
        for q in [q for q in sys.path if q not in _path0]:
            sys.path.remove(q)
        lf = load_fit_module()
        caches = sorted(glob.glob(os.path.join(sysdir, "sweep*",
                                                "*.npz")))
        log(f"{name_sys}: {len(caches)} caches, pickle "
            f"{os.path.basename(pkl)}")
        n_viol = n_accept = n_fail = n_mismatch = 0
        for cache in caches:
            base = os.path.basename(cache)[:-4]
            parts = base.split("_")
            try:
                lead = int(parts[-3].replace("lead", ""))
                ik = int(parts[-2].replace("ik", ""))
            except (ValueError, IndexError):
                continue
            try:
                z = np.load(cache, allow_pickle=True)
                fit = z["fit"].item()
                rec = z["rec"].item() if "rec" in z else {}
            except Exception as exc:                              # noqa: BLE001
                results.append(dict(system=name_sys, cache=base,
                                    error=f"unpickle: {exc}"))
                continue
            try:
                E, F = lf.extract_dense(R, lead, ik)
            except Exception as exc:                              # noqa: BLE001
                results.append(dict(system=name_sys, cache=base,
                                    error=f"extract: {exc}"))
                continue
            scale = float(np.sqrt(np.sum(np.abs(F) ** 2,
                                         axis=(1, 2))).max())
            norms = lf.residue_norms(np.asarray(fit.C))
            worst = float(norms.max() / max(scale, 1e-300))
            row = dict(system=name_sys, cache=base, pickle=os.path.basename(pkl),
                       n_poles=int(fit.n_poles), worst_res_over_cap=worst)
            if worst <= 10.0:
                n_viol += 0
                row["violators"] = False
                results.append(row)
                continue
            n_viol += 1
            # mapping sanity: the uncapped fit should match this data
            try:
                frob = float(lf.frobenius_rms_error(
                    fit.evaluate(E), F))
            except Exception:                                      # noqa: BLE001
                frob = float("nan")
            rec_rms = float(rec.get("rms", float("nan")))
            if rec_rms == rec_rms and frob > 20.0 * max(rec_rms, 1e-12):
                n_mismatch += 1
                row.update(violators=True, mismatch=True, frob=frob,
                           rec_rms=rec_rms)
                results.append(row)
                continue
            try:
                out = lf.cap_residues(E, F, fit, cap_mult=10.0)
                h = out.history
                row.update(violators=True,
                           failed=bool(h.get("residue_cap_failed",
                                            False)),
                           n_dropped=int(h.get("n_dropped", 0)),
                           n_clipped=int(h.get("n_clipped", 0)),
                           rms_uncapped=float(h["rms_uncapped"]),
                           rms_after=float(h["rms"]),
                           frob_before=float(h["frobenius_before"]),
                           frob_after=float(h["frobenius_after"]))
                if row["failed"]:
                    n_fail += 1
                    row["rms_rejected_drop"] = h.get(
                        "rms_rejected_drop")
                    if "band_rejected_drop" in h:
                        row["band_rejected_drop_ratio"] = h[
                            "band_rejected_drop"][
                            "worst_band_ratio"]
                else:
                    n_accept += 1
            except Exception as exc:                               # noqa: BLE001
                row.update(violators=True, error=f"cap: {exc}")
                n_fail += 1
            results.append(row)
        log(f"{name_sys}: violators {n_viol}/{len(caches)}, "
            f"accepted {n_accept}, failed {n_fail}, "
            f"mismatched {n_mismatch}")
    json.dump(results, open(OUT, "w"), indent=1, default=float)
    log(f"wrote {OUT} ({len(results)} rows)")
    # aggregate
    log("--- AGGREGATE (cap_mult=10, default guards) ---")
    for name_sys in sorted({r["system"] for r in results}):
        rows = [r for r in results if r["system"] == name_sys]
        viol = [r for r in rows if r.get("violators")]
        acc = [r for r in viol if r.get("failed") is False]
        fail = [r for r in viol if r.get("failed")]
        mism = [r for r in viol if r.get("mismatch")]
        costs = [r["rms_after"] / max(r["rms_uncapped"], 1e-300)
                 for r in acc if r.get("rms_after") is not None]
        log(f"{name_sys:<14} caches={len(rows):<4} viol={len(viol):<4} "
            f"accept={len(acc):<3} fail={len(fail):<3} "
            f"mismatch={len(mism):<3}"
            + (f" rms-cost median x{np.median(costs):.2f} "
               f"max x{np.max(costs):.2f}" if costs else ""))


if __name__ == "__main__":
    main()
