"""AI_fit_sweep.py — reusable parameter-sweep machinery for the
AI_fit_base fitting routine (born from the nogap_Ag5 sweep rounds,
2026-10-09; see Data2/nogap_Ag5/README_gram_refined.md).

Two stages, mirroring the proven workflow:

  1. GAMMA level  sweep_gamma(): fit configs with MULTI-START
     (deterministic control + jittered starts) + HEALTH GATE (padded
     min eigenvalue AND sample rms together - never rms alone), with
     on-disk caching so reruns resume.

  2. T/DOS level  evaluate_pair(): inject a (lead -> fit) pair into
     an R, run the renormalization, and compare T + DOS against a
     sampled-SE reference on a banded grid. Optional plot output.

USAGE (follows the GUIDE_3py_integration.md discipline):

    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")   # BEFORE numpy
    R = pickle.load(gzip.open('tmp.gz', 'rb'))      # pickles FIRST
    sys.path.insert(0, ROOT); import AI_fit_base as lf
    import AI_fit_sweep as sf
    sys.path.remove(ROOT)

    E, F = lf.extract_dense(R, 0, 0)
    sw = sf.GammaSweep(lf, E, F, cache_dir="sweep_cache",
                       base_kw=dict(n_init=24, max_poles=90, psd=True,
                                    eta_raw=2e-2, tr_solver="gram"))
    results = sw.run({
        "w03":      dict(w_max_raw=0.3),
        "w02":      dict(w_max_raw=0.2),
        "w02_m15":  dict(w_max_raw=0.2, eta_floor_mult=1.5),
    })                       # cached fits resume on rerun

    # T level: lead1 may inherit lead0's poles (symmetric systems) or
    # be fitted fully - compute BOTH and let T decide per system:
    pair = sw.best_pair_inherit(R)          # seconds
    row = sf.evaluate_pair(lf, R, pair, E_ref, T_ref, dos_ref,
                           teta=2e-2, corr=(-3, 3), twin=(-5, 9),
                           find_correction_kw=dict(),
                           plot="sweep_cache/fig_w03.png")

Standing rules baked in: OMP_NUM_THREADS=1 (set it before numpy -
caller's job); judge fits by min eig AND rms TOGETHER; CombinedFit.rms
(fit_boosted) is the relative FROBENIUS rms while FitResult.rms
(fit / fit_fixed_poles) is quadrature-weighted on scaled data - the
health gate compares like with like within one stage, and
evaluate_pair measures the model with frobenius_rms_error, the ONE
metric valid across fit object types.
"""
import glob
import json
import os
import pickle
import time

import numpy as np

__all__ = ["GammaSweep", "evaluate_pair", "frobenius_table",
           "pole_inherit_fit"]


def _starts(lf, E, F, n_init, e_margin, n_starts, jitter, seed):
    """Deterministic control start + (n_starts-1) jittered copies.

    The jitter is deliberately tiny (1e-6 eV raw): it does not change
    the physics, it only samples which attractor basin the adaptive
    loop falls into (HANDOFF 4.13 - single-start comparisons on
    basin-sensitive systems carry no information).
    """
    fd = lf.prepare(E, F)
    c0, w0 = lf._poles_init_dispatch("skrf", fd.E, fd.F, n_init,
                                     e_margin=e_margin, psd=True)
    c0 = np.asarray(c0, dtype=float)
    w0 = np.asarray(w0, dtype=float)
    starts = [("c0", c0, w0)]
    rng = np.random.default_rng(seed)
    for s in range(1, n_starts):
        de = rng.uniform(-jitter, jitter, c0.size) / fd.e_scale
        dw = rng.uniform(-jitter, jitter, c0.size) / fd.e_scale
        starts.append((f"s{s}", c0 + de, w0 + dw))
    return starts


def fit_health(lf, r, E, pad=2.0, npts=801):
    """(rms, padded min eigenvalue) health record of a fit.

    rms is whatever the fit object reports (see the module docstring
    for the metric caveat); min_eig is the model's smallest
    eigenvalue on a grid PADDED beyond the window - the PSD health of
    the transport model, not just on the samples.
    """
    Eg = np.linspace(min(E) - pad, max(E) + pad, npts)
    return dict(rms=float(r.rms),
                min_eig=float(np.linalg.eigvalsh(r.evaluate(Eg)).min()))


def is_healthy(rec, min_eig_tol=-1.0, rms_max=0.05):
    """Joint health gate (user-set policy 2026-10-09): padded min eig
    AND sample rms together. Thresholds are heuristics from the
    Ag3Au3/nogap production runs - loosen rms_max for systems that
    honestly cannot reach it, never loosen min_eig_tol below -1."""
    return rec["min_eig"] > min_eig_tol and rec["rms"] <= rms_max


def pole_inherit_fit(lf, E1, F1, f0, psd=True, psd_force=True):
    """Lead1 from lead0's poles (symmetric systems): fit_fixed_poles
    at the FIXED inherited poles. MATCH f0's psd policy (psd_force
    only if the source fit's PSD model was applied)."""
    return lf.fit_fixed_poles(E1, F1, f0.e, f0.w, psd=psd,
                             psd_force=psd_force)


class GammaSweep:
    """Multi-start, health-gated, cached Gamma-level fit sweep.

    NOTE on init-affecting configs: the sweep always passes explicit
    init_poles (control + jittered starts). A config that sets
    'init', 'n_init' or 'e_margin' gets its OWN start set generated
    through _poles_init_dispatch with those values (so the knob is
    actually exercised). Under psd=True the dispatch ALWAYS uses the
    NNLS PSD dictionary regardless of the init MODE (AI_fit_base's
    _poles_init_dispatch design) - 'init' is only a real knob for
    psd=False fits.

    n_res: residual-boost poles (fit_boosted). n_res=None runs the
    PLAIN adaptive fit (no boost) - use this for pole budgets where
    the boost's extra poles are not accounted (n_res=0 does NOT
    mean "no boost": fit_boosted pops max_poles from the residual
    kwargs, so n_res=0 silently runs an UNCAPPED residual fit).
    """

    START_KEYS = ("init", "n_init", "e_margin")

    def __init__(self, lf, E, F, cache_dir="sweep_cache", base_kw=None,
                 n_res=8, n_starts=3, n_init=24, e_margin=0.15,
                 jitter=1e-6, seed=1,
                 min_eig_tol=-1.0, rms_max=0.05, verbose=True,
                 lead_label=0):
        """lead_label: cache/log label for this (lead, ik) slice -
        for MULTI-K systems run one GammaSweep per (lead, ik) with
        lead_label=f"{lead}_ik{ik}" so the caches never collide."""
        self.lf = lf
        self.E, self.F = E, F
        self.cache_dir = cache_dir
        self.base_kw = dict(base_kw or {})
        self.n_res = n_res
        self.n_starts = n_starts
        self.n_init = n_init
        self.e_margin = e_margin
        self.jitter = jitter
        self.seed = seed
        self.min_eig_tol = min_eig_tol
        self.rms_max = rms_max
        self.verbose = verbose
        self.lead_label = lead_label
        self.channel_tol = 1e-4 if np.asarray(F).shape[1] > 32 else 0.0
        os.makedirs(cache_dir, exist_ok=True)
        self.starts = _starts(lf, E, F, n_init, e_margin, n_starts,
                              jitter, seed)
        self.results = {}

    def _cache_path(self, name, lead, tag):
        return os.path.join(self.cache_dir,
                            f"{name}_lead{lead}_{tag}.npz")

    def _fit_one(self, name, extra, tag, c0, w0):
        cache = self._cache_path(name, self.lead_label, tag)
        if os.path.exists(cache):
            d = np.load(cache, allow_pickle=True)
            return d["fit"].item(), dict(d["rec"].item())
        t = time.time()
        kw = dict(self.base_kw,
                  channel_tol=self.channel_tol,
                  init_poles=(c0, w0))
        kw.update(extra)
        if self.n_res is None:
            # plain adaptive fit, NO residual boost. NOTE: do NOT use
            # n_res=0 for this - fit_boosted(n_res=0) runs an UNCAPPED
            # adaptive fit of the residual (max_poles is popped from
            # res_kw), which silently violates any pole budget.
            r = self.lf.fit(self.E, self.F, **kw)
        else:
            r = self.lf.fit_boosted(self.E, self.F, kw,
                                    n_res=self.n_res)
        rec = fit_health(self.lf, r, self.E)
        rec["wall"] = time.time() - t
        np.savez_compressed(cache, fit=r, rec=rec)
        return r, rec

    def _config_starts(self, extra):
        """Start set for one config: the constructor's unless the
        config touches the init-affecting knobs."""
        if not any(k in extra for k in self.START_KEYS):
            return self.starts
        fd = self.lf.prepare(self.E, self.F)
        kw = dict(init=extra.get("init", "skrf"),
                   n_init=extra.get("n_init", self.n_init),
                   e_margin=extra.get("e_margin", self.e_margin))
        c0, w0 = self.lf._poles_init_dispatch(
            kw["init"], fd.E, fd.F, kw["n_init"],
            e_margin=kw["e_margin"], psd=True)
        c0 = np.asarray(c0, dtype=float)
        w0 = np.asarray(w0, dtype=float)
        starts = [("c0", c0, w0)]
        rng = np.random.default_rng(self.seed)
        for s in range(1, self.n_starts):
            de = rng.uniform(-self.jitter, self.jitter,
                             c0.size) / fd.e_scale
            dw = rng.uniform(-self.jitter, self.jitter,
                             c0.size) / fd.e_scale
            starts.append((f"s{s}", c0 + de, w0 + dw))
        return starts

    def run(self, configs, lead=0):
        """configs: {name: extra fit kwargs}. Returns {name: record}
        with the health-gated best fit per config (best rms among
        healthy starts, else best of bad lot with a warning flag)."""
        for name, extra in configs.items():
            cands = []
            for tag, c0, w0 in self._config_starts(extra):
                r, rec = self._fit_one(name, extra, tag, c0, w0)
                rec.update(tag=tag, r=r)
                cands.append(rec)
                if self.verbose:
                    print(f"[sweep:{name}] lead{lead} {tag}: "
                          f"rms {rec['rms']:.4g} n={r.n_poles} "
                          f"minEig {rec['min_eig']:+.3g}", flush=True)
            good = [c for c in cands
                    if is_healthy(c, self.min_eig_tol, self.rms_max)]
            pick = min(good or cands, key=lambda c: c["rms"])
            pick = dict(pick)
            pick["n_healthy"] = len(good)
            pick["no_healthy_start"] = not good
            self.results[name] = pick
            if self.verbose:
                print(f"[sweep:{name}] selected {pick['tag']} "
                      f"({len(good)}/{len(cands)} healthy, "
                      f"rms {pick['rms']:.4g})", flush=True)
        return self.results

    def best_pair_inherit(self, R, name=None, lead0=0, lead1=1, ik=0,
                          psd=True, psd_force=True):
        """(lead0, lead1) pair with lead1 INHERITING lead0's poles.
        For the own-fit alternative run a second GammaSweep on lead1
        and compare at the T level (inheritance is an option, not a
        mandate - HANDOFF / guide section 8)."""
        pick = (self.results[name] if name is not None
                else min(self.results.values(),
                         key=lambda c: c["rms"]))
        f0 = pick["r"]
        E1, F1 = self.lf.extract_dense(R, lead1, ik)
        f1 = pole_inherit_fit(self.lf, E1, F1, f0,
                              psd=psd, psd_force=psd_force)
        return {(lead0, ik): f0, (lead1, ik): f1}, pick, f1


def evaluate_pair(lf, R, pair, E_ref, T_ref, dos_ref, teta, corr,
                  twin, bands=None, plot=None, title="",
                  renormalise=True, E_fit=None, F_fit=None):
    """Inject `pair` {(lead, ik): fit} into R, renormalize, and score
    T + DOS against the sampled-SE reference.

    corr  : (Emin, Emax) window for find_correction (MUST match the
            system's own 3.py)
    twin  : (Emin, Emax) T/DOS comparison window
    teta  : comparison eta - MUST equal the contour eta for van-Hove
            systems (HANDOFF 6.3 nogap lesson)
    bands : optional [(a, b), ...] sub-window rms breakdown
    plot  : optional path; writes a 3-panel figure (T, DOS, fitted
            function vs sampled with the trace channel + inset)
    E_fit/F_fit: optional (E, F) of lead0 on the FIT grid, for the
            fitted-function panel + the fit rms row

    NOTE: R is MUTATED by inject_fits - pass a freshly loaded copy
    per evaluation (the loop `RX = pickle.load(...)` pattern in the
    Data2 drivers).
    """
    from Zandpack.FittingTools import find_correction
    RX = R
    lf.inject_fits(RX, pair)
    if renormalise:
        RX.Renormalise_H(find_correction(RX, Emin=corr[0],
                                         Emax=corr[1]))
    n = 600
    Eg = np.linspace(twin[0], twin[1], n)
    _, T_new = RX.Inspect_transmission_from_hilbert_transform(
        Eg, eta=teta, return_result=True)
    _, dos_k = RX.Inspect_dos_from_hilbert_transform(
        Eg, eta=teta, return_result=True)
    dos_new = (dos_k * RX.tbtwkpt[:, None]).sum(axis=0)
    T_at = np.interp(Eg, E_ref, T_ref)
    dos_at = np.interp(Eg, E_ref, dos_ref)

    def rms(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    row = dict(t_rms=rms(T_new.real, T_at),
               dos_rms=rms(dos_new, dos_at),
               osc=float(np.abs(np.diff(T_new.real - T_at)).mean()),
               n_poles={str(k): int(v.n_poles) for k, v in
                        pair.items()})
    if E_fit is not None:
        f0 = pair[(0, 0)]
        row["fit_rms_0"] = float(lf.frobenius_rms_error(
            f0.evaluate(E_fit), F_fit))
    if bands:
        row["band_t"] = {f"{a}_{b}":
                         rms(T_new.real[(Eg >= a) & (Eg < b)],
                             T_at[(Eg >= a) & (Eg < b)])
                         for a, b in bands}

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
        ax[0].plot(E_ref, T_ref, ".", ms=3, color="k",
                   label="sampled-SE ref")
        ax[0].plot(Eg, T_new.real, "-", lw=1.2, color="tab:red",
                   label="fit")
        ax[0].set_ylabel("T")
        ax[0].grid(alpha=0.3)
        ax[0].legend(fontsize=9)
        ax[0].set_title(f"{title}  T rms {row['t_rms']:.4g}, "
                        f"DOS rms {row['dos_rms']:.3g}, "
                        f"n={row['n_poles']}")
        ax[1].plot(E_ref, dos_ref, ".", ms=3, color="k",
                   label="sampled-SE ref")
        ax[1].plot(Eg, dos_new, "-", lw=1.2, color="tab:red")
        ax[1].set_ylabel("DOS")
        ax[1].grid(alpha=0.3)
        ax[1].legend(fontsize=9)
        if E_fit is not None:
            tr_smp = np.real(np.trace(F_fit, axis1=1, axis2=2))
            Egd = np.linspace(min(E_fit), max(E_fit), 2401)
            tr_fit = np.real(np.trace(f0.evaluate(Egd),
                                      axis1=1, axis2=2))
            ax[2].plot(E_fit, tr_smp, ".", ms=3, color="k",
                       label="sampled")
            ax[2].plot(Egd, tr_fit, "-", lw=1.2, color="tab:red",
                       label="fit")
            ax[2].set_ylabel(r"Tr $\Gamma_0$")
            ax[2].legend(fontsize=9)
            inset = ax[2].inset_axes([0.55, 0.35, 0.43, 0.6])
            m = (E_fit >= 2.0) & (E_fit <= 4.5)
            inset.plot(E_fit[m], tr_smp[m], ".", ms=3, color="k")
            inset.plot(Egd, tr_fit, "-", lw=1.2, color="tab:red")
            inset.set_xlim(2.0, 4.5)
            inset.grid(alpha=0.3)
        ax[2].set_xlabel("E (eV)")
        ax[2].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot, dpi=120)
        plt.close(fig)
    return row


def frobenius_table(lf, results, E, F, by="t_rms"):
    """Pretty-print a results dict (from evaluate_pair runs)."""
    if results:
        print(f"{'variant':<28} " +
              " ".join(f"{k:>9}" for k in
                       ("t_rms", "dos_rms", "osc")))
        for k, v in sorted(results.items(),
                           key=lambda kv: kv[1].get(by, 1e9)):
            print(f"{k:<28} {v['t_rms']:9.4f} "
                  f"{v['dos_rms']:9.3f} {v['osc']:9.4g}")
