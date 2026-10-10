"""Sanity check of the Ag3Au3 t_metrics pipeline: inject the exactera
cached pair (lead0 + lead1) and evaluate T rms. Must reproduce the
stored method_summary rms_new ~ 0.0100 if the harness T-eval is
correct."""
import gzip
import os
import pickle
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spread_common as sc  # noqa: E402


def main():
    os.chdir("/tmp")
    with gzip.open(os.path.join(sc.ROOT, "Data2", "Ag3Au3",
                               "tmp.gz"), "rb") as f:
        R = pickle.load(f)
    lf = sc.import_ai_fit()
    sys.path.insert(0, sc.ROOT)
    import refine_sampling  # noqa: F401
    sys.path.remove(sc.ROOT)
    E_ref, T_ref, _ = lf.sampled_transmission_dos(R)
    d = np.load(os.path.join(sc.ROOT, "Data2", "Ag3Au3",
                            "cmp_fits_exactera.npz"),
                allow_pickle=True)
    fits = {(0, 0): d["lead0_ik0"].item(),
            (1, 0): d["lead1_ik0"].item()}

    from Zandpack.FittingTools import find_correction  # noqa: E402
    lf.inject_fits(R, fits)
    R.Renormalise_H(find_correction(R, Emin=-5, Emax=5.0))
    Eg = np.linspace(-5, 9, 600)
    _, T_new = R.Inspect_transmission_from_hilbert_transform(
        Eg, eta=1e-2, return_result=True)
    T_ref_at = np.interp(Eg, E_ref, T_ref)
    print(f"[sanity] exactera pair: T rms "
          f"{sc.rms(T_new.real, T_ref_at):.4g} "
          f"(stored method_summary rms_new 0.0100)", flush=True)
    print(f"[sanity] T_ref range: {T_ref.min():.3g} .. {T_ref.max():.3g}",
          flush=True)


if __name__ == "__main__":
    main()
