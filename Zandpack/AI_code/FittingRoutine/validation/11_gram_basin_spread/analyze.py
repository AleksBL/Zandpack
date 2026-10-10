"""Summarize the basin-spread JSONLs (ag3au3_results.jsonl,
rsse_results.jsonl): per solver, distributions of wall / sample rms /
pole count / padded min eig / T rms / T osc across the perturbed
starts, plus the per-start scatter plot figure (spread.png).

Usage: python3 analyze.py [--plot]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

METRICS = [("wall", "s"), ("rms", None), ("n_poles", None),
           ("min_eig", None), ("t_rms", None), ("t_osc", None)]
SOLVERS = ["exact", "gram_cut1e-7", "gram_cut1e-9", "gram_cut0"]


def load(name):
    path = os.path.join(HERE, name)
    recs = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
    return recs


def stats_table(recs, system):
    print(f"\n===== {system}: distributions across starts "
          f"(c0 control + jittered starts) =====")
    for metric, unit in METRICS:
        rows = []
        for sol in SOLVERS:
            vals = [r[metric] for r in recs
                    if r["solver"] == sol and metric in r]
            if not vals:
                continue
            v = np.asarray(vals, dtype=float)
            rows.append((sol, len(v), v.min(), np.median(v), v.mean(),
                         v.max(), v.std(ddof=1) if len(v) > 1 else 0.0))
        if not rows:
            continue
        print(f"\n  {metric}{' [' + unit + ']' if unit else ''}:")
        print("    solver          n    min      median    mean    "
              "       max        std")
        for sol, n, mn, med, mean, mx, sd in rows:
            print(f"    {sol:14s} {n:2d}  {mn:10.4g} {med:10.4g} "
                  f"{mean:10.4g} {mx:10.4g} {sd:10.4g}")


def per_start(recs, system):
    print(f"\n  {system}: per-start values (rms | n | t_rms | min_eig)")
    starts = sorted({r["start"] for r in recs},
                    key=lambda s: (s != "c0", int(s[1:])))
    print("    start  " + "".join(f"{s:>34s}" for s in SOLVERS))
    for st in starts:
        cells = []
        for sol in SOLVERS:
            m = [r for r in recs
                 if r["solver"] == sol and r["start"] == st]
            if m:
                r = m[0]
                cells.append(f"{r['rms']:8.4g} {r['n_poles']:4d} "
                             f"{r['t_rms']:8.4g} {r['min_eig']:+9.2g}")
            else:
                cells.append(" " * 30)
        print(f"    {st:5s} " + "".join(f"{c:>34s}" for c in cells))


def plot(recs, system, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    panels = [("rms", "sample rms"), ("t_rms", "T rms vs reference"),
              ("min_eig", "padded min eig"), ("wall", "wall [s]")]
    for a, (m, lbl) in zip(ax, panels):
        for sol in SOLVERS:
            vals = [r[m] for r in recs if r["solver"] == sol and m in r]
            a.plot(np.arange(len(vals)), vals, "o-", ms=4, label=sol)
        a.set_title(lbl)
        a.grid(alpha=0.3)
        a.set_xlabel("start index (0 = control)")
    ax[0].legend(fontsize=8)
    fig.suptitle(f"{system}: perturbed-start basin spread")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"spread_{tag}.png"), dpi=130)
    plt.close(fig)
    print(f"  -> spread_{tag}.png")


def main():
    do_plot = "--plot" in sys.argv
    for name, system, tag in [("ag3au3_results.jsonl", "Ag3Au3", "ag3au3"),
                              ("rsse_results.jsonl", "VerySmallRSSE",
                               "rsse")]:
        recs = load(name)
        if not recs:
            print(f"[{system}] no records yet")
            continue
        stats_table(recs, system)
        per_start(recs, system)
        if do_plot:
            plot(recs, system, tag)
    print()


if __name__ == "__main__":
    main()
