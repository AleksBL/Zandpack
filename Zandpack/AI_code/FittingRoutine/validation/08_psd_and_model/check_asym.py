"""Which Data/ and Data2/ examples have asymmetric (lower-triangle)
block structures -> affected by the extract_dense halving bug?"""
import os
import sys

ROOT = "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine"
sys.path.insert(0, os.path.join(ROOT, "bench"))
import extract_data as ex  # noqa: E402

for name in ex.EXAMPLES:
    path = ex.EXAMPLES[name]
    if not os.path.exists(path):
        continue
    R = ex.load_pickle(path)
    for lead in range(R.num_leads):
        G = R.Nonortho_Gammas[lead]
        have = {(int(i), int(j)) for i, j in G.inds}
        missing = [(I, J) for (I, J) in have if (J, I) not in have
                   and I != J]
        tag = f"{name} lead{lead}: " + ("complete" if not missing
                                       else f"MISSING partners {missing}")
        print(tag)
