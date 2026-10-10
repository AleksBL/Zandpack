"""Debug/analysis helper for the dumped cProfile stats: inspect the key
structure and print the aggregated breakdown (the main profile script
crashed on name matching - this works from the .prof file directly)."""
import os
import sys

import pstats

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ps = pstats.Stats(os.path.join(HERE, "ag3au3_gram_c0.prof"))
    print("total_tt:", ps.total_tt)
    print("n stats entries:", len(ps.stats))
    keys = list(ps.stats.keys())[:8]
    for k in keys:
        print("key sample:", k)
    # any AI_fit entries?
    ai = [k for k in ps.stats if "AI_fit" in str(k[0])]
    print("AI_fit-file entries:", len(ai))
    for k in ai[:10]:
        v = ps.stats[k]
        print(f"   {k[2]:30s} cc={v[0]} nc={v[1]} tt={v[2]:.2f} "
              f"ct={v[3]:.2f}")


if __name__ == "__main__":
    main()
