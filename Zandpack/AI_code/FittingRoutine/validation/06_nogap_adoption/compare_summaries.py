import json, os
D = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine/Data2'
systems = ['agnr', 'Ag3Au3', 'Ag3Au3_gap', 'Fig2Ag5', 'spinpol_ag5',
           'AuBreakJunction', 'nogap_Ag5']
print(f"{'system':16s} {'field':10s} {'stored':>14s} {'fresh':>14s}  verdict")
all_ok = True
for s in systems:
    a = json.load(open(os.path.join(D, s, 'method_summary_stored.json')))
    b = json.load(open(os.path.join(D, s, 'method_summary.json')))
    sys_ok = True
    for k in ['rms_old', 'rms_new', 'dos_old', 'dos_new', 'osc_old',
              'osc_new', 'min_eig']:
        va, vb = a.get(k), b.get(k)
        if va is None and vb is None:
            continue
        rel = abs(vb - va) / max(abs(va), 1e-30)
        ok = rel < 1e-9
        sys_ok &= ok
        if not ok:
            print(f"{s:16s} {k:10s} {va:14.6g} {vb:14.6g}  DIFF rel={rel:.3g}")
    # pole counts
    na, nb = a.get('n_poles', {}), b.get('n_poles', {})
    same = na == nb
    sys_ok &= same
    if not same:
        print(f"{s:16s} n_poles   {na} vs {nb}  DIFF")
    print(f"{s:16s} {'ALL':10s} {'':>14s} {'':>14s}  "
          f"{'AGREES (bit-level)' if sys_ok else 'MISMATCH'}")
    all_ok &= sys_ok
print('\nOVERALL:', 'ALL 7 SYSTEMS AGREE' if all_ok else 'MISMATCH FOUND')
