#!/usr/bin/env python3
"""phase_detail9d.py — pour les phases données : t_real first/last, top kernels
(par nom exact), top CPY (par nom exact). Usage :
  python3 phase_detail9d.py <prefix> [ph1,ph2,...]
"""
import json, re, sys
from collections import defaultdict

prefix = sys.argv[1]
WANT = {int(x) for x in sys.argv[2].split(",")} if len(sys.argv) > 2 else None

pat = re.compile(r"^(KER|CPY) seq=(\d+) gpu_s=(\d+) gpu_e=(\d+) dur_ns=(\d+) stream=(\d+) corr=(\d+) t_real_us=(\d+) name=(.*)$")

# re-lire et reconstruire phases avec t_real
items = []
with open(prefix + ".names") as f:
    for line in f:
        m = pat.match(line)
        if not m: continue
        kind = 0 if m.group(1) == "KER" else 1
        items.append((int(m.group(3)), int(m.group(4)), int(m.group(8)), kind, m.group(9)))
items.sort(key=lambda x: (x[0], x[1]))
GAP = int(30 * 1e6)
phases, cur, ue = [], [], -1
for it in items:
    if cur and it[0] > ue + GAP:
        phases.append(cur); cur, ue = [], -1
    cur.append(it)
    if ue == -1 or it[1] > ue: ue = it[1]
if cur: phases.append(cur)

ph84_wall = None
for pi, ph in enumerate(phases):
    if WANT and pi not in WANT: continue
    s0, e0 = min(x[0] for x in ph), max(x[1] for x in ph)
    tr0 = min(x[2] for x in ph); tr1 = max(x[2] for x in ph)
    kers = [x for x in ph if x[3] == 0]
    cpys = [x for x in ph if x[3] == 1]
    kd = defaultdict(lambda: [0, 0.0]); cd = defaultdict(lambda: [0, 0.0])
    for x in kers:
        kd[x[4]][0] += 1; kd[x[4]][1] += x[1]-x[0]
    for x in cpys:
        cd[x[4]][0] += 1; cd[x[4]][1] += x[1]-x[0]
    print(f"\n===== ph{pi} : wall={ (e0-s0)/1e6:.1f} ms | t_real {tr0} -> {tr1} (dur { (tr1-tr0)/1e6:.0f} ms) | nker={len(kers)} ncpy={len(cpys)}")
    print("  TOP kernels (nom exact) :")
    for n, (c, d) in sorted(kd.items(), key=lambda kv: -kv[1][1])[:12]:
        print(f"    {d/1e6:>9.2f} ms x{c:<6} {n[:95]}")
    if cpys:
        print("  TOP cpys :")
        for n, (c, d) in sorted(cd.items(), key=lambda kv: -kv[1][1])[:8]:
            print(f"    {d/1e6:>9.2f} ms x{c:<6} {n[:95]}")
