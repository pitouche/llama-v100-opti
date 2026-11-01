#!/usr/bin/env python3
"""phase9d.py — découpage en phases (gaps union >30 ms) de la trace run9d +
breakdown familles par phase. Usage :
  python3 phase9d.py <prefix> [gap_ms] [n_top]
Sort : stdout = rapport texte ; <prefix>.phases.json = détail machines."""
import json, re, sys, time
from collections import defaultdict

prefix = sys.argv[1]
GAP_MS = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
N_TOP = int(sys.argv[3]) if len(sys.argv) > 3 else 25

pat = re.compile(r"^(KER|CPY) seq=(\d+) gpu_s=(\d+) gpu_e=(\d+) dur_ns=(\d+) stream=(\d+) corr=(\d+) t_real_us=(\d+) name=(.*)$")

t0 = time.time()
iv = []  # (s, e, kind, name)
with open(prefix + ".names") as f:
    for line in f:
        m = pat.match(line)
        if not m:
            continue
        k = m.group(1)
        s, e = int(m.group(3)), int(m.group(4))
        nm = m.group(9)
        if k == "KER":
            iv.append((s, e, 0, nm))
        else:
            iv.append((s, e, 1, nm))
print(f"lignes KER/CPY : {len(iv)}  ({time.time()-t0:.1f}s)", flush=True)

iv.sort(key=lambda x: (x[0], x[1]))

# ---- merge par stream? -> union globale : tri, merge global (multi-stream chevauche) ----
def union_stats(items):
    # items: (s,e) -> (union_ns, overlap_ns)
    us, ov = 0, 0
    cs, ce = None, None
    for s, e in items:
        if cs is None:
            cs, ce = s, e
            continue
        if s > ce:
            us += ce - cs
            cs, ce = s, e
        else:
            ov += min(ce, e) - s
            ce = max(ce, e)
    us += ce - cs
    return us, ov

# ---- découpage en phases sur la union ----
# on itère les items triés ; si le début d'un intervalle dépasse la fin courante
# de l'union d'au moins GAP -> nouvelle phase.
GAP_NS = int(GAP_MS * 1e6)
phases = []
cur = []
u_end = -1
t1 = time.time()
for it in iv:
    if cur and it[0] > u_end + GAP_NS:
        phases.append(cur); cur = []
        u_end = -1
    cur.append(it)
    if u_end == -1 or it[1] > u_end:
        u_end = it[1]
if cur:
    phases.append(cur)
print(f"phases : {len(phases)}  ({time.time()-t1:.1f}s)", flush=True)

fam_t = {}
def fam(name):
    if name in fam_t:
        return fam_t[name]
    n = name
    if "flash" in n: f = "flash_attn"
    elif "fa_" in n or n.startswith("fa_"): f = "fa"
    elif "mul_mat_vec_q" in n or "mmvq" in n: f = "mmvq(mul_mat_vec_q)"
    elif "mapped_rows" in n: f = "mapped_rows"
    elif "gated_delta" in n or "gdn" in n: f = "gated_delta_net"
    elif "quantize" in n: f = "quantize"
    elif "dequantize" in n: f = "dequantize"
    elif "convert" in n: f = "convert"
    elif "concat" in n: f = "concat"
    elif "rms_norm" in n: f = "rms_norm"
    elif "rope" in n: f = "rope"
    elif "set_rows" in n or "k_set" in n: f = "kv_set_rows"
    elif "copy" in n: f = "copy"
    elif "cutlass" in n or "gemm" in n or "gemv" in n: f = "gemm/gemv(cutlass)"
    elif "moe" in n or "experts" in n: f = "moe"
    elif "softmax" in n: f = "softmax"
    elif "argmax" in n: f = "argmax/sample"
    elif "nary" in n: f = "nary"
    elif "binary" in n: f = "binary"
    elif "unary" in n: f = "unary"
    elif "embedding" in n: f = "embedding"
    else: f = "aut:" + n[:48]
    fam_t[name] = f
    return f

out = []
t2 = time.time()
for pi, ph in enumerate(phases):
    s0 = min(x[0] for x in ph)
    e0 = max(x[1] for x in ph)
    us, ov = union_stats([ (x[0], x[1]) for x in ph ])
    kers = [x for x in ph if x[2] == 0]
    cpy = [x for x in ph if x[2] == 1]
    fdur = defaultdict(float); fcnt = defaultdict(int)
    cpydur = 0.0; cpybytes = 0  # bytes in name? 'kind' absent -> juste durée
    for x in kers:
        f = fam(x[3]); fdur[f] += (x[1]-x[0]); fcnt[f] += 1
    for x in cpy:
        cpydur += (x[1]-x[0])
    wall_ms = (e0-s0)/1e6
    busy_ms = us/1e6
    occ = 100.0*us/(e0-s0)
    # top gaps
    gaps = []
    ue = -1
    for x in sorted(ph, key=lambda y: (y[0], y[1])):
        if ue >= 0 and x[0] - ue > 100000:
            gaps.append(x[0]-ue)
        ue = max(ue, x[1])
    gaps.sort(reverse=True)
    row = {
        "ph": pi, "wall_ms": round(wall_ms,1), "busy_ms": round(busy_ms,1),
        "occ_%": round(occ,1), "nker": len(kers), "ncpy": len(cpy),
        "cpy_ms": round(cpydur/1e6,1),
        "top_gaps_ms": [round(g/1e6,1) for g in gaps[:6]],
        "fams": {f: {"ms": round(v/1e6,1), "n": c} for f, v in
                 sorted(fdur.items(), key=lambda kv: -kv[1])[:N_TOP] for c in [fcnt[f]]},
    }
    out.append(row)
print(f"statistiques : {time.time()-t2:.1f}s", flush=True)

# ---- rapport ----
print()
print(f"{'ph':>3} {'wall ms':>10} {'busy ms':>10} {'occ%':>6} {'nker':>7} {'ncpy':>6} {'cpy ms':>8}  top_gaps(ms)")
for r in out:
    print(f"{r['ph']:>3} {r['wall_ms']:>10} {r['busy_ms']:>10} {r['occ_%']:>6} {r['nker']:>7} {r['ncpy']:>6} {r['cpy_ms']:>8}  {r['top_gaps_ms']}")

print()
print("=== FAMILLES par phase (top 12, ms du GPU busy) ===")
for r in out:
    if r["busy_ms"] < 5 and r["ph"] < 3:
        continue
    print(f"\n--- ph{r['ph']}  wall={r['wall_ms']}ms busy={r['busy_ms']}ms occ={r['occ_%']}% nker={r['nker']}")
    tot = sum(v["ms"] for v in r["fams"].values())
    for f, v in list(r["fams"].items())[:12]:
        print(f"   {v['ms']:>9} ms x{v['n']:<7} {100*v['ms']/max(tot,1e-9):5.1f}%  {f}")

with open(prefix + ".phases.json", "w") as f:
    json.dump(out, f, indent=1)
print(f"\nécrit : {prefix}.phases.json")
