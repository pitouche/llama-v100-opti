#!/usr/bin/env python3
"""Depouillement night_bench_results.jsonl (labels exp/leg fixes)."""
import json

rows = [json.loads(l) for l in open("night_bench_results.jsonl")]
NAMES = {
  "EXP1-H3": ["A-prodH2", "B-H3-42d"],
  "EXP2-BATCH": ["A-b2048", "B-b4096", "C-b8192"],
  "EXP3-NT": ["A-t6", "B-t12"],
}
for r in rows:
    if r["leg"] in NAMES:
        r["leg"], r["exp"] = r["exp"], r["leg"]

def med(v):
    v = sorted(x for x in v if x)
    return v[len(v)//2] if v else None

def g(exp, leg, ctx=None, key="eval_tps"):
    return med([r[key] for r in rows if r["exp"]==exp and r["leg"]==leg
                and (ctx is None or r["ctx"]==ctx) and r.get(key)])

L = ["# DEPouilLEMENT NIGHT BENCH (fix du summary — donnees brutes identiques)", "",
     "22 points, 2 reps, MTP OFF, -np 1, KV q8_0, FA auto, .so H2 prod / H3 (42d9581).", ""]
for tag, legs in NAMES.items():
    ctxs = sorted({r["ctx"] for r in rows if r["exp"]==tag})
    L.append(f"## {tag}")
    L.append("| ctx | " + " | ".join(legs) + " |")
    L.append("|---|" + "---|"*len(legs))
    for ctx in ctxs:
        L.append("| %d |" % ctx + "".join(" %s |" % (g(tag,n,ctx) or "-") for n in legs))
    L.append("")
    L.append("| ctx | " + " | ".join(legs) + " |  (prefill t/s)")
    L.append("|---|" + "---|"*len(legs))
    for ctx in ctxs:
        L.append("| %d |" % ctx + "".join(" %s |" % (g(tag,n,ctx,"prompt_tps") or "-") for n in legs))
    L.append("")
    base = legs[0]
    for ctx in ctxs:
        b = g(tag, base, ctx)
        ds = []
        for n in legs[1:]:
            v = g(tag, n, ctx)
            if b and v: ds.append("%s: %+.1f%%" % (n, (v-b)/b*100))
        L.append("decode @ %d vs %s: %s" % (ctx, base, "; ".join(ds) if ds else "n/a"))
    L.append("")

L.append("## VRAM pic par leg (MiB)")
seen = {}
for r in rows:
    if r.get("vram_mib"):
        seen[r["leg"]] = max(seen.get(r["leg"], 0), r["vram_mib"])
for k, v in sorted(seen.items()):
    L.append("- %s: %d" % (k, v))

open("night_bench_depeouille.md","w").write("\n".join(L)+"\n")
print("\n".join(L))
