#!/usr/bin/env python3
"""
t3_vldg_bench.py — A/B e2e T3 vldg (double-buffer V, GO microbench) vs prod, MTP OFF.

Gold standard T3 (OPTI §5.1) : interleave 2 reps, serveur FRAIS par leg, port 8917,
CVD=0 (V100), -b 2048 (config prod), KV q8_0, FA auto, N_GEN 64, warmup 1024.
MTP OFF partout (pas de --spec-type) → le kernel attention FA = 100 % du decode,
même régime que le microbench vldg (n_kv≈107k).

2 legs :
  A-prod : .so prod ACTUEL (4fd152f9 = T3 + levier B + levier C) — référence
  B-vldg : .so vldg (87f85442 = T3 réécrit double-buffer V, B+C intacts, bit-exact)

Seule différence entre les deux .so = le kernel T3 (le microbench a prouvé le diff
SASS = 48 instr sur 3792, réordonnancement local, 0 instr ajoutée ; levier B REG:128
et levier C 17 LDG.E.128.CONSTANT intacts). Donc **B vs A = le gain e2e de vldg**.
Attente : decode B > A (croissant avec le ctx), prefill B ≈ A (le T3 ne touche que
le decode, pas le prefill qui est b2048).

Contextes : 114 688 (≈ microbench n_kv≈107k) et 196 608 (longueur max, gain croissant).

USAGE :
  python3 supervisor_bench.py run t3_vldg_bench.py 2 --detach
"""
import json
import os
import signal
import subprocess
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(D, "night_bench.py"), "__name__": "t3_vldg_harness"}
exec(open(os.path.join(D, "night_bench.py")).read().split('if __name__ == "__main__":')[0], ns)

SO_VLDG = "/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib/lib/libggml-cuda-vldg.so"

OUT_JSONL = os.path.join(D, "t3_vldg_bench_results.jsonl")
OUT_SUMMARY = os.path.join(D, "t3_vldg_bench_summary.md")
OUT_SERVER_LOG = os.path.join(D, "t3_vldg_bench_server.log")
OUT_GPUTS = os.path.join(D, "t3_vldg_bench_gputimes.tsv")

ns["OUT_SERVER_LOG"] = OUT_SERVER_LOG
ns["OUT_GPUTS"] = OUT_GPUTS

log = ns["log"]
SO_PROD = ns["SO_PROD"]
PORT = ns["PORT"]
N_GEN = ns["N_GEN"]


def md5s(p):
    import hashlib
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:8]


def main():
    signal.signal(signal.SIGINT, ns["on_sig"])
    signal.signal(signal.SIGTERM, ns["on_sig"])
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    log(f"BENCH T3 VLDG (MTP OFF) — reps={reps} — lock tenu par le superviseur")
    if not ns["port_free"](PORT):
        sys.exit(f"port {PORT} occupé — abandon")
    for f in (SO_PROD, SO_VLDG):
        if not os.path.isfile(f):
            sys.exit(f"lib manquante : {f}")
    pm, vm = md5s(SO_PROD), md5s(SO_VLDG)
    log(f"  · SO prod md5 = {pm} (attendu 4fd152f9)")
    log(f"  · SO vldg md5 = {vm} (attendu 87f85442)")
    if pm == vm:
        sys.exit("ABORT : prod == vldg — rien à mesurer")

    ns["stop_and_free"]()
    open(OUT_SERVER_LOG, "w").close()
    open(OUT_GPUTS, "w").close()

    sampler = subprocess.Popen(["python3", os.path.join(D, "gputimes_sampler.py"), OUT_GPUTS])

    CTX_LONG = [114688, 196608]

    EXP = [
        ("EXP-VLDG", "A-prod", SO_PROD, None,               CTX_LONG),
        ("EXP-VLDG", "B-vldg", SO_VLDG, None,               CTX_LONG),
    ]

    allrows = []
    try:
        for rep in range(1, reps + 1):
            legs = EXP if rep % 2 == 1 else list(reversed(EXP))
            for tag, name, so, extra_env, ctxs in legs:
                try:
                    log(f"=== VLDG rep{rep} :: LEG {name} (md5 {md5s(so)})")
                    ns["start_server"](so, 2048, None, extra_env)
                    ns["warmup"](ns["log_size"]())
                    for ctx in ctxs:
                        since = ns["log_size"]()
                        log(f"  · point {ctx} → {N_GEN} gen")
                        api, met = ns["run_point"](ctx, N_GEN, since)
                        row = {"exp": tag, "leg": name, "rep": rep, "ctx": ctx,
                               "mtp": "off", "batch": 2048, "nthreads": None,
                               "vram_mib": ns["vram_mib"](), **api, **met}
                        allrows.append(row)
                        log(f"    → prefill {met.get('prompt_tps','?')} t/s | "
                            f"decode {met.get('eval_tps','?')} t/s | {api.get('error','ok')}")
                except Exception as e:
                    log(f"  · {name} rep{rep} ECHEC : {e}")
                finally:
                    ns["stop_server"]()
                    time.sleep(12)
    finally:
        try:
            sampler.terminate(); sampler.wait(timeout=10)
        except Exception:
            pass
    ns["restore_service"]()

    with open(OUT_JSONL, "a") as f:
        for r in allrows:
            f.write(json.dumps(r) + "\n")

    med = ns["med"]
    L = ["# BENCH T3 VLDG (double-buffer V, GO microbench) — 28/10 — MTP OFF, -b 2048, interleave", "",
         f"date : {time.strftime('%Y-%m-%d %H:%M UTC')}",
         f"reps : {reps} · service restauré : {'OUI' if ns['_restored'] else 'NON'}",
         f"SO prod = {pm} · SO vldg = {vm}", "",
         "Seule diff entre les .so = le kernel T3. B vs A = gain e2e vldg. "
         "Attente : decode B > A (croissant avec le ctx), prefill B ≈ A.", ""]
    for ctx in CTX_LONG:
        L.append(f"## ctx {ctx}")
        L.append("| leg | rep | decode t/s | prefill t/s |")
        L.append("|---|---|---|---|")
        for rep in range(1, reps + 1):
            for name in ("A-prod", "B-vldg"):
                rs = [r for r in allrows if r["leg"] == name and r["ctx"] == ctx and r.get("rep") == rep]
                if rs:
                    L.append(f"| {name} | {rep} | {rs[0].get('eval_tps','?')} | {rs[0].get('prompt_tps','?')} |")
        L.append("")
        vals = {}
        for name in ("A-prod", "B-vldg"):
            rs = [r for r in allrows if r["leg"] == name and r["ctx"] == ctx]
            vals[name] = [r["eval_tps"] for r in rs if r.get("eval_tps")]
        bpr = [r["prompt_tps"] for r in allrows if r["leg"] == "A-prod" and r["ctx"] == ctx and r.get("prompt_tps")]
        vpr = [r["prompt_tps"] for r in allrows if r["leg"] == "B-vldg" and r["ctx"] == ctx and r.get("prompt_tps")]
        bp, vp = med(vals["A-prod"]), med(vals["B-vldg"])
        if bp and vp:
            L.append(f"Δ decode vs A-prod (médiane) : **{(vp-bp)/bp*100:+.2f}%**  "
                     f"(A={bp} t/s → B={vp} t/s)")
        bpm, vpm = med(bpr), med(vpr)
        if bpm and vpm:
            L.append(f"Δ prefill vs A-prod (médiane) : {(vpm-bpm)/bpm*100:+.2f}%  "
                     f"(A={bpm} t/s → B={vpm} t/s) — doit rester ~0 (T3 ne touche que le decode)")
        L.append("")

    # VRAM pic
    L.append("## VRAM pic par leg (MiB)")
    L.append("| leg | max VRAM |")
    L.append("|---|---|")
    seen = {}
    for r in allrows:
        if r.get("vram_mib"):
            seen[r["leg"]] = max(seen.get(r["leg"], 0), r["vram_mib"])
    for k, v in sorted(seen.items()):
        L.append(f"| {k} | {v} |")

    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(L) + "\n")
    log(f"FIN — synthèse : {OUT_SUMMARY}")
    sys.exit(0 if ns["_restored"] else 3)


if __name__ == "__main__":
    main()
