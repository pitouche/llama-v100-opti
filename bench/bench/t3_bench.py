#!/usr/bin/env python3
"""
t3_bench.py — A/B T3 (anyei #153 fattn-mma-volta-small), MTP OFF partout.

Piste : kernel attention tensor-core m8n8k4 pour shapes "small-width"
(Q rows 1..8, head 64/128/256, KV q8_0 ou f16). Cible directe : le decode
T=1 (rows=1) sur KV q8_0 — notre config prod. Gain revendiqué anyei :
+60% attention @128k.

3 legs (interleave, serveur frais par leg) :
  A-prod : .so prod H2 (a4b6f440) — référence, pas de T3
  B-t3on : .so T3, T3 actif (default ON)
  C-t3off: .so T3, GGML_CUDA_FA_NO_VOLTA_SMALL=1 (A/B même binaire)

B vs C = effet pur T3. A vs C doit être ~0 (le .so T3 ne diffère que par T3).
-b 2048 partout (config prod), KV q8_0, FA auto, CVD=0 (V100), port 8917.
Contextes : 114688 et 196608. Warmup 1k. Service stoppé en début,
restauré en finally.

USAGE :
  python3 supervisor_bench.py run t3_bench.py 2 --detach
"""
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request

D = os.path.dirname(os.path.abspath(__file__))
# réutilise le harnais de nuit (identiques : serveur, stop/restore, parsing)
ns = {"__file__": os.path.join(D, "night_bench.py"), "__name__": "night_bench_harness"}
exec(open(os.path.join(D, "night_bench.py")).read().split('if __name__ == "__main__":')[0], ns)

SO_T3 = "/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib/lib/libggml-cuda-t3.so"

OUT_JSONL = os.path.join(D, "t3_bench_results.jsonl")
OUT_SUMMARY = os.path.join(D, "t3_bench_summary.md")
OUT_SERVER_LOG = os.path.join(D, "t3_bench_server.log")
OUT_GPUTS = os.path.join(D, "t3_bench_gputimes.tsv")

# rediriger les logs du harnais vers les fichiers T3
ns["OUT_SERVER_LOG"] = OUT_SERVER_LOG
ns["OUT_GPUTS"] = OUT_GPUTS

log = ns["log"]
SO_PROD = ns["SO_PROD"]
PORT = ns["PORT"]
N_GEN = ns["N_GEN"]

def main():
    signal.signal(signal.SIGINT, ns["on_sig"])
    signal.signal(signal.SIGTERM, ns["on_sig"])
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    log(f"BENCH T3 (MTP OFF) — reps={reps} — lock tenu par le superviseur")
    if not ns["port_free"](PORT):
        sys.exit(f"port {PORT} occupe — abandon")
    for f in (SO_PROD, SO_T3):
        if not os.path.isfile(f):
            sys.exit(f"lib manquante : {f}")
    log(f"  · SO prod md5 = {__import__('hashlib').md5(open(SO_PROD,'rb').read()).hexdigest()[:8]}")
    log(f"  · SO T3   md5 = {__import__('hashlib').md5(open(SO_T3,'rb').read()).hexdigest()[:8]}")

    ns["stop_and_free"]()
    open(OUT_SERVER_LOG, "w").close()
    open(OUT_GPUTS, "w").close()

    sampler = subprocess.Popen(["python3", os.path.join(D, "gputimes_sampler.py"), OUT_GPUTS])

    CTX_LONG = [114688, 196608]

    EXP = [
        ("EXP-T3", "A-prod",  SO_PROD, None,       CTX_LONG),
        ("EXP-T3", "B-t3on",  SO_T3,    None,      CTX_LONG),
        ("EXP-T3", "C-t3off", SO_T3,    {"GGML_CUDA_FA_NO_VOLTA_SMALL": "1"}, CTX_LONG),
    ]

    allrows = []
    try:
        for rep in range(1, reps + 1):
            legs = EXP if rep % 2 == 1 else list(reversed(EXP))
            for tag, name, so, extra_env, ctxs in legs:
                try:
                    log(f"=== T3 rep{rep} :: LEG {name}")
                    ns["start_server"](so, 2048, None, extra_env)
                    wu = ns["warmup"](ns["log_size"]())
                    for ctx in ctxs:
                        since = ns["log_size"]()
                        log(f"  · point {ctx} → {N_GEN} gen")
                        api, met = ns["run_point"](ctx, N_GEN, since)
                        row = {"exp": tag, "leg": name, "rep": rep, "ctx": ctx,
                               "mtp": "off", "batch": 2048, "nthreads": None,
                               "vram_mib": ns["vram_mib"](),
                               **api, **met}
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
    L = ["# BENCH T3 (anyei #153) — 18/10 — MTP OFF, -b 2048, interleave", "",
         f"date : {time.strftime('%Y-%m-%d %H:%M UTC')}",
         f"reps : {reps} · service restauré : {'OUI' if ns['_restored'] else 'NON'}", ""]
    for ctx in CTX_LONG:
        L.append(f"## ctx {ctx}")
        L.append("| leg | rep | decode t/s | prefill t/s |")
        L.append("|---|---|---|---|")
        for rep in range(1, reps + 1):
            for name in ("A-prod", "B-t3on", "C-t3off"):
                rs = [r for r in allrows if r["leg"] == name and r["ctx"] == ctx and r.get("rep") == rep]
                if rs:
                    L.append(f"| {name} | {rep} | {rs[0].get('eval_tps','?')} | {rs[0].get('prompt_tps','?')} |")
        L.append("")
        vals = {}
        for name in ("A-prod", "B-t3on", "C-t3off"):
            rs = [r for r in allrows if r["leg"] == name and r["ctx"] == ctx]
            vals[name] = [r["eval_tps"] for r in rs if r.get("eval_tps")]
        bp = med(vals["A-prod"])
        L.append("Δ decode vs A-prod (médiane) :")
        for name in ("B-t3on", "C-t3off"):
            v = med(vals[name])
            if bp and v:
                L.append(f"  - {name}: {(v-bp)/bp*100:+.2f}%")
        L.append("")
    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(L) + "\n")
    log(f"FIN — synthèse : {OUT_SUMMARY}")
    sys.exit(0 if ns["_restored"] else 3)

if __name__ == "__main__":
    main()
