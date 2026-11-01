#!/usr/bin/env python3
"""
t1_bench.py — A/B T1 (anyei #153 mmsmt) en MTP ON (config prod), 18/10.

Piste : mmsmt = matmul small-batch (rows 2..8) sur tensor-core, types quant
Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/IQ4_XS/Q3_K/Q2_K/Q2_0. Cible : le MTP verify
(draft n-max 3 -> rows = n+1 = 4..8, exactement le régime T1).

MTP ON est ICI volontairement la config (consigne user 18/10 : "bench T1 avec
MTP on"). Serveur en config prod exacte : -c 167000, -b 2048 -ub 2048,
--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling,
KV q8_0, FA auto, context-shift NON (pas besoin, points < ctx).

3 legs (interleave, serveur frais par leg) :
  A-t3   : .so T3 (= prod actuelle), T1 absent
  B-t1t3 : .so T1T3, T1 actif (default ON)
  C-smt0 : .so T1T3 + GGML_CUDA_SMT=0 (A/B même binaire)

B vs A = gain réel du deploy ; C vs A ≈ 0 (attendu) ; B vs C = effet pur T1.
Points : 114688 et 160000 (167000 = max config prod, 160000 + 64 gen < 167000).
"""
import json
import os
import signal
import subprocess
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(D, "night_bench.py"), "__name__": "night_bench_harness"}
src = open(os.path.join(D, "night_bench.py")).read().split('if __name__ == "__main__":')[0]

# overrides MTP ON : MAX_CTX 167000 (config prod) + flags spec dans server_args
src = src.replace("MAX_CTX = 196672", "MAX_CTX = 167000")
src = src.replace(
    '        # MTP OFF : pas de --spec-type (consigne 18/10 : MTP exclu)\n',
    '        # MTP ON : config prod exacte (18/10 — bench T1 avec MTP on)\n'
    '        "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-backend-sampling",\n'
)
exec(src, ns)

SO_T3 = "/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib/lib/libggml-cuda-t3.so"
SO_T1T3 = "/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib/lib/libggml-cuda-t1t3.so"

OUT_JSONL = os.path.join(D, "t1_bench_results.jsonl")
OUT_SUMMARY = os.path.join(D, "t1_bench_summary.md")
OUT_SERVER_LOG = os.path.join(D, "t1_bench_server.log")
OUT_GPUTS = os.path.join(D, "t1_bench_gputimes.tsv")

ns["OUT_SERVER_LOG"] = OUT_SERVER_LOG
ns["OUT_GPUTS"] = OUT_GPUTS

log = ns["log"]
PORT = ns["PORT"]
N_GEN = ns["N_GEN"]

def main():
    signal.signal(signal.SIGINT, ns["on_sig"])
    signal.signal(signal.SIGTERM, ns["on_sig"])
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    log(f"BENCH T1 (MTP ON, config prod) — reps={reps} — lock tenu par le superviseur")
    if not ns["port_free"](PORT):
        sys.exit(f"port {PORT} occupe — abandon")
    for f in (SO_T3, SO_T1T3):
        if not os.path.isfile(f):
            sys.exit(f"lib manquante : {f}")
    log(f"  · SO T3   md5 = {__import__('hashlib').md5(open(SO_T3,'rb').read()).hexdigest()[:8]}")
    log(f"  · SO T1T3 md5 = {__import__('hashlib').md5(open(SO_T1T3,'rb').read()).hexdigest()[:8]}")

    ns["stop_and_free"]()
    open(OUT_SERVER_LOG, "w").close()
    open(OUT_GPUTS, "w").close()

    sampler = subprocess.Popen(["python3", os.path.join(D, "gputimes_sampler.py"), OUT_GPUTS])

    CTX_LONG = [114688, 160000]

    EXP = [
        ("EXP-T1", "A-t3",   SO_T3,   None,      CTX_LONG),
        ("EXP-T1", "B-t1t3", SO_T1T3, None,      CTX_LONG),
        ("EXP-T1", "C-smt0", SO_T1T3, {"GGML_CUDA_SMT": "0"}, CTX_LONG),
    ]

    allrows = []
    try:
        for rep in range(1, reps + 1):
            legs = EXP if rep % 2 == 1 else list(reversed(EXP))
            for tag, name, so, extra_env, ctxs in legs:
                try:
                    log(f"=== T1 rep{rep} :: LEG {name}")
                    ns["start_server"](so, 2048, None, extra_env)
                    wu = ns["warmup"](ns["log_size"]())
                    for ctx in ctxs:
                        since = ns["log_size"]()
                        log(f"  · point {ctx} → {N_GEN} gen")
                        api, met = ns["run_point"](ctx, N_GEN, since)
                        row = {"exp": tag, "leg": name, "rep": rep, "ctx": ctx,
                               "mtp": "on", "batch": 2048, "nthreads": None,
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
    L = ["# BENCH T1 (anyei #153 mmsmt) — 18/10 — MTP ON (config prod), -b 2048, interleave", "",
         f"date : {time.strftime('%Y-%m-%d %H:%M UTC')}",
         f"reps : {reps} · service restauré : {'OUI' if ns['_restored'] else 'NON'}",
         "config : -c 167000, --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling, KV q8_0, FA auto", ""]
    for ctx in CTX_LONG:
        L.append(f"## ctx {ctx}")
        L.append("| leg | rep | decode t/s | prefill t/s |")
        L.append("|---|---|---|---|")
        for rep in range(1, reps + 1):
            for name in ("A-t3", "B-t1t3", "C-smt0"):
                rs = [r for r in allrows if r["leg"] == name and r["ctx"] == ctx and r.get("rep") == rep]
                if rs:
                    L.append(f"| {name} | {rep} | {rs[0].get('eval_tps','?')} | {rs[0].get('prompt_tps','?')} |")
        L.append("")
        vals = {}
        for name in ("A-t3", "B-t1t3", "C-smt0"):
            rs = [r for r in allrows if r["leg"] == name and r["ctx"] == ctx]
            vals[name] = [r["eval_tps"] for r in rs if r.get("eval_tps")]
        bp = med(vals["A-t3"])
        L.append("Δ decode vs A-t3 (médiane) :")
        for name in ("B-t1t3", "C-smt0"):
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
