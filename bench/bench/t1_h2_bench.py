#!/usr/bin/env python3
"""
t1_h2_bench.py — A/B H2 (on/off) dans le .so T1T3, MTP ON (config prod), 18/10.

Question posée : T1T3 avec H2 vs T1T3 SANS H2 (MTP ON).
T1 et T3 restent ACTIFS dans les deux legs (c'est "T1T3") ; seule l'env H2 varie.

Le dispatch fattn teste H2 (VOLTA_Q8_W4, rows==4, head256, KV q8_0) AVANT T3
(VOLTA_SMALL, rows 1-8). Donc :
  - H2 ON  : rows==4 -> H2 kernel ; rows 1,2,3,5,6,7,8 -> T3 kernel
  - H2 OFF : rows 1-8 -> T3 kernel (H2 désactivé)
A vs B isole l'effet de H2 au-dessus de T3 (est-ce que H2 rows==4 vaut encore
quelque chose maintenant que T3 couvre rows 1-8 ?).

Même .so (libggml-cuda-t1t3.so, md5 04f41b70), A/B par env, pas de rebuild :
  A-t1t3-h2   : T1T3, H2 ON (défaut), T3 ON, T1 ON      -> pas d'env extra
  B-t1t3-noh2 : T1T3, H2 OFF (GGML_CUDA_VOLTA_Q8_FATTN_TC=0), T3 ON, T1 ON

Config prod exacte (MTP ON) : -c 167000, -b 2048 -ub 2048,
--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling,
KV q8_0, FA auto. Points : 114688 et 160000.
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
src = src.replace("MAX_CTX = 196672", "MAX_CTX = 167000")
src = src.replace(
    '        # MTP OFF : pas de --spec-type (consigne 18/10 : MTP exclu)\n',
    '        # MTP ON : config prod exacte (18/10 — bench H2 on/off T1T3)\n'
    '        "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-backend-sampling",\n'
)
exec(src, ns)

SO_T1T3 = "/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib/lib/libggml-cuda-t1t3.so"

OUT_JSONL = os.path.join(D, "t1_h2_bench_results.jsonl")
OUT_SUMMARY = os.path.join(D, "t1_h2_bench_summary.md")
OUT_SERVER_LOG = os.path.join(D, "t1_h2_bench_server.log")
OUT_GPUTS = os.path.join(D, "t1_h2_bench_gputimes.tsv")

ns["OUT_SERVER_LOG"] = OUT_SERVER_LOG
ns["OUT_GPUTS"] = OUT_GPUTS

log = ns["log"]
PORT = ns["PORT"]
N_GEN = ns["N_GEN"]

def main():
    signal.signal(signal.SIGINT, ns["on_sig"])
    signal.signal(signal.SIGTERM, ns["on_sig"])
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    log(f"BENCH H2 on/off dans T1T3 (MTP ON) — reps={reps} — lock tenu par le superviseur")
    if not ns["port_free"](PORT):
        sys.exit(f"port {PORT} occupe — abandon")
    if not os.path.isfile(SO_T1T3):
        sys.exit(f"lib manquante : {SO_T1T3}")
    log(f"  · SO T1T3 md5 = {__import__('hashlib').md5(open(SO_T1T3,'rb').read()).hexdigest()[:8]}")

    ns["stop_and_free"]()
    open(OUT_SERVER_LOG, "w").close()
    open(OUT_GPUTS, "w").close()

    sampler = subprocess.Popen(["python3", os.path.join(D, "gputimes_sampler.py"), OUT_GPUTS])

    CTX_LONG = [114688, 160000]

    # même .so, seule l'env H2 varie (T3+T1 actifs dans les deux)
    EXP = [
        ("EXP-T1H2", "A-t1t3-h2",   SO_T1T3, None,                                    CTX_LONG),
        ("EXP-T1H2", "B-t1t3-noh2", SO_T1T3, {"GGML_CUDA_VOLTA_Q8_FATTN_TC": "0"},    CTX_LONG),
    ]

    allrows = []
    try:
        for rep in range(1, reps + 1):
            legs = EXP if rep % 2 == 1 else list(reversed(EXP))
            for tag, name, so, extra_env, ctxs in legs:
                try:
                    log(f"=== H2 rep{rep} :: LEG {name} (env {extra_env})")
                    ns["start_server"](so, 2048, None, extra_env)
                    wu = ns["warmup"](ns["log_size"]())
                    for ctx in ctxs:
                        since = ns["log_size"]()
                        log(f"  · point {ctx} → {N_GEN} gen")
                        api, met = ns["run_point"](ctx, N_GEN, since)
                        row = {"exp": tag, "leg": name, "rep": rep, "ctx": ctx,
                               "mtp": "on", "batch": 2048, "nthreads": None,
                               "h2": (extra_env or {}).get("GGML_CUDA_VOLTA_Q8_FATTN_TC", "1"),
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
    L = ["# BENCH H2 on/off dans T1T3 (anyei #153) — 18/10 — MTP ON (config prod), -b 2048, interleave", "",
         f"date : {time.strftime('%Y-%m-%d %H:%M UTC')}",
         f"reps : {reps} · service restauré : {'OUI' if ns['_restored'] else 'NON'}",
         "config : -c 167000, --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling, KV q8_0, FA auto",
         "meme .so T1T3 ; A = H2 ON (defaut), B = H2 OFF (GGML_CUDA_VOLTA_Q8_FATTN_TC=0) ; T3+T1 actifs partout", ""]
    for ctx in CTX_LONG:
        L.append(f"## ctx {ctx}")
        L.append("| leg | rep | decode t/s | prefill t/s |")
        L.append("|---|---|---|---|")
        for rep in range(1, reps + 1):
            for name in ("A-t1t3-h2", "B-t1t3-noh2"):
                rs = [r for r in allrows if r["leg"] == name and r["ctx"] == ctx and r.get("rep") == rep]
                if rs:
                    L.append(f"| {name} | {rep} | {rs[0].get('eval_tps','?')} | {rs[0].get('prompt_tps','?')} |")
        L.append("")
        vals = {}
        for name in ("A-t1t3-h2", "B-t1t3-noh2"):
            rs = [r for r in allrows if r["leg"] == name and r["ctx"] == ctx]
            vals[name] = [r["eval_tps"] for r in rs if r.get("eval_tps")]
        bp = med(vals["A-t1t3-h2"])
        L.append("Δ decode vs A-t1t3-h2 (médiane) :")
        for name in ("B-t1t3-noh2",):
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
