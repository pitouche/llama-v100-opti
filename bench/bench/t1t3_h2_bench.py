#!/usr/bin/env python3
"""
t1t3_h2_bench.py — A/B H2 vs T3 sur la geometrie rows==4 (18/10), MTP ON (config prod).

Question : maintenant que T3 couvre rows 1-8, H2 (VOLTA_Q8_W4, rows==4 only) est-il
encore utile ? En dispatch H2 est teste AVANT T3 -> sur rows==4 (MTP verify, 3 drafts
acceptes, n-max 3 prod) c'est H2 qui a la priorite. A vs B isole l'effet pur de H2
(meme binaire T1T3, seul le gate H2 change).

2 legs (interleave, serveur frais par leg), meme binaire SO_T1T3 :
  A-h2on  : T1 on (defaut), T3 on (defaut), H2 ON  (defaut)  -> rows==4 -> H2
  B-h2off : T1 on (defaut), T3 on (defaut), H2 OFF (GGML_CUDA_VOLTA_Q8_FATTN_TC=0)
            -> rows 1-4 -> T3

A vs B = gain/re regression de H2 sur rows==4. Tout le reste (T1, T3, binaire) est
identique -> le delta est attribuable a H2 uniquement.

Config prod exacte : -c 167000, -b 2048 -ub 2048, --spec-type draft-mtp
--spec-draft-n-max 3 --spec-draft-backend-sampling, KV q8_0, FA auto, MTP ON.
Points : 114688 et 160000.
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
    '        # MTP ON : config prod exacte (bench H2)\n'
    '        "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-backend-sampling",\n'
)
exec(src, ns)

SO_T1T3 = "/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib/lib/libggml-cuda-t1t3.so"

OUT_JSONL = os.path.join(D, "t1t3_h2_bench_results.jsonl")
OUT_SUMMARY = os.path.join(D, "t1t3_h2_bench_summary.md")
OUT_SERVER_LOG = os.path.join(D, "t1t3_h2_bench_server.log")
OUT_GPUTS = os.path.join(D, "t1t3_h2_bench_gputimes.tsv")

ns["OUT_SERVER_LOG"] = OUT_SERVER_LOG
ns["OUT_GPUTS"] = OUT_GPUTS

log = ns["log"]
PORT = ns["PORT"]
N_GEN = ns["N_GEN"]

def main():
    signal.signal(signal.SIGINT, ns["on_sig"])
    signal.signal(signal.SIGTERM, ns["on_sig"])
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    log(f"BENCH H2 (T1T3 avec/sans H2) MTP ON — reps={reps} — lock superviseur")
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

    # meme binaire, seul le gate H2 change (GGML_CUDA_VOLTA_Q8_FATTN_TC)
    EXP = [
        ("EXP-H2", "A-h2on",  SO_T1T3, None,                        CTX_LONG),
        ("EXP-H2", "B-h2off", SO_T1T3, {"GGML_CUDA_VOLTA_Q8_FATTN_TC": "0"}, CTX_LONG),
    ]

    allrows = []
    try:
        for rep in range(1, reps + 1):
            legs = EXP if rep % 2 == 1 else list(reversed(EXP))
            for tag, name, so, extra_env, ctxs in legs:
                try:
                    log(f"=== H2 rep{rep} :: LEG {name} (env={extra_env})")
                    ns["start_server"](so, 2048, None, extra_env)
                    ns["warmup"](ns["log_size"]())
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
    L = ["# BENCH H2 (T1T3 avec/sans H2) — 18/10 — MTP ON (config prod), -b 2048, interleave", "",
         f"date : {time.strftime('%Y-%m-%d %H:%M UTC')}",
         f"reps : {reps} · service restauré : {'OUI' if ns['_restored'] else 'NON'}",
         "config : -c 167000, --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling, KV q8_0, FA auto",
         "binaire : SO_T1T3 (T1 on, T3 on) — seul le gate H2 GGML_CUDA_VOLTA_Q8_FATTN_TC change", ""]
    for ctx in CTX_LONG:
        L.append(f"## ctx {ctx}")
        L.append("| leg | rep | decode t/s | prefill t/s |")
        L.append("|---|---|---|---|")
        for rep in range(1, reps + 1):
            for name in ("A-h2on", "B-h2off"):
                rs = [r for r in allrows if r["leg"] == name and r["ctx"] == ctx and r.get("rep") == rep]
                if rs:
                    L.append(f"| {name} | {rep} | {rs[0].get('eval_tps','?')} | {rs[0].get('prompt_tps','?')} |")
        L.append("")
        vals = {}
        for name in ("A-h2on", "B-h2off"):
            rs = [r for r in allrows if r["leg"] == name and r["ctx"] == ctx]
            vals[name] = [r["eval_tps"] for r in rs if r.get("eval_tps")]
        ab = med(vals["A-h2on"]); bb = med(vals["B-h2off"])
        L.append("Δ decode vs A-h2on (médiane) :")
        if ab and bb:
            L.append(f"  - B-h2off: {(bb-ab)/ab*100:+.2f}%  (= effet pur de H2 : positif => H2 utile)")
        L.append("")
    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(L) + "\n")
    log(f"FIN — synthèse : {OUT_SUMMARY}")
    sys.exit(0 if ns["_restored"] else 3)

if __name__ == "__main__":
    main()
