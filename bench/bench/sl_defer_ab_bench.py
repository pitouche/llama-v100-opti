#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sl_defer_ab_bench.py — A/B shortlist/DEFER_PROMPT, MTP ON, config lente, .so 8035caf4 figé.

Suite de h2_ab_bench.py (H2 éliminé : Δ ≈ 0 à ±6%). Isole la prochaine variable :
  A-full    : config lente exacte (shortlist 131072 + DEFER_PROMPT=1 + H2 OFF)  → doit refaire ~15.7
  B-noflags : même config SANS shortlist SANS DEFER_PROMPT (H2 OFF)
Si B remonte → le couple shortlist/DEFER est la cause dans le .so 8035caf4.
Si B reste lent → c'est le .so 8035caf4 lui-même (vs d2df07b8 du 18/10).

Aucun swap de .so. Serveur frais par leg, interleave, points 114688 + 160000.

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run sl_defer_ab_bench.py [reps] --detach
"""
import json
import os
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(D, "night_bench.py"), "__name__": "sl_defer_harness"}
src = open(os.path.join(D, "night_bench.py")).read().split('if __name__ == "__main__":')[0]

src = src.replace("MAX_CTX = 196672", "MAX_CTX = 167000")
src = src.replace(
    '        # MTP OFF : pas de --spec-type (consigne 18/10 : MTP exclu)\n',
    '        # MTP ON : config lente Step A\n'
    '        "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-backend-sampling",\n'
)
exec(src, ns)

SO = ns["SO_PROD"]  # 8035caf4

OUT_JSONL = os.path.join(D, "sl_defer_ab_bench_results.jsonl")
OUT_SUMMARY = os.path.join(D, "sl_defer_ab_bench_summary.md")
OUT_SERVER_LOG = os.path.join(D, "sl_defer_ab_bench_server.log")
OUT_GPUTS = os.path.join(D, "sl_defer_ab_bench_gputimes.tsv")

ns["OUT_SERVER_LOG"] = OUT_SERVER_LOG
ns["OUT_GPUTS"] = OUT_GPUTS

log = ns["log"]
N_GEN = ns["N_GEN"]

FULL = {
    "GGML_CUDA_QWEN35_MTP_SHORTLIST":
        "/home/guile/llama-experiments/src/llama-volta-merged/data/mtp-shortlists/qwen38-27b-exact-131072.i32",
    "LLAMA_ARG_SPEC_MTP_DEFER_PROMPT": "1",
    "GGML_CUDA_VOLTA_Q8_FATTN_TC": "0",
}
NOFLAGS = {"GGML_CUDA_VOLTA_Q8_FATTN_TC": "0"}  # pas de shortlist, pas de DEFER


def main():
    import signal
    signal.signal(signal.SIGINT, ns["on_sig"])
    signal.signal(signal.SIGTERM, ns["on_sig"])
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    log("SL/DEFER A/B (config lente, MTP ON, .so 8035caf4) — reps=%d — lock tenu par le superviseur" % reps)
    if not ns["port_free"](ns["PORT"]):
        sys.exit("port %d occupe — abandon" % ns["PORT"])
    log("  · SO prod md5 = %s" % __import__("hashlib").md5(open(SO, "rb").read()).hexdigest()[:8])

    ns["stop_and_free"]()
    open(OUT_SERVER_LOG, "w").close()
    open(OUT_GPUTS, "w").close()
    sampler = ns["subprocess"].Popen(["python3", os.path.join(D, "gputimes_sampler.py"), OUT_GPUTS])

    CTX = [114688, 160000]
    LEGS = [
        ("A-full", FULL),
        ("B-noflags", NOFLAGS),
    ]

    allrows = []
    try:
        for rep in range(1, reps + 1):
            legs = LEGS if rep % 2 == 1 else list(reversed(LEGS))
            for name, extra in legs:
                try:
                    log("=== SLDEFER rep%d :: LEG %s" % (rep, name))
                    ns["start_server"](SO, 2048, None, extra)
                    ns["warmup"](ns["log_size"]())
                    for ctx in CTX:
                        since = ns["log_size"]()
                        log("  · point %d → %d gen" % (ctx, N_GEN))
                        api, met = ns["run_point"](ctx, N_GEN, since)
                        row = {"leg": name, "rep": rep, "ctx": ctx, "mtp": "on",
                               "batch": 2048, "vram_mib": ns["vram_mib"](),
                               **api, **met}
                        allrows.append(row)
                        log("    → prefill %s t/s | decode %s t/s | %s"
                            % (met.get("prompt_tps", "?"), met.get("eval_tps", "?"),
                               api.get("error", "ok")))
                except Exception as e:
                    log("  · %s rep%d ECHEC : %s" % (name, rep, e))
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

    L = ["# SL/DEFER A/B — config lente, MTP ON, .so 8035caf4 figé", "",
         "date : " + time.strftime("%Y-%m-%d %H:%M UTC"),
         "legs : A-full (shortlist+DEFER+H2off) vs B-noflags (sans shortlist/DEFER, H2off)",
         "config : -c 167000, -b 2048 -ub 2048, --mmproj, MTP ON (n-max 3), KV q8_0, FA auto,",
         "         W4R4=1, FIT 1024, CVD=0",
         "service restauré : %s" % ("OUI" if ns["_restored"] else "NON"), ""]
    med = ns["med"]
    for ctx in CTX:
        L.append("## ctx %d" % ctx)
        L.append("| leg | rep | decode t/s | prefill t/s | VRAM |")
        L.append("|---|---|---|---|---|")
        for rep in range(1, reps + 1):
            for name, _ in LEGS:
                rs = [r for r in allrows if r["leg"] == name and r["ctx"] == ctx and r["rep"] == rep]
                if rs:
                    L.append("| %s | %d | %s | %s | %s |"
                             % (name, rep, rs[0].get("eval_tps", "?"),
                                rs[0].get("prompt_tps", "?"), rs[0].get("vram_mib", "?")))
        L.append("")
        vals = {}
        for name, _ in LEGS:
            vals[name] = [r["eval_tps"] for r in allrows if r["leg"] == name and r["ctx"] == ctx
                          if r.get("eval_tps")]
        a, b = med(vals["A-full"]), med(vals["B-noflags"])
        if a and b:
            L.append("Δ decode B-noflags vs A-full : **%+.1f%%** (%.2f → %.2f t/s)"
                     % ((b - a) / a * 100, a, b))
        L.append("")
    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(L) + "\n")
    log("FIN — synthèse : %s" % OUT_SUMMARY)
    sys.exit(0 if ns["_restored"] else 3)


if __name__ == "__main__":
    main()
