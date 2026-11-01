#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sl_vs_defer_bench.py — Découpe shortlist vs DEFER_PROMPT, config lente batch 2048.

Suite de sl_defer_ab_bench.py (shortlist+DEFER = cause, 3.4x, acceptation identique).
Isole quel flag casse le verify (batch 2048, mmproj, ctx 114688, .so 8035caf4, H2 off):
  B-slonly   : shortlist 131072 UNIQUEMENT (pas de DEFER)
  C-deferonly: DEFER_PROMPT=1 UNIQUEMENT (pas de shortlist)
Contrôles déjà mesurés (sl_defer_ab) : full=16.26 (lent), noflags=56.40 (rapide).

Aucun swap de .so. Serveur frais par leg, point 114688 (+160000 si temps).

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run sl_vs_defer_bench.py [reps] --detach
"""
import json
import os
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(D, "night_bench.py"), "__name__": "sl_vs_defer_harness"}
src = open(os.path.join(D, "night_bench.py")).read().split('if __name__ == "__main__":')[0]

src = src.replace("MAX_CTX = 196672", "MAX_CTX = 167000")
src = src.replace(
    '        # MTP OFF : pas de --spec-type (consigne 18/10 : MTP exclu)\n',
    '        # MTP ON : config lente Step A\n'
    '        "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-backend-sampling",\n'
)
exec(src, ns)

SO = ns["SO_PROD"]  # 8035caf4

SL_PATH = "/home/guile/llama-experiments/src/llama-volta-merged/data/mtp-shortlists/qwen38-27b-exact-131072.i32"

OUT_JSONL = os.path.join(D, "sl_vs_defer_bench_results.jsonl")
OUT_SUMMARY = os.path.join(D, "sl_vs_defer_bench_summary.md")
OUT_SERVER_LOG = os.path.join(D, "sl_vs_defer_bench_server.log")
OUT_GPUTS = os.path.join(D, "sl_vs_defer_bench_gputimes.tsv")

ns["OUT_SERVER_LOG"] = OUT_SERVER_LOG
ns["OUT_GPUTS"] = OUT_GPUTS

log = ns["log"]
N_GEN = ns["N_GEN"]

BASE = {"GGML_CUDA_VOLTA_Q8_FATTN_TC": "0"}  # H2 off (innocent, figé pour reproductibilité)
SLONLY = dict(BASE, **{"GGML_CUDA_QWEN35_MTP_SHORTLIST": SL_PATH})
DEFERONLY = dict(BASE, **{"LLAMA_ARG_SPEC_MTP_DEFER_PROMPT": "1"})

LEGS = [
    ("B-slonly", SLONLY),
    ("C-deferonly", DEFERONLY),
]


def main():
    import signal
    signal.signal(signal.SIGINT, ns["on_sig"])
    signal.signal(signal.SIGTERM, ns["on_sig"])
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    log("SL-vs-DEFER (config lente b2048, MTP ON, .so 8035caf4) — reps=%d — lock superviseur" % reps)
    if not ns["port_free"](ns["PORT"]):
        sys.exit("port %d occupe — abandon" % ns["PORT"])
    log("  · SO prod md5 = %s" % __import__("hashlib").md5(open(SO, "rb").read()).hexdigest()[:8])

    ns["stop_and_free"]()
    open(OUT_SERVER_LOG, "w").close()
    open(OUT_GPUTS, "w").close()
    sampler = ns["subprocess"].Popen(["python3", os.path.join(D, "gputimes_sampler.py"), OUT_GPUTS])

    CTX = [114688, 160000]
    allrows = []
    try:
        for rep in range(1, reps + 1):
            legs = LEGS if rep % 2 == 1 else list(reversed(LEGS))
            for name, extra in legs:
                try:
                    log("=== SLVSDEFER rep%d :: LEG %s" % (rep, name))
                    ns["start_server"](SO, 2048, None, extra)
                    ns["warmup"](ns["log_size"]())
                    for ctx in CTX:
                        since = ns["log_size"]()
                        log("  · point %d → %d gen" % (ctx, N_GEN))
                        api, met = ns["run_point"](ctx, N_GEN, since)
                        row = {"leg": name, "rep": rep, "ctx": ctx, "mtp": "on",
                               "batch": 2048, "vram_mib": ns["vram_mib"](), **api, **met}
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

    med = ns["med"]
    L = ["# SL vs DEFER — config lente b2048, MTP ON, .so 8035caf4", "",
         "date : " + time.strftime("%Y-%m-%d %H:%M UTC"),
         "legs : B-slonly (shortlist seul) vs C-deferonly (DEFER seul)",
         "contrôles (sl_defer_ab) : full=16.26 (lent) | noflags=56.40 (rapide) @114k",
         "service restauré : %s" % ("OUI" if ns["_restored"] else "NON"), ""]
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
        for name, _ in LEGS:
            vals = [r["eval_tps"] for r in allrows if r["leg"] == name and r["ctx"] == ctx if r.get("eval_tps")]
            m = med(vals)
            if m:
                L.append("  - %s : %.2f t/s (vs full 16.26 / noflags 56.40)" % (name, m))
        L.append("")
    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(L) + "\n")
    log("FIN — synthèse : %s" % OUT_SUMMARY)
    sys.exit(0 if ns["_restored"] else 3)


if __name__ == "__main__":
    main()
