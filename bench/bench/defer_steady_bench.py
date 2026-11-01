#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""defer_steady_bench.py — Vérification décisive : le coût DEFER_PROMPT est-IL PONCTUEL ?

Contexte (résultats 29/10 00:15-00:44 UTC, .so 8035caf4 figé, config lente b2048) :
  noflags   56.40 t/s @114k  (rapide)
  full (SL+DEFER) 16.26 t/s  (lent)   = SL 58.12 (neutre) + DEFER 14.79 (cause)
Analyse code (server-context.cpp ~L4433) : DEFER_PROMPT diffère le catch-up MTP du
prompt et le rejoue SYNCHRONEMENT au 1er token généré. Arithmétique du gap :
  gap @114k = 4328.6-1134.8 = 3194 ms sur 111713 tok nouveaux = 28.6 us/tok
  gap @160k = 2444.4-1272.2 = 1172 ms sur  46179 tok nouveaux = 25.4 us/tok
  ratio gap 2.72 vs ratio tokens 2.42 → proportionnel au prompt NOUVEAU.
Si le coût est ponctuel (1× au 1er token), la 2e requête sur le MÊME prompt
(checkpoint restauré, ~0 tok nouveaux) doit mesurer le vrai steady-state (~56 t/s).

Séquence : 1 serveur (config lente EXACTE : shortlist+DEFER+H2off, MTP ON, b2048,
c167000, mmproj) → warmup → REQ A: prompt 111713 tok, 64 gen (réf. contaminée)
→ REQ B: MÊME prompt, 128 gen (steady-state attendu) → arrêt/restore.

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run defer_steady_bench.py --detach
"""
import json
import os
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(D, "night_bench.py"), "__name__": "defer_steady_harness"}
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
FULL = {
    "GGML_CUDA_QWEN35_MTP_SHORTLIST": SL_PATH,
    "LLAMA_ARG_SPEC_MTP_DEFER_PROMPT": "1",
    "GGML_CUDA_VOLTA_Q8_FATTN_TC": "0",
}

OUT_JSONL = os.path.join(D, "defer_steady_bench_results.jsonl")
OUT_SUMMARY = os.path.join(D, "defer_steady_bench_summary.md")
OUT_SERVER_LOG = os.path.join(D, "defer_steady_bench_server.log")
OUT_GPUTS = os.path.join(D, "defer_steady_bench_gputimes.tsv")

ns["OUT_SERVER_LOG"] = OUT_SERVER_LOG
ns["OUT_GPUTS"] = OUT_GPUTS

log = ns["log"]
CTX = 114688


def main():
    import signal
    signal.signal(signal.SIGINT, ns["on_sig"])
    signal.signal(signal.SIGTERM, ns["on_sig"])
    if not ns["port_free"](ns["PORT"]):
        sys.exit("port %d occupe — abandon" % ns["PORT"])
    log("DEFER STEADY (config lente exacte, MTP ON, .so 8035caf4) — lock superviseur")
    log("  · SO prod md5 = %s" % __import__("hashlib").md5(open(SO, "rb").read()).hexdigest()[:8])

    ns["stop_and_free"]()
    open(OUT_SERVER_LOG, "w").close()
    open(OUT_GPUTS, "w").close()
    sampler = ns["subprocess"].Popen(["python3", os.path.join(D, "gputimes_sampler.py"), OUT_GPUTS])

    rows = []
    try:
        ns["start_server"](SO, 2048, None, FULL)
        ns["warmup"](ns["log_size"]())

        # REQ A : prompt 111713 tok, 64 gen (contaminé par le catch-up, ~15-16 t/s attendu)
        since = ns["log_size"]()
        log("· REQ A : prompt %d tok, 64 gen (fenêtre contaminée)" % CTX)
        apiA, metA = ns["run_point"](CTX, 64, since)
        rows.append({"req": "A-contam", "ctx": CTX, "gen": 64, "vram_mib": ns["vram_mib"](),
                     **apiA, **metA})
        log("  → prefill %s t/s | decode %s t/s | %s"
            % (metA.get("prompt_tps", "?"), metA.get("eval_tps", "?"), apiA.get("error", "ok")))
        time.sleep(3)

        # REQ B : MÊME prompt (checkpoint restauré, ~0 tok nouveaux), 128 gen → steady-state
        since = ns["log_size"]()
        log("· REQ B : MÊME prompt, 128 gen (steady-state attendu ~56 t/s)")
        apiB, metB = ns["run_point"](CTX, 128, since)
        rows.append({"req": "B-steady", "ctx": CTX, "gen": 128, "vram_mib": ns["vram_mib"](),
                     **apiB, **metB})
        log("  → prefill %s t/s (prompt %s) | decode %s t/s | %s"
            % (metB.get("prompt_tps", "?"), metB.get("prompt_tokens", "?"),
               metB.get("eval_tps", "?"), apiB.get("error", "ok")))
    except Exception as e:
        log("· ECHEC : %s" % e)
    finally:
        try:
            sampler.terminate(); sampler.wait(timeout=10)
        except Exception:
            pass
        ns["stop_server"]()
    ns["restore_service"]()

    with open(OUT_JSONL, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # extrait log : checkpoint restauré REQ B
    import re
    ckpt = [l for l in open(OUT_SERVER_LOG, errors="replace")
            if re.search(r"restored context checkpoint|cached n_tokens|eval time", l)]
    L = ["# DEFER STEADY — le coût DEFER_PROMPT est-il ponctuel ?", "",
         "date : " + time.strftime("%Y-%m-%d %H:%M UTC"),
         "config : lente EXACTE (shortlist+DEFER+H2off), MTP ON, b2048, c167000, mmproj, .so 8035caf4",
         "service restauré : %s" % ("OUI" if ns["_restored"] else "NON"), "",
         "| req | prompt tok traités | decode t/s | généré |",
         "|---|---|---|---|"]
    for r in rows:
        L.append("| %s | %s | %s | %s |"
                 % (r["req"], r.get("prompt_tokens", "?"), r.get("eval_tps", "?"),
                    r.get("eval_tokens", "?")))
    L += ["", "Lignes serveur (checkpoint/timings, extrait) :"]
    L += ["    " + l.strip()[:160] for l in ckpt[-14:]]
    if len(rows) == 2 and rows[0].get("eval_tps") and rows[1].get("eval_tps"):
        L += ["", "Lecture : REQ B >> REQ A → le coût DEFER est PONCTUEL (catch-up 1× au 1er token).",
              "REQ B ≈ 56 t/s → le steady-state MTP .so 8035caf4 est PLUS RAPIDE que le 18/10 (45.7)."]
    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(L) + "\n")
    log("FIN — synthèse : %s" % OUT_SUMMARY)
    sys.exit(0 if ns["_restored"] else 3)


if __name__ == "__main__":
    main()
