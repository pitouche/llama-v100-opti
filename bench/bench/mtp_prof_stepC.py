#!/usr/bin/env python3
"""
mtp_prof_stepC.py — Étape C : OÙ PASSE LE TEMPS MTP (steady-state, harnais corrigé).

Suite directe du diagnostic 29/10 (mtp_defer_diagnostic_20261029.md). Le harnais v1
mesurait la 1re fenêtre de decode après prompt neuf = catch-up DEFER (artefact). Ici
run_point2 (night_bench corrigé) envoie 2 requêtes sur le MÊME prompt :
  W1 = 64 gen  (prompt neuf → fenêtre contaminée, coût de catch-up)
  W2 = 128 gen (même prompt checkpointé → vrai steady-state)
+ spec_delta par fenêtre (drafts, acc, t_draft/t_accept) via les stats cumulées.

Décomposition cible (steady W2, MTP ON n_max=3) :
  t_cycle = eval_ms / n_drafts
  t_draft = t_draft_delta / n_drafts          (génération du draft, mesuré)
  t_verify = t_cycle - t_draft - t_accept     (verify 27B sur rows=1..4 + sampling + gaps)
+ référence MTP-OFF (même protocole W1/W2) pour le coût de verify unitaire.

Config lente prod EXACTE (shortlist + DEFER + H2 off), MTP ON/OFF, b2048, c167000,
mmproj, .so 8035caf4 figé. Point unique 114688 (réf prod).

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run mtp_prof_stepC.py --detach
"""
import json
import os
import subprocess
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(D, "night_bench.py"), "__name__": "mtp_prof_stepC"}
src = open(os.path.join(D, "night_bench.py")).read().split('if __name__ == "__main__":')[0]
src = src.replace("MAX_CTX = 196672", "MAX_CTX = 167000")
src = src.replace(
    '    if nthreads:\n        a += ["-t", str(nthreads)]\n    return a',
    '    if MTP_ON:\n'
    '        a += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-backend-sampling"]\n'
    '    if nthreads:\n        a += ["-t", str(nthreads)]\n    return a'
)
ns["MTP_ON"] = True
exec(src, ns)

SO_PROD = ns["SO_PROD"]
PORT = ns["PORT"]

OUT_JSONL = os.path.join(D, "mtp_prof_stepC_results.jsonl")
OUT_SUMMARY = os.path.join(D, "mtp_prof_stepC_summary.md")
OUT_SERVER_LOG = os.path.join(D, "mtp_prof_stepC_server.log")
OUT_GPUTS = os.path.join(D, "mtp_prof_stepC_gputimes.tsv")

ns["OUT_SERVER_LOG"] = OUT_SERVER_LOG
ns["OUT_GPUTS"] = OUT_GPUTS

log = ns["log"]

ENV_SLOW = {
    "GGML_CUDA_VOLTA_Q6_W4R4": "1",
    "GGML_CUDA_VOLTA_Q8_FATTN_TC": "0",
    "LLAMA_ARG_FIT_TARGET": "1024",
    "GGML_CUDA_QWEN35_MTP_SHORTLIST":
        "/home/guile/llama-experiments/src/llama-volta-merged/data/mtp-shortlists/qwen38-27b-exact-131072.i32",
    "LLAMA_ARG_SPEC_MTP_DEFER_PROMPT": "1",
}


def decompose(row):
    """Décomposition des phases à partir de spec_delta (steady W2)."""
    d = row.get("spec_delta_w2") or {}
    nd, ev = d.get("gen_drafts"), row.get("eval_ms_w2")
    if not nd or not ev:
        return {}
    t_cycle = ev / nd
    t_draft = (d.get("dur_g_ms", 0) / nd)
    t_acc = (d.get("dur_a_ms", 0) / nd)
    return {
        "n_drafts_w2": nd,
        "t_cycle_ms": round(t_cycle, 1),
        "t_draft_ms": round(t_draft, 2),
        "t_accept_ms": round(t_acc, 3),
        "t_verify_plus_ms": round(max(t_cycle - t_draft - t_acc, 0.0), 1),
        "mean_acc_len": d.get("mean_acc_len"),
        "acc_rate": d.get("acc_rate"),
    }


def main():
    import signal
    signal.signal(signal.SIGINT, ns["on_sig"])
    signal.signal(signal.SIGTERM, ns["on_sig"])
    if not ns["port_free"](PORT):
        sys.exit("port %d occupé — abandon" % PORT)
    log("PROF MTP STEP C — où passe le temps (harnais corrigé W1/W2) — lock superviseur")
    md5 = __import__("hashlib").md5(open(SO_PROD, "rb").read()).hexdigest()[:8]
    log("  · SO prod md5 = %s (figé)" % md5)

    ns["stop_and_free"]()
    open(OUT_SERVER_LOG, "w").close()
    open(OUT_GPUTS, "w").close()
    sampler = subprocess.Popen(["python3", os.path.join(D, "gputimes_sampler.py"), OUT_GPUTS, "0.1"])

    allrows = []
    try:
        for mtp in ("on", "off"):
            try:
                ns["MTP_ON"] = (mtp == "on")
                log(f"=== LEG MTP-{mtp.upper()} (114688, W1=64 W2=128)")
                ns["start_server"](SO_PROD, 2048, None, dict(ENV_SLOW))
                ns["warmup"](ns["log_size"]())
                since = ns["log_size"]()
                log("  · run_point2 (W1 prompt neuf 64 gen, W2 checkpoint 128 gen)")
                met = ns["run_point2"](114688, 64, 128, since)
                met["decomp"] = decompose(met)
                row = {"leg": f"mtp-{mtp}", "ctx": 114688, "so": md5,
                       "vram_mib": ns["vram_mib"](), **met}
                allrows.append(row)
                log(f"    → W1 (contaminée) {met.get('first_window_tps')} t/s | "
                    f"W2 (steady) {met.get('steady_tps')} t/s | catchup {met.get('catchup_ms')} ms")
                log(f"    → spec W2 : {met.get('spec_delta_w2')}")
                log(f"    → décomp  : {met.get('decomp')}")
            except Exception as e:
                log(f"  · MTP-{mtp.upper()} ECHEC : {e}")
            finally:
                ns["stop_server"]()
                time.sleep(12)
    finally:
        try:
            sampler.terminate(); sampler.wait(timeout=10)
        except Exception:
            pass
    ns["restore_service"]()

    with open(OUT_JSONL, "w") as f:
        for r in allrows:
            f.write(json.dumps(r, default=str) + "\n")

    L = ["# PROF MTP STEP C — où passe le temps MTP (harnais corrigé W1/W2) — 29/10", "",
         f"date : {time.strftime('%Y-%m-%d %H:%M UTC')}",
         f"SO prod figé = {md5} · service restauré : {'OUI' if ns['_restored'] else 'NON'}",
         "config lente prod exacte (H2 OFF, shortlist 131072, DEFER_PROMPT=1, KV q8_0, FA auto, -b 2048, c167000, mmproj)",
         "point : 114688 · W1 = 64 gen prompt neuf (catch-up) · W2 = 128 gen même prompt checkpointé (steady)", "",
         "| leg | W1 t/s | catchup ms | W2 steady t/s | W2 drafts | mean acc len | t_cycle | t_draft | t_verify+ |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in allrows:
        dc = r.get("decomp") or {}
        L.append("| %s | %s | %s | %s | %s | %s | %s ms | %s ms | %s ms |" % (
            r["leg"], r.get("first_window_tps", "?"), r.get("catchup_ms", "?"),
            r.get("steady_tps", "?"), dc.get("n_drafts_w2", "?"), dc.get("mean_acc_len", "?"),
            dc.get("t_cycle_ms", "?"), dc.get("t_draft_ms", "?"), dc.get("t_verify_plus_ms", "?")))
    L += ["",
          "Lecture : t_cycle = W2 eval_ms / n_drafts (1 cycle = 1 draft + 1 verify rows=1..4).",
          "t_draft mesuré par le serveur (t_draft_us). t_verify+ = verify 27B + sampling + gaps hôtes.",
          f"artefacts : {OUT_GPUTS} / {OUT_SERVER_LOG} / {OUT_JSONL}"]
    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(L) + "\n")
    log(f"FIN — synthèse : {OUT_SUMMARY}")
    sys.exit(0 if ns["_restored"] else 3)


if __name__ == "__main__":
    main()
