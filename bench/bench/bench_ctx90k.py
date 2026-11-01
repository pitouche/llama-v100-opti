#!/usr/bin/env python3
"""
bench_ctx90k.py — BENCH DE REGIME (24/10) : séparer effet ctx long (FA) vs effet
acceptance MTP sur le per-token decode.

Contexte (reconciliation 32.6 vs 63 t/s, note aca153ff) :
  - per-token decode @ 36k, acceptance filler 3.94, warm = ~63 t/s (CUPTI 92.7% occ)
  - prod 37-45 t/s = acceptance 2.68 (x1.47 en steps) + ctx ~90k (FA/KV ~x2)
Prédiction per-token (sans l'overhead one-shot MTP, amorti sur 1024 gen) :
  - effet ctx seul (acceptance filler tient) : ~55-58 t/s
  - effet ctx + acceptance prod (2.68)      : ~43 t/s
Mesure : 2 points sur MÊME leg (config prod exacte, MTP ON, .so prod),
  P1 : ctx 36312, 1024 gen   (baseline même régime long-gen)
  P2 : ctx 90112, 1024 gen   (régime prod long-ctx)
fillers DISTINCTS (pas de prefix-match / pas de cache de prompt entre P1 et P2).
Le overhead one-shot MTP (~45 ms/ubatch x 44 ubatches @ 90k) est inclus dans le
eval time mais amorti sur 1024 gen (~2 s / 1024 tok -> < 1% d'impact).

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run bench_ctx90k.py [reps] --detach
"""
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(D, "night_bench.py"), "__name__": "night_bench_harness"}
src = open(os.path.join(D, "night_bench.py")).read().split('if __name__ == "__main__":')[0]

# MTP ON : config prod actuelle (-c 128000) + flags spec
src = src.replace("MAX_CTX = 196672", "MAX_CTX = 128000")
src = src.replace(
    '        # MTP OFF : pas de --spec-type (consigne 18/10 : MTP exclu)\n',
    '        # MTP ON : config prod exacte (24/10 — bench régime ctx)\n'
    '        "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-backend-sampling",\n'
)
exec(src, ns)

SO_PROD = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
OUT_JSONL = os.path.join(D, "bench_ctx90k_results.jsonl")
OUT_SUMMARY = os.path.join(D, "bench_ctx90k_summary.md")
OUT_SERVER_LOG = os.path.join(D, "bench_ctx90k_server.log")
OUT_GPUTS = os.path.join(D, "bench_ctx90k_gputimes.tsv")

ns["OUT_SERVER_LOG"] = OUT_SERVER_LOG
ns["OUT_GPUTS"] = OUT_GPUTS

log = ns["log"]
PORT = ns["PORT"]

# env prod complet (unit ollama-v100)
ENV_BASE = {
    "GGML_CUDA_VOLTA_Q6_W4R4": "1",
    "GGML_CUDA_VOLTA_Q8_FATTN_TC": "0",
    "LLAMA_ARG_FIT_TARGET": "1024",
    "GGML_CUDA_QWEN35_MTP_SHORTLIST":
        "/home/guile/llama-experiments/src/llama-volta-merged/data/mtp-shortlists/qwen38-27b-exact-131072.i32",
    "LLAMA_ARG_SPEC_MTP_DEFER_PROMPT": "1",
}

N_GEN = 1024

# deux fillers de densité comparable (~24 mots / ~30 tokens), SANS prefix commun
FILL_36 = ("La tour de contrôle surveille le passage des avions au crépuscule "
           "pendant que le chef de piste organise les départs de la soirée. ")
FILL_90 = ("Le phare de la pointe balaye la baie endormie tandis que les pêcheurs "
           "réparent leurs filets usés sur le quai de bois près du moulin. ")

POINTS = [
    ("P1-36k", 36312, FILL_36),
    ("P2-90k", 90112, FILL_90),
]


def parse_mtp(logf, since_pos):
    """acceptance / mean len / acc rate/pos depuis le log."""
    try:
        with open(logf, "r", errors="replace") as f:
            data = f.read()[since_pos:]
    except OSError:
        return {}
    out = {}
    m = re.findall(r"draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([\d.]+)", data)
    if m:
        out["mtp_acceptance"] = float(m[-1][0])
        out["mtp_mean_len"] = float(m[-1][3])
    m = re.findall(r"#mean acc len = ([\d.]+), #acc rate/pos = \(([\d.]+), ([\d.]+), ([\d.]+)\)", data)
    if m:
        out["acc_mean_len"] = float(m[-1][0])
        out["acc_rate_pos"] = (float(m[-1][1]), float(m[-1][2]), float(m[-1][3]))
    return out


orig_parse = ns["parse_point"]
def parse_point2(logf, since_pos):
    out = orig_parse(logf, since_pos)
    out.update(parse_mtp(OUT_SERVER_LOG, since_pos))
    return out
ns["parse_point"] = parse_point2


def make_prompt_text(fill):
    def prompt_text(ctx_target_tokens):
        n_words = ctx_target_tokens * 4 // 5
        words = fill.split() * max(1, n_words // max(1, fill.count(" ") - 1) + 1)
        return " ".join(words[: n_words])
    return prompt_text


def so_prefix(p):
    try:
        return hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
    except OSError:
        return "?"


def main():
    signal.signal(signal.SIGINT, ns["on_sig"])
    signal.signal(signal.SIGTERM, ns["on_sig"])
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    log(f"BENCH REGIME CTX (MTP ON, config prod) — reps={reps} — N_GEN={N_GEN}")
    if not ns["port_free"](PORT):
        sys.exit(f"port {PORT} occupé — abandon")
    if so_prefix(SO_PROD) != "fa3e75c2":
        sys.exit(f"ABORT : .so prod {so_prefix(SO_PROD)} != fa3e75c2 — prod dérivé, ne pas bench")
    log(f"  · SO PROD md5 = {so_prefix(SO_PROD)} (leg unique A-prod)")

    ns["stop_and_free"]()
    open(OUT_SERVER_LOG, "w").close()
    open(OUT_GPUTS, "w").close()

    sampler = subprocess.Popen(["python3", os.path.join(D, "gputimes_sampler.py"), OUT_GPUTS])

    allrows = []
    try:
        for rep in range(1, reps + 1):
            log(f"=== rep{rep} :: A-prod (so={so_prefix(SO_PROD)})")
            try:
                ns["start_server"](SO_PROD, 2048, None, ENV_BASE)
                wu = ns["warmup"](ns["log_size"]())
                for pname, ctx, fill in POINTS:
                    ns["prompt_text"] = make_prompt_text(fill)
                    since = ns["log_size"]()
                    log(f"  · {pname} : point {ctx} -> {N_GEN} gen (warmup={wu})")
                    api, met = ns["run_point"](ctx, N_GEN, since)
                    row = {"leg": "A-prod", "rep": rep, "point": pname, "ctx": ctx,
                           "mtp": "on", "n_gen": N_GEN, "so": so_prefix(SO_PROD),
                           **api, **met}
                    allrows.append(row)
                    log(f"    → prefill {met.get('prompt_tps','?')} t/s "
                        f"({met.get('prompt_tokens','?')} tok) | "
                        f"decode {met.get('eval_tps','?')} t/s ({met.get('eval_tokens','?')} tok) | "
                        f"acc {met.get('mtp_acceptance','?')} mean_len {met.get('mtp_mean_len','?')} | "
                        f"rate/pos {met.get('acc_rate_pos','?')} | {api.get('error','ok')}")
            except Exception as e:
                log(f"  · rep{rep} ECHEC : {e}")
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

    L = ["# BENCH DE REGIME CTX — 36k vs 90k, 1024 gen — MTP ON config prod — 24/10", "",
         f"date : {time.strftime('%Y-%m-%d %H:%M UTC')}",
         f"reps : {reps} · service restauré : {'OUI' if ns['_restored'] else 'NON'}",
         f".so prod {so_prefix(SO_PROD)} · -b 2048 · MTP ON (spec-draft-n-max 3) · {N_GEN} gen",
         "",
         "| point | rep | prefill t/s | decode t/s | wall s | acc | mean len | acc rate/pos |",
         "|---|---|---|---|---|---|---|---|"]
    for pname, ctx, _ in POINTS:
        for rep in range(1, reps + 1):
            r = next((r for r in allrows if r["point"] == pname and r.get("rep") == rep), None)
            if not r:
                continue
            rp = r.get("acc_rate_pos")
            rps = f"({rp[0]:.3f}, {rp[1]:.3f}, {rp[2]:.3f})" if rp else "?"
            L.append(f"| {pname} | {rep} | {r.get('prompt_tps','?')} | "
                     f"{r.get('eval_tps','?')} | {r.get('wall_s','?')} | "
                     f"{r.get('mtp_acceptance','?')} | {r.get('mtp_mean_len','?')} | {rps} |")
    L += ["", "## médianes (toutes reps)"]
    for pname, _, _ in POINTS:
        v = [r["eval_tps"] for r in allrows if r["point"] == pname and r.get("eval_tps")]
        L.append(f"  - {pname}: decode {ns['med'](v)} t/s")
    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(L) + "\n")
    log(f"FIN — synthèse : {OUT_SUMMARY}")
    sys.exit(0 if ns["_restored"] else 3)


if __name__ == "__main__":
    main()
