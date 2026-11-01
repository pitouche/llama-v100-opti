#!/usr/bin/env python3
"""
bench_mtp_sampling.py — MTP-ACCEPT (24/10 04:45 UTC) : isoler l'effet SAMPLING sur l'acceptance MTP.

Hypothèse (après analyse source + journal prod, cf. note aca153ff section MTP-ACCEPT) :
  - Le draft MTP echantillonne via une chain FIXEE : top_k(10) SANS temperature
    (speculative.cpp:506-520, backend_sampling=1 ; prod log : n_max=3, n_min=0, p_min=0.00).
  - Le TARGET echantillonne avec la temperature du request (prod agent = temp>0).
  - Tous nos benches = temp 0/top_k 1 (target greedy) -> le draft (argmax in top-10) et le
    target (argmax) sont quasi alignes -> acceptance filler 2.88-3.03 / clean 3.94.
  - Prod : acc 2.62-2.68, acc rate/pos (0.717, 0.521, 0.385).
  - WARNING ctx_dft pos_max = faux positif systematique (pattern N_prev-1-4, checkpoint N-4)
    -> PAS la cause.

Mesure : MEME regime 90k (filler P2 de bench_ctx90k, KV fraiche par point), MÊME leg,
  3 regimes de sampling : (temp 0, top_k 1) / (0.7, top_k 20) / (1.0, top_k 40).
  ignore_eos force (256 gen exacts par point pour des stats d'acceptance comparables).
Si acc(1.0) ~= acc prod (2.6-2.7) -> la chute prod est l'effet sampling, pas un bug.
Levier code ensuite : appliquer la temperature du request a la chain du draft.

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run bench_mtp_sampling.py [reps] --detach
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

# MTP ON : config prod exacte (24/10)
src = src.replace("MAX_CTX = 196672", "MAX_CTX = 128000")
src = src.replace(
    '        # MTP OFF : pas de --spec-type (consigne 18/10 : MTP exclu)\n',
    '        # MTP ON : config prod exacte\n'
    '        "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-backend-sampling",\n'
)
exec(src, ns)

SO_PROD = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
OUT_JSONL = os.path.join(D, "bench_mtp_sampling_results.jsonl")
OUT_SUMMARY = os.path.join(D, "bench_mtp_sampling_summary.md")
OUT_SERVER_LOG = os.path.join(D, "bench_mtp_sampling_server.log")
OUT_GPUTS = os.path.join(D, "bench_mtp_sampling_gputimes.tsv")

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

N_GEN = 256
CTX = 90112

# meme filler que P2 de bench_ctx90k (regime prod 90k)
FILL_90 = ("Le phare de la pointe balaye la baie endormie tandis que les pêcheurs "
           "réparent leurs filets usés sur le quai de bois près du moulin. ")

# (tag, temperature, top_k)
POINTS = [
    ("T0-greedy", 0.0, 1),
    ("T07-topk20", 0.7, 20),
    ("T10-topk40", 1.0, 40),
]


def parse_mtp(logf, since_pos):
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

# run_point avec temperature/top_k parametrables + ignore_eos (256 gen exacts)
orig_run_point = ns["run_point"]

def run_point_samp(ctx_target, n_gen, since_pos, tag="", temp=0.0, top_k=1):
    text = ns["prompt_text"](ctx_target)
    data = {
        "model": ns["_state"]["model_id"], "stream": False,
        "messages": [{"role": "user", "content": text + "\nRéponds en une phrase."}],
        "max_tokens": n_gen, "temperature": temp, "top_k": top_k, "top_p": 1.0,
        "ignore_eos": True,
    }
    t0 = time.time()
    try:
        r = ns["http_json"](f"http://127.0.0.1:{ns['PORT']}/v1/chat/completions", data,
                            timeout=ns["TIMEOUT_S"])
        u = (r or {}).get("usage", {})
        api = {"prompt_tokens_api": u.get("prompt_tokens"),
               "gen_tokens_api": u.get("completion_tokens"),
               "wall_s": round(time.time() - t0, 1)}
    except Exception as e:
        api = {"error": str(e)}
    return api, parse_point2(OUT_SERVER_LOG, since_pos)

ns["run_point"] = run_point_samp


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
    log(f"BENCH MTP SAMPLING (regime 90k, MTP ON config prod) — reps={reps} — N_GEN={N_GEN} CTX={CTX}")
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
                ns["prompt_text"] = make_prompt_text(FILL_90)
                for tag, temp, top_k in POINTS:
                    since = ns["log_size"]()
                    log(f"  · {tag} : temp={temp} top_k={top_k} ctx {CTX} -> {N_GEN} gen (warmup={wu})")
                    api, met = ns["run_point"](CTX, N_GEN, since, tag=tag, temp=temp, top_k=top_k)
                    row = {"leg": "A-prod", "rep": rep, "point": tag, "temp": temp, "top_k": top_k,
                           "ctx": CTX, "mtp": "on", "n_gen": N_GEN, "so": so_prefix(SO_PROD),
                           **api, **met}
                    allrows.append(row)
                    log(f"    → decode {met.get('eval_tps','?')} t/s ({met.get('eval_tokens','?')} tok) | "
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

    L = ["# BENCH MTP SAMPLING — régime 90k, 3 temperatures, 256 gen — MTP ON config prod — 24/10", "",
         f"date : {time.strftime('%Y-%m-%d %H:%M UTC')}",
         f"reps : {reps} · service restauré : {'OUI' if ns['_restored'] else 'NON'}",
         f".so prod {so_prefix(SO_PROD)} · -b 2048 · MTP ON (spec-draft-n-max 3, backend-sampling) · {N_GEN} gen ignore_eos",
         "",
         "Réf : prod acc 2.62-2.68 (0.717, 0.521, 0.385) ; filler temp0 (bench_ctx90k P2) 2.88 (0.834, 0.658, 0.492) ; clean temp0 3.94 (run9c).",
         "",
         "| point | temp | top_k | rep | decode t/s | gen tok | acc | mean len | acc rate/pos |",
         "|---|---|---|---|---|---|---|---|---|"]
    for tag, temp, top_k in POINTS:
        for rep in range(1, reps + 1):
            r = next((r for r in allrows if r["point"] == tag and r.get("rep") == rep), None)
            if not r:
                continue
            rp = r.get("acc_rate_pos")
            rps = f"({rp[0]:.3f}, {rp[1]:.3f}, {rp[2]:.3f})" if rp else "?"
            L.append(f"| {tag} | {temp} | {top_k} | {rep} | {r.get('eval_tps','?')} | "
                     f"{r.get('eval_tokens','?')} | {r.get('mtp_acceptance','?')} | "
                     f"{r.get('mtp_mean_len','?')} | {rps} |")
    L += ["", "## médianes (toutes reps)"]
    for tag, _, _ in POINTS:
        vt = [r["eval_tps"] for r in allrows if r["point"] == tag and r.get("eval_tps")]
        vm = [r["mtp_mean_len"] for r in allrows if r["point"] == tag and r.get("mtp_mean_len")]
        mt = ns["med"](vt) if vt else "?"
        mm = ns["med"](vm) if vm else "?"
        L.append(f"  - {tag}: decode {mt} t/s, mean_len {mm}")
    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(L) + "\n")
    log(f"FIN — synthèse : {OUT_SUMMARY}")

if __name__ == "__main__":
    main()
