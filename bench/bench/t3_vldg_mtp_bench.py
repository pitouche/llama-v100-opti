#!/usr/bin/env python3
"""
t3_vldg_mtp_bench.py — A/B e2e T3 vldg (double-buffer V, GO microbench + GO e2e MTP-OFF)
vs prod, **MTP ON = config prod exacte** (28/10).

Suite du gold standard : le gain MTP-OFF est validé (+2,25 % @114k, +2,81 % @196k).
Ici on mesure en régime réel prod (MTP ON, n-max 3) : le kernel T3 traite alors
TANT les rows=1 (draft) QUE les rows=2..4 (verify) — exactement les shapes T=1..4
du microbench vldg (+11 % @T=1, +14,5..18 % @T=2..3).

2 legs (interleave 2 reps, serveur FRAIS par leg, port 8917, CVD=0) :
  A-prod : .so prod ACTUEL (4fd152f9 = T3 + levier B + levier C)
  B-vldg : .so vldg (87f85442 = T3 réécrit, B+C intacts, bit-exact)

Même binaire sinon → **B vs A = gain e2e MTP ON de vldg**.
Config prod : -c 167000, -b 2048, --spec-type draft-mtp --spec-draft-n-max 3
--spec-draft-backend-sampling, KV q8_0, FA auto, env prod (H2 OFF, shortlist,
DEFER_PROMPT=1). Points : 114688 et 160000 (mêmes points que §5.2/5.4).
Mesure : decode t/s + acceptance MTP (doit rester identique — bit-exact).

USAGE :
  python3 supervisor_bench.py run t3_vldg_mtp_bench.py 2 --detach
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
ns = {"__file__": os.path.join(D, "night_bench.py"), "__name__": "t3_vldg_mtp_harness"}
src = open(os.path.join(D, "night_bench.py")).read().split('if __name__ == "__main__":')[0]
src = src.replace("MAX_CTX = 196672", "MAX_CTX = 167000")
src = src.replace(
    '        # MTP OFF : pas de --spec-type (consigne 18/10 : MTP exclu)\n',
    '        # MTP ON : config prod exacte (bench vldg MTP ON)\n'
    '        "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-backend-sampling",\n'
)
exec(src, ns)

SO_VLDG = "/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib/lib/libggml-cuda-vldg.so"

OUT_JSONL = os.path.join(D, "t3_vldg_mtp_bench_results.jsonl")
OUT_SUMMARY = os.path.join(D, "t3_vldg_mtp_bench_summary.md")
OUT_SERVER_LOG = os.path.join(D, "t3_vldg_mtp_bench_server.log")
OUT_GPUTS = os.path.join(D, "t3_vldg_mtp_bench_gputimes.tsv")

ns["OUT_SERVER_LOG"] = OUT_SERVER_LOG
ns["OUT_GPUTS"] = OUT_GPUTS

log = ns["log"]
SO_PROD = ns["SO_PROD"]
PORT = ns["PORT"]
N_GEN = ns["N_GEN"]

# env prod complet (unit ollama-v100 — OPTI §4)
ENV_BASE = {
    "GGML_CUDA_VOLTA_Q6_W4R4": "1",
    "GGML_CUDA_VOLTA_Q8_FATTN_TC": "0",
    "LLAMA_ARG_FIT_TARGET": "1024",
    "GGML_CUDA_QWEN35_MTP_SHORTLIST":
        "/home/guile/llama-experiments/src/llama-volta-merged/data/mtp-shortlists/qwen38-27b-exact-131072.i32",
    "LLAMA_ARG_SPEC_MTP_DEFER_PROMPT": "1",
}


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


def md5s(p):
    try:
        return hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
    except OSError:
        return "?"


def main():
    signal.signal(signal.SIGINT, ns["on_sig"])
    signal.signal(signal.SIGTERM, ns["on_sig"])
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    log(f"BENCH T3 VLDG (MTP ON, config prod) — reps={reps} — lock tenu par le superviseur")
    if not ns["port_free"](PORT):
        sys.exit(f"port {PORT} occupé — abandon")
    for f in (SO_PROD, SO_VLDG):
        if not os.path.isfile(f):
            sys.exit(f"lib manquante : {f}")
    pm, vm = md5s(SO_PROD), md5s(SO_VLDG)
    log(f"  · SO prod md5 = {pm} (attendu 4fd152f9)")
    log(f"  · SO vldg md5 = {vm} (attendu 87f85442)")
    if pm == vm:
        sys.exit("ABORT : prod == vldg — rien à mesurer")

    ns["stop_and_free"]()
    open(OUT_SERVER_LOG, "w").close()
    open(OUT_GPUTS, "w").close()

    sampler = subprocess.Popen(["python3", os.path.join(D, "gputimes_sampler.py"), OUT_GPUTS])

    CTX_LONG = [114688, 160000]

    EXP = [
        ("EXP-VLDG-MTP", "A-prod", SO_PROD, dict(ENV_BASE), CTX_LONG),
        ("EXP-VLDG-MTP", "B-vldg", SO_VLDG, dict(ENV_BASE), CTX_LONG),
    ]

    allrows = []
    try:
        for rep in range(1, reps + 1):
            legs = EXP if rep % 2 == 1 else list(reversed(EXP))
            for tag, name, so, extra_env, ctxs in legs:
                try:
                    log(f"=== VLDG-MTP rep{rep} :: LEG {name} (md5 {md5s(so)})")
                    ns["start_server"](so, 2048, None, extra_env)
                    ns["warmup"](ns["log_size"]())
                    for ctx in ctxs:
                        since = ns["log_size"]()
                        log(f"  · point {ctx} → {N_GEN} gen")
                        api, met = ns["run_point"](ctx, N_GEN, since)
                        row = {"exp": tag, "leg": name, "rep": rep, "ctx": ctx,
                               "mtp": "on", "batch": 2048, "nthreads": None,
                               "vram_mib": ns["vram_mib"](), **api, **met}
                        allrows.append(row)
                        log(f"    → decode {met.get('eval_tps','?')} t/s | "
                            f"acc {met.get('mtp_acceptance','?')} mean_len {met.get('mtp_mean_len','?')} | "
                            f"rate/pos {met.get('acc_rate_pos','?')} | {api.get('error','ok')}")
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
            f.write(json.dumps(r, default=str) + "\n")

    med = ns["med"]
    L = ["# BENCH T3 VLDG (double-buffer V) — MTP ON config prod — 28/10", "",
         f"date : {time.strftime('%Y-%m-%d %H:%M UTC')}",
         f"reps : {reps} · service restauré : {'OUI' if ns['_restored'] else 'NON'}",
         f"SO prod = {pm} · SO vldg = {vm}",
         "ctx 114688 / 160000 · -b 2048 · MTP ON (spec-draft-n-max 3) · 64 gen",
         "Seule diff entre les .so = le kernel T3 → B vs A = gain e2e MTP ON de vldg.",
         "L'acceptance MTP doit rester identique (bit-exact).", ""]
    for ctx in CTX_LONG:
        L.append(f"## ctx {ctx}")
        L.append("| leg | rep | decode t/s | prefill t/s | acc | mean len | acc rate/pos |")
        L.append("|---|---|---|---|---|---|---|")
        for rep in range(1, reps + 1):
            for name in ("A-prod", "B-vldg"):
                rs = [r for r in allrows if r["leg"] == name and r["ctx"] == ctx and r.get("rep") == rep]
                if rs:
                    r = rs[0]
                    rp = r.get("acc_rate_pos")
                    rps = f"({float(rp[0]):.3f},{float(rp[1]):.3f},{float(rp[2]):.3f})" if rp else "?"
                    L.append(f"| {name} | {rep} | {r.get('eval_tps','?')} | {r.get('prompt_tps','?')} | "
                             f"{r.get('mtp_acceptance','?')} | {r.get('mtp_mean_len','?')} | {rps} |")
        L.append("")
        vals = {}
        for name in ("A-prod", "B-vldg"):
            rs = [r for r in allrows if r["leg"] == name and r["ctx"] == ctx]
            vals[name] = [r["eval_tps"] for r in rs if r.get("eval_tps")]
        bp, vp = med(vals["A-prod"]), med(vals["B-vldg"])
        if bp and vp:
            L.append(f"Δ decode vs A-prod (médiane) : **{(vp-bp)/bp*100:+.2f}%**  (A={bp} → B={vp} t/s)")
        L.append("")

    L.append("## VRAM pic par leg (MiB)")
    L.append("| leg | max VRAM |")
    L.append("|---|---|")
    seen = {}
    for r in allrows:
        if r.get("vram_mib"):
            seen[r["leg"]] = max(seen.get(r["leg"], 0), r["vram_mib"])
    for k, v in sorted(seen.items()):
        L.append(f"| {k} | {v} |")

    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(L) + "\n")
    log(f"FIN — synthèse : {OUT_SUMMARY}")
    sys.exit(0 if ns["_restored"] else 3)


if __name__ == "__main__":
    main()
