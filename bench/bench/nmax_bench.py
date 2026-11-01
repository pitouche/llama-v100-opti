#!/usr/bin/env python3
"""
nmax_bench.py — Bench A/B --spec-draft-n-max 3 vs 4 sur le .so T3 PRODUCTION (d2df07b8), MTP ON.

Contexte (19/10) : le GGUF prod (Qwen3.8-27B q6 MTP) porte un SEUL module MTP
(blk.64.nextn.*, n_mtp_layers=1, mode "single head", pas de chain_heads).
Le code draft (common/speculative.cpp, common_speculative_impl_draft_mtp) utilise
le même head à chaque pas de draft (batch croissant pos0+1, pos0+2, ...) : n_max est
donc libre, pas plafonné au nombre de couches MTP. Avec n_max=3 le mean acc len
observé est ~3.5-3.8 (saturé sur le plafond, rate pos2 ~0.83-0.91) → on teste n_max=4.

Config : même .so prod, même config prod, seul --spec-draft-n-max change.
2 legs interleave, serveur frais par leg, points 114688 + 160000.

Lecture attendue :
- A : mean acc len ≈ 3.5-3.8, acc rate pos0-2 ≈ (0.97, 0.89, 0.84)
- B : si gain → acc rate pos3 > ~0.5 et decode t/s en hausse ; mean acc len monte.
"""
import json
import os
import signal
import subprocess
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(D, "night_bench.py"), "__name__": "nmax_bench_harness"}
src = open(os.path.join(D, "night_bench.py")).read().split('if __name__ == "__main__":')[0]
src = src.replace("MAX_CTX = 196672", "MAX_CTX = 167000")
src = src.replace(
    '        # MTP OFF : pas de --spec-type (consigne 18/10 : MTP exclu)\n',
    '        # MTP ON : config prod (nmax sweep, --spec-draft-n-max injeté par le bench)\n'
    '        "--spec-type", "draft-mtp", "--spec-draft-backend-sampling",\n'
)
exec(src, ns)

SO_T3 = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"  # prod d2df07b8

OUT_JSONL = os.path.join(D, "nmax_bench_results.jsonl")
OUT_SUMMARY = os.path.join(D, "nmax_bench_summary.md")
OUT_SERVER_LOG = os.path.join(D, "nmax_bench_server.log")
OUT_GPUTS = os.path.join(D, "nmax_bench_gputimes.tsv")

ns["OUT_SERVER_LOG"] = OUT_SERVER_LOG
ns["OUT_GPUTS"] = OUT_GPUTS

log = ns["log"]
PORT = ns["PORT"]
N_GEN = ns["N_GEN"]


def start_server_nmax(n_max, extra_env=None):
    """start_server du harness + --spec-draft-n-max (l'argument est ajouté au server_args
    en post-traitement via le hack : on patche ns['server_args'] avant appel)."""
    a = ns["server_args"](2048, None)
    i = a.index("--spec-type")  # a[i+1] = "draft-mtp" -> insérer APRES la paire
    a = a[:i+2] + ["--spec-draft-n-max", str(n_max)] + a[i+2:]
    ns["_server_args_override"] = a
    # le harness construit ses args via server_args() ; on intercepte :
    orig = ns["server_args"]
    ns["server_args"] = lambda *x, **k: a
    try:
        ns["start_server"](SO_T3, 2048, None, extra_env)
    finally:
        ns["server_args"] = orig


def scrape_acc_stats():
    """Dernières lignes de stats MTP du log serveur (mean acc len, acc rate/pos)."""
    out = {}
    try:
        with open(OUT_SERVER_LOG, "r", errors="replace") as f:
            tail = f.readlines()[-400:]
        lines = [l for l in tail if "common_specu: statistics" in l or "draft acceptance" in l]
        out["last_spec_stats"] = [l.strip() for l in lines if "statistics" in l][-2:]
        out["last_slot_acc"] = [l.strip() for l in lines if "draft acceptance" in l][-2:]
    except Exception as e:
        out["err"] = str(e)
    return out


def main():
    signal.signal(signal.SIGINT, ns["on_sig"])
    signal.signal(signal.SIGTERM, ns["on_sig"])
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    log(f"BENCH nmax (n-max 3 vs 4, .so T3 prod) MTP ON — reps={reps}")
    if not ns["port_free"](PORT):
        sys.exit(f"port {PORT} occupe — abandon")
    if not os.path.isfile(SO_T3):
        sys.exit(f"lib manquante : {SO_T3}")
    log(f"  · SO prod md5 = {__import__('hashlib').md5(open(SO_T3,'rb').read()).hexdigest()[:8]}")

    ns["stop_and_free"]()
    open(OUT_SERVER_LOG, "w").close()
    open(OUT_GPUTS, "w").close()

    sampler = subprocess.Popen(["python3", os.path.join(D, "gputimes_sampler.py"), OUT_GPUTS])

    CTX_LONG = [114688, 160000]

    EXP = [
        ("EXP-NMAX", "A-nmax3", 3, CTX_LONG),
        ("EXP-NMAX", "B-nmax4", 4, CTX_LONG),
    ]

    allrows = []
    try:
        for rep in range(1, reps + 1):
            legs = EXP if rep % 2 == 1 else list(reversed(EXP))
            for tag, name, nmax, ctxs in legs:
                try:
                    log(f"=== NMAX rep{rep} :: LEG {name} (n-max={nmax})")
                    start_server_nmax(nmax)
                    ns["warmup"](ns["log_size"]())
                    for ctx in ctxs:
                        since = ns["log_size"]()
                        log(f"  · point {ctx} → {N_GEN} gen")
                        api, met = ns["run_point"](ctx, N_GEN, since)
                        acc = scrape_acc_stats()
                        row = {"exp": tag, "leg": name, "rep": rep, "ctx": ctx,
                               "mtp": "on", "nmax": nmax, "batch": 2048, "nthreads": None,
                               "vram_mib": ns["vram_mib"](),
                               **api, **met, "acc": acc}
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
    L = ["# BENCH nmax — n-max 3 vs 4 — .so T3 PROD (d2df07b8) — 19/10 — MTP ON (config prod), -b 2048, interleave", "",
         f"date : {time.strftime('%Y-%m-%d %H:%M UTC')}",
         f"reps : {reps} · service restauré : {'OUI' if ns['_restored'] else 'NON'}",
         "config : -c 167000, --spec-type draft-mtp --spec-draft-backend-sampling, KV q8_0, FA auto",
         "binaire : SO_T3 PROD — A = --spec-draft-n-max 3 (prod actuelle), B = --spec-draft-n-max 4", ""]
    for ctx in CTX_LONG:
        L.append(f"## ctx {ctx}")
        L.append("| leg | rep | decode t/s | prefill t/s | acc (dernier slot) |")
        L.append("|---|---|---|---|---|")
        for rep in range(1, reps + 1):
            for name in ("A-nmax3", "B-nmax4"):
                rs = [r for r in allrows if r["leg"] == name and r["ctx"] == ctx and r.get("rep") == rep]
                if rs:
                    accs = rs[0].get("acc", {}).get("last_slot_acc", [])
                    L.append(f"| {name} | {rep} | {rs[0].get('eval_tps','?')} | "
                             f"{rs[0].get('prompt_tps','?')} | {accs[-1] if accs else '?'} |")
        L.append("")
        vals = {}
        for name in ("A-nmax3", "B-nmax4"):
            rs = [r for r in allrows if r["leg"] == name and r["ctx"] == ctx]
            vals[name] = [r["eval_tps"] for r in rs if r.get("eval_tps")]
        ab = med(vals["A-nmax3"]); bb = med(vals["B-nmax4"])
        L.append("Δ decode vs A-nmax3 (médiane) :")
        if ab and bb:
            L.append(f"  - B-nmax4: {(bb-ab)/ab*100:+.2f}%")
        L.append("")
    L.append("## stats MTP brutes (spec statistics, par leg)")
    for r in allrows:
        a = r.get("acc", {})
        L.append(f"- {r['leg']} rep{r['rep']} ctx{r['ctx']}: " + " || ".join(a.get("last_spec_stats", ["?"])))
    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(L) + "\n")
    log(f"FIN — synthèse : {OUT_SUMMARY}")
    sys.exit(0 if ns["_restored"] else 3)


if __name__ == "__main__":
    main()
