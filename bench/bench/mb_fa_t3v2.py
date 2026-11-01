#!/usr/bin/env python3
"""
mb_fa_t3v2.py — Microbench candidat T3 v2 : .so candidat (MB_FA_SO) vs .so prod T3 (d2df07b8).

Usage : T3V2_SO=/chemin/libggml-cuda-<variant>.so python3 supervisor_bench.py run mb_fa_t3v2.py <full|probe> --detach

Pour chaque .so (prod, candidat) et chaque mode (t3, stock) :
  - même harnais mb_fa, même géométrie (D=256, 24qh/4kvh, q8_0, n_kv=107351, T=1..8, 200 iters)
  - validation : la route `stock` des deux .so doit être quasi identique (le stock ne contient
    pas T3) -> prouve que le build du candidat n'a rien cassé hors kernel ; la route `t3` du
    candidat est le candidat lui-même.

Sortie : bench/mb_fa_t3v2_results.jsonl (lignes {variant, mode, T, us_med, out})
"""
import json
import os
import signal
import subprocess
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(D, "night_bench.py"), "__name__": "mb_fa_t3v2"}
src = open(os.path.join(D, "night_bench.py")).read().split('if __name__ == "__main__":')[0]
exec(src, ns)

log = ns["log"]
restore_service = ns["restore_service"]
stop_and_free = ns["stop_and_free"]
port_free = ns["port_free"]
PORT = ns["PORT"]

MB = os.path.join(D, "mb_fa")
OUT_JSONL = os.path.join(D, "mb_fa_t3v2_results.jsonl")
SO_PROD = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
SO_CAND = os.environ.get("T3V2_SO", "")

def run_variant(so, variant, iters, n_kv):
    env = dict(os.environ)
    # le .so candidat est lié contre le toolchain CUDA 12.6 privé (rpath absent) ;
    # la prod tourne sans car le service a déjà le .so chargé — on fournit le path ici
    env["LD_LIBRARY_PATH"] = ":".join([
        "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib",
        os.environ.get("LD_LIBRARY_PATH", ""),
    ])
    env["MB_FA_SO"] = so
    rows = []
    t0 = time.time()
    p = subprocess.run([MB, "t3", str(iters), str(n_kv)],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if p.returncode != 0:
        log(f"  · {variant} t3 : rc={p.returncode}\n---- stderr ----\n{p.stderr[-2000:]}")
        return None
    for line in p.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            r = json.loads(line)
            r["variant"] = variant
            r["mode"] = "t3"
            rows.append(r)
    p2 = subprocess.run([MB, "stock", str(iters), str(n_kv)],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if p2.returncode != 0:
        log(f"  · {variant} stock : rc={p2.returncode}\n---- stderr ----\n{p2.stderr[-2000:]}")
        return None
    for line in p2.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            r = json.loads(line)
            r["variant"] = variant
            r["mode"] = "stock"
            rows.append(r)
    log(f"  · {variant} : {len(rows)} lignes en {time.time()-t0:.0f}s")
    return rows

def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "full"
    iters = 200 if phase == "full" else 50
    n_kv = 107351 if phase == "full" else 2048

    signal.signal(signal.SIGINT, ns["on_sig"])
    signal.signal(signal.SIGTERM, ns["on_sig"])

    log(f"BENCH MB_FA T3v2 {phase} — cand={SO_CAND}")
    if not os.path.isfile(MB): sys.exit(f"binaire manquant : {MB}")
    if not SO_CAND or not os.path.isfile(SO_CAND): sys.exit(f".so candidat manquant : {SO_CAND!r}")
    if not port_free(PORT): sys.exit(f"port {PORT} occupé — abandon")

    if not stop_and_free():
        log("  · V100 pas libérée — on continue quand même")

    ok = True
    all_rows = []
    for so, variant in ((SO_PROD, "prod"), (SO_CAND, "cand")):
        rows = run_variant(so, variant, iters, n_kv)
        if rows is None:
            ok = False
            break
        all_rows.extend(rows)

    if ok and all_rows:
        # validation 1 : la route `stock` (T3 absent) doit être identique entre les deux .so
        d = {}
        for r in all_rows:
            d[(r["variant"], r["mode"], r["T"])] = r
        stock_worst = 0.0
        for T in sorted({t for (v, m, t) in d if m == "stock"}):
            a, b = d[("prod", "stock", T)]["us_med"], d[("cand", "stock", T)]["us_med"]
            stock_worst = max(stock_worst, abs(a-b)/max(a,b))
        # validation 2 : out[] de la route t3 candidat vs prod (m même attention)
        out_worst = 0.0
        for T in sorted({t for (v, m, t) in d if m == "t3"}):
            a, b = d[("prod", "t3", T)]["out"], d[("cand", "t3", T)]["out"]
            for j in range(4):
                out_worst = max(out_worst, abs(a[j]-b[j]) / (0.5 + abs(b[j])))
        log(f"  · VALIDATION : stock cand vs prod maxrel={stock_worst*100:.2f}% (doit rester <2%)")
        log(f"  · VALIDATION : out t3 cand vs prod maxrel={out_worst:.2e} (doit rester <1e-3)")
        ok = stock_worst < 0.02 and out_worst < 1e-3
        with open(OUT_JSONL, "w") as f:
            for r in all_rows:
                f.write(json.dumps(r) + "\n")
        write_summary(all_rows, n_kv)

    restore_service()
    st = ns["svc_state"]()
    log(f"  · FIN {phase} : service={st}, results={len(all_rows)}")
    if st != "active":
        log("  · !! service pas active — Intervention humaine requise")
        sys.exit(1)
    if not ok:
        sys.exit(2)

def write_summary(rows, n_kv):
    d = {}
    for r in rows:
        d[(r["variant"], r["mode"], r["T"])] = r
    lines = []
    lines.append("# Microbench T3 v2 — candidat vs prod (n_kv=%d)" % n_kv)
    lines.append("")
    lines.append("| T | prod t3 (µs) | cand t3 (µs) | Δ cand | prod stock (µs) | cand stock (µs) |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for T in sorted({t for (_, _, t) in d}):
        pt = d.get(("prod", "t3", T), {}).get("us_med")
        ct = d.get(("cand", "t3", T), {}).get("us_med")
        ps = d.get(("prod", "stock", T), {}).get("us_med")
        cs = d.get(("cand", "stock", T), {}).get("us_med")
        delta = f"{100*(pt-ct)/pt:+.1f}%" if pt and ct else "—"
        lines.append(f"| {T} | {pt:.0f} | {ct:.0f} | {delta} | {ps:.0f} | {cs:.0f} |")
    lines.append("")
    lines.append("Δ cand = (prod_t3 - cand_t3)/prod_t3 : positif = candidat plus rapide.")
    open(os.path.join(D, "mb_fa_t3v2_summary.md"), "w").write("\n".join(lines))
    log("  · summary -> mb_fa_t3v2_summary.md")

if __name__ == "__main__":
    main()
