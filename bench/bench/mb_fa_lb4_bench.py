#!/usr/bin/env python3
"""
mb_fa_lb4_bench.py — A/B levier B : launch_bounds minblocks=4 (NT=1), 24/10.

2 lancements du microbench mb_fa (MÊME binaire, géométrie prod MTP-OFF), dlopen du .so
choisi par MB_FA_SO, SANS env GGML_CUDA_FATTN_PB (launcher identique à la prod, pb auto):
  REF : prod fa3e75c2 (lba : NT=1 -> minblocks=3, 162 regs, 3 blocs/SM)
  LB4 : libggml-cuda-t3lba_lb4.so (feaf05aa : NT=1 -> minblocks=4, 124 regs, 4 blocs/SM)

Seule différence : le launch_bounds du kernel T3 sur NT=1. NT>=2 (T>=2) garde minblocks=2
dans les deux -> T=2..8 servent de contrôle (doivent être identiques à ±bruit).
Le levier B ne peut donc agir QUE sur T=1 (le cas decode single-token, latence-bound).

Protocole : bench/t3_lbB_protocol.md. Lancer via
  python3 supervisor_bench.py run mb_fa_lb4_bench.py --detach
Usage : mb_fa_lb4_bench.py [iters] [n_kv]  (défauts 200 / 107351)
"""
import json
import os
import signal
import subprocess
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(D, "night_bench.py"), "__name__": "mb_fa_lb4_harness"}
src = open(os.path.join(D, "night_bench.py")).read().split('if __name__ == "__main__":')[0]
exec(src, ns)

log = ns["log"]
svc_state = ns["svc_state"]
restore_service = ns["restore_service"]
stop_and_free = ns["stop_and_free"]
port_free = ns["port_free"]
PORT = ns["PORT"]

MB = os.path.join(D, "mb_fa")
PROD_SO = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
LB4_SO = "/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib/lib/libggml-cuda-t3lba_lb4.so"
OUT_JSONL = os.path.join(D, "mb_fa_lb4_results.jsonl")
OUT_SUMMARY = os.path.join(D, "mb_fa_lb4_summary.md")

VARIANTS = [
    ("REF", PROD_SO),
    ("LB4", LB4_SO),
]

def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_kv = int(sys.argv[2]) if len(sys.argv) > 2 else 107351

    signal.signal(signal.SIGINT, ns["on_sig"])
    signal.signal(signal.SIGTERM, ns["on_sig"])

    log(f"BENCH MB_FA_LB4 B — launch_bounds NT=1 minblocks=4 — iters={iters} n_kv={n_kv}")
    if not os.path.isfile(MB):
        sys.exit(f"binaire manquant : {MB}")
    if not os.path.isfile(LB4_SO):
        sys.exit(f"so candidat manquant : {LB4_SO}")
    if not port_free(PORT):
        sys.exit(f"port {PORT} occupé — abandon")

    md5 = subprocess.run(["md5sum", PROD_SO], capture_output=True, text=True).stdout.split()[0][:8]
    log(f"  · prod md5 avant : {md5}")
    if not stop_and_free():
        log("  · V100 pas libérée — on continue quand même (VRAM résiduelle ?)")

    results, ok = [], True
    for name, so in VARIANTS:
        env = os.environ.copy()
        env["MB_FA_SO"] = so
        env["LD_LIBRARY_PATH"] = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12" + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        env.pop("GGML_CUDA_FATTN_PB", None)   # launcher identique à la prod : pb auto
        t0 = time.time()
        p = subprocess.run([MB, "t3", str(iters), str(n_kv)],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        if p.returncode != 0:
            log(f"  · {name} : rc={p.returncode}\n---- stderr ----\n{p.stderr[-3000:]}")
            ok = False
            break
        rows = []
        for line in p.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    log(f"  · {name} : ligne JSON illisible {line!r} ({e})")
        for r in rows:
            r["variant"] = name
            results.append(r)
        log(f"  · {name} : rc=0 en {time.time()-t0:.0f}s, {len(rows)} lignes (so={os.path.basename(so)})")
        if p.stderr:
            log("    stderr : " + p.stderr.strip().replace("\n", " | ")[:400])

    # --- validation numérique : REF et LB4 doivent donner la même attention ---
    maxrel = 0.0
    if ok:
        byT = {}
        for r in results:
            byT.setdefault(r["T"], {})[r["variant"]] = r
        for T, v in byT.items():
            if len(v) < 2:
                continue
            for j in range(4):
                ref = v["REF"]["out"][j]
                maxrel = max(maxrel, abs(v["LB4"]["out"][j] - ref) / (0.5 + abs(ref)))
        ok = maxrel < 1e-4
        log(f"  · VALIDATION numérique : maxrel={maxrel:.2e} (tol 1e-4) -> {'OK' if ok else 'KO'}")

    if ok and results:
        with open(OUT_JSONL, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        write_summary(results, n_kv, iters)
    else:
        log("  · bench pas concluant — résultats non écrits")

    restore_service()
    st = svc_state()
    md5b = subprocess.run(["md5sum", PROD_SO], capture_output=True, text=True).stdout.split()[0][:8]
    log(f"  · FIN B : service={st}, prod md5 après={md5b}, results={len(results)}")
    if st != "active":
        log("  · !! service pas active — Intervention humaine requise")
        sys.exit(1)
    if md5b != md5:
        log("  · !! prod md5 a CHANGÉ — Intervention humaine requise")
        sys.exit(1)
    if not ok:
        sys.exit(2)

def write_summary(results, n_kv, iters):
    byT = {}
    for r in results:
        byT.setdefault(r["T"], {})[r["variant"]] = r
    lines = []
    lines.append("# Microbench FA T3 — levier B : launch_bounds NT=1 minblocks=4 (124 regs)")
    lines.append("")
    lines.append(f"Géométrie prod : D=256, 24 qh, 4 kvh (gqa 6), KV q8_0, Q f32, n_kv={n_kv} (pad 256). "
                 f"{iters} iters, médiane. MTP-OFF. Launcher identique (pb auto=60 @T=1).")
    lines.append("REF = prod fa3e75c2 (lba : NT=1 minblocks=3, 162 regs, 3 blocs/SM). "
                 "LB4 = t3lba_lb4 (feaf05aa : NT=1 minblocks=4, 124 regs, 0 spill, 4 blocs/SM).")
    lines.append("**Seul T=1 (NT=1) change** ; NT>=2 (T>=2) garde minblocks=2 dans les deux -> contrôle.")
    lines.append("")
    lines.append("| T | NT | REF µs (162r) | LB4 µs (124r) | LB4 vs REF | note |")
    lines.append("|---:|---:|---:|---:|---:|---|")
    for T in sorted(byT):
        v = byT[T]
        ref = v.get("REF", {}).get("us_med")
        lb4 = v.get("LB4", {}).get("us_med")
        def pct(a, b):
            if a is None or b is None or b == 0:
                return "—"
            return f"{100*(b-a)/b:+.1f}%"
        nt = {1:1, 2:2, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3}.get(T, 1)   # ntiles: (T·6+7)/8, pack GQA=6
        note = "**levier B actif**" if T == 1 else "contrôle (NT>=2 identique)"
        lines.append(f"| {T} | {nt} | {ref:.0f} | {lb4:.0f} | {pct(lb4, ref)} | {note} |")
    lines.append("")
    lines.append("Lecture : « LB4 vs REF » = (REF − LB4)/REF : positif = LB4 plus rapide. "
                 "Seul T=1 peut gagner ; T=2..8 doivent rester ~0 (sinon = autre effet en jeu).")
    open(OUT_SUMMARY, "w").write("\n".join(lines))
    log(f"  · summary -> {OUT_SUMMARY}")

if __name__ == "__main__":
    main()
