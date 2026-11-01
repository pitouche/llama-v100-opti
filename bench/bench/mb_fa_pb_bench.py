#!/usr/bin/env python3
"""
mb_fa_pb_bench.py — A/B levier A : parallel_blocks forcé (2/4 waves) sur T3, 24/10.

3 lancements du microbench mb_fa (MÊME binaire, géométrie prod MTP-OFF), via dlopen
du .so choisi par MB_FA_SO :
  REF   : prod fa3e75c2, pas d'env          -> pb=60 (heuristique 1 wave)
  PB120 : libggml-cuda-t3lba_pb.so, env=120  -> pb=120 (2 waves @T=1-2)
  PB240 : idem, env=240                     -> pb=240 (4 waves @T=1-2)

Protocole : bench/t3_pb_A_protocol.md. Lancer via
  python3 supervisor_bench.py run mb_fa_pb_bench.py --detach
Usage : mb_fa_pb_bench.py [iters] [n_kv]  (défauts 200 / 107351)
"""
import json
import os
import signal
import subprocess
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(D, "night_bench.py"), "__name__": "mb_fa_pb_harness"}
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
PB_SO = "/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib/lib/libggml-cuda-t3lba_pb.so"
OUT_JSONL = os.path.join(D, "mb_fa_pb_results.jsonl")
OUT_SUMMARY = os.path.join(D, "mb_fa_pb_summary.md")

# (nom, so, GGML_CUDA_FATTN_PB)
VARIANTS = [
    ("REF",   PROD_SO, None),
    ("PB120", PB_SO,   "120"),
    ("PB240", PB_SO,   "240"),
]

def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_kv = int(sys.argv[2]) if len(sys.argv) > 2 else 107351

    signal.signal(signal.SIGINT, ns["on_sig"])
    signal.signal(signal.SIGTERM, ns["on_sig"])

    log(f"BENCH MB_FA_PB A — pb forcé — iters={iters} n_kv={n_kv}")
    if not os.path.isfile(MB):
        sys.exit(f"binaire manquant : {MB}")
    if not os.path.isfile(PB_SO):
        sys.exit(f"so candidat manquant : {PB_SO}")
    if not port_free(PORT):
        sys.exit(f"port {PORT} occupé — abandon")

    md5 = subprocess.run(["md5sum", PROD_SO], capture_output=True, text=True).stdout.split()[0][:8]
    log(f"  · prod md5 avant : {md5}")
    if not stop_and_free():
        log("  · V100 pas libérée — on continue quand même (VRAM résiduelle ?)")

    results, ok = [], True
    for name, so, pb in VARIANTS:
        env = os.environ.copy()
        env["MB_FA_SO"] = so
        # le .so candidat n'a pas le rpath cuda_v12 du .so prod -> on fournit libcudart/cublas 12.6
        env["LD_LIBRARY_PATH"] = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12" + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        if pb:
            env["GGML_CUDA_FATTN_PB"] = pb
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
        log(f"  · {name} : rc=0 en {time.time()-t0:.0f}s, {len(rows)} lignes (so={os.path.basename(so)} pb={pb})")
        if p.stderr:
            log("    stderr : " + p.stderr.strip().replace("\n", " | ")[:400])

    # --- validation numérique : les 3 variantes doivent donner la même attention ---
    maxrel = 0.0
    if ok:
        byT = {}
        for r in results:
            byT.setdefault(r["T"], {})[r["variant"]] = r
        for T, v in byT.items():
            if len(v) < 3:
                continue
            for j in range(4):
                ref = v["REF"]["out"][j]
                for name in ("PB120", "PB240"):
                    maxrel = max(maxrel, abs(v[name]["out"][j] - ref) / (0.5 + abs(ref)))
        ok = maxrel < 1e-4
        log(f"  · VALIDATION numérique : maxrel={maxrel:.2e} (tol 1e-4, diff. d'ordre de somme) -> {'OK' if ok else 'KO'}")

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
    log(f"  · FIN A : service={st}, prod md5 après={md5b}, results={len(results)}")
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
    lines.append("# Microbench FA T3 — levier A : parallel_blocks forcé (2/4 waves)")
    lines.append("")
    lines.append(f"Géométrie prod : D=256, 24 qh, 4 kvh (gqa 6), KV q8_0, Q f32, n_kv={n_kv} (pad 256). "
                 f"{iters} iters, médiane. MTP-OFF.")
    lines.append("REF = prod fa3e75c2 (pb auto=60 @T=1-2) · PB120/PB240 = t3lba_pb (4863c97f) + env "
                 "GGML_CUDA_FATTN_PB (kernel T3/lba identique au prod, seul le pb diffère).")
    lines.append("")
    lines.append("| T | REF µs (pb60) | PB120 µs | PB240 µs | 120 vs REF | 240 vs REF |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for T in sorted(byT):
        v = byT[T]
        def us(nm):
            return v.get(nm, {}).get("us_med")
        def pct(a, b):
            if a is None or b is None or b == 0:
                return "—"
            return f"{100*(b-a)/b:+.1f}%"
        lines.append(f"| {T} | {us('REF'):.0f} | {us('PB120'):.0f} | {us('PB240'):.0f} | {pct(us('PB120'), us('REF'))} | {pct(us('PB240'), us('REF'))} |")
    lines.append("")
    lines.append("Lecture : « X vs REF » = (REF − X)/REF : positif = X plus rapide. "
                 "pb forcé : @T=1-2 ntiles_dst=4 (pb 60/120/240 = 1/2/4 waves), "
                 "@T=3-5 pb auto=40 (120/240 = 1/2 waves), @T=6-8 pb auto=20 (120/240 = 1/2 waves).")
    open(OUT_SUMMARY, "w").write("\n".join(lines))
    log(f"  · summary -> {OUT_SUMMARY}")

if __name__ == "__main__":
    main()
