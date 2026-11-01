#!/usr/bin/env python3
"""
mb_fa_bench.py — Microbench kernel FA rows 1-8 (wmma stock / H2 BC=8 / T3) via dlopen du .so prod.

Utilise le harness C++ `mb_fa` (dlopen libggml-cuda.so prod T3 + libggml-base.so.0,
appel direct de ggml_cuda_flash_attn_ext -> MÊME dispatch que la prod). Une route
par processus, séparée par env :
  t3    : T3 on,  H2 off  -> rows 1-8
  h2    : T3 off, H2 on   -> rows == 4 uniquement (les autres T retombent sur stock)
  stock : les deux off    -> vec (rows 1-2) / tile (rows 3-8) = le "wmma stock"

Phase `probe` (dé-risking, ~2 min) : T=4, n_kv=2048, 50 iters, validation numérique
inter-kernel (les 3 routes doivent donner la même attention).
Phase `full` (le bench) : T=1..8, n_kv=107351, 200 iters, table des µs.

Le bench stoppe/remonte le service lui-même (contrainte README superviseur).
Lancer via : python3 supervisor_bench.py run mb_fa_bench.py <probe|full> --detach
"""
import json
import os
import signal
import subprocess
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(D, "night_bench.py"), "__name__": "mb_fa_harness"}
src = open(os.path.join(D, "night_bench.py")).read().split('if __name__ == "__main__":')[0]
exec(src, ns)

log = ns["log"]
run = ns["run"]
svc_state = ns["svc_state"]
restore_service = ns["restore_service"]
stop_and_free = ns["stop_and_free"]
port_free = ns["port_free"]
PORT = ns["PORT"]

MB = os.path.join(D, "mb_fa")
OUT_JSONL = os.path.join(D, "mb_fa_results.jsonl")
OUT_SUMMARY = os.path.join(D, "mb_fa_summary.md")

def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "probe"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else (50 if phase == "probe" else 200)
    n_kv = int(sys.argv[3]) if len(sys.argv) > 3 else (2048 if phase == "probe" else 107351)

    signal.signal(signal.SIGINT, ns["on_sig"])
    signal.signal(signal.SIGTERM, ns["on_sig"])

    log(f"BENCH MB_FA {phase} — iters={iters} n_kv={n_kv}")
    if not os.path.isfile(MB):
        sys.exit(f"binaire manquant : {MB}")
    if not port_free(PORT):
        sys.exit(f"port {PORT} occupé — abandon")

    if not stop_and_free():
        log("  · V100 pas libérée — on continue quand même (VRAM résiduelle ?)")

    results = []
    ok = True
    for mode in ("t3", "h2", "stock"):
        t0 = time.time()
        p = subprocess.run([MB, mode, str(iters), str(n_kv)],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            log(f"  · {mode} : rc={p.returncode}\n---- stderr ----\n{p.stderr[-3000:]}")
            ok = False
            break
        rows = []
        for line in p.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    log(f"  · {mode} : ligne JSON illisible {line!r} ({e})")
        for r in rows:
            r["mode"] = mode
            results.append(r)
        log(f"  · {mode} : rc=0 en {time.time()-t0:.0f}s, {len(rows)} lignes")
        if p.stderr:
            log("    stderr : " + p.stderr.strip().replace("\n", " | ")[:500])

    # --- validation numérique (phase probe) ---
    if phase == "probe" and ok:
        byT = {}
        for r in results:
            byT.setdefault(r["T"], {})[r["mode"]] = r
        v = byT.get(4, {})
        if len(v) == 3:
            outs = {m: r["out"] for m, r in v.items()}
            # Tolérance RELATIVE : T3 et stock (fp16) concordent à ~5e-4 ; H2 garde ses
            # accumulateurs partiels en bf16 (pacc) -> écart ~1-2 ulp bf16 = ~1.5e-3.
            # Seuil 3e-3 relatif = « même attention », au-delà = bug de wiring.
            maxrel = 0.0
            for m in ("t3", "h2", "stock"):
                for j in range(4):
                    ref = outs["t3"][j]
                    maxrel = max(maxrel, abs(outs[m][j] - ref) / (0.5 + abs(ref)))
            ok = maxrel < 3e-3
            log(f"  · VALIDATION inter-kernel : maxrel={maxrel:.2e} (tol 3e-3 relatif) -> {'OK' if ok else 'KO'}")
            for m in ("t3", "h2", "stock"):
                log(f"      {m:6s} out={outs[m]}")

    # --- sortie ---
    if ok and results:
        with open(OUT_JSONL, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        if phase == "full":
            write_summary(results, n_kv)
    elif phase == "probe":
        log("  · probe pas concluant — pas de full, on remonte le service")

    # restauration (garantie avant sortie)
    restore_service()
    st = svc_state()
    log(f"  · FIN {phase} : service={st}, results={len(results)}")
    if st != "active":
        log("  · !! service pas active — Intervention humaine requise")
        sys.exit(1)
    if not ok:
        sys.exit(2)

def write_summary(results, n_kv):
    byT = {}
    for r in results:
        byT.setdefault(r["T"], {})[r["mode"]] = r
    lines = []
    lines.append("# Microbench kernel FA rows 1-8 — wmma stock / H2 BC=8 / T3")
    lines.append("")
    lines.append(f"Géométrie prod : D=256, 24 qh, 4 kvh (gqa 6), KV q8_0, Q f32, mask f16=0, n_kv={n_kv} (pad 256).")
    lines.append(f".so prod T3 (d2df07b8), dispatch réel, 200 iters (médiane).")
    lines.append("")
    lines.append("| T (rows) | stock (µs) | H2 (µs) | T3 (µs) | T3 vs stock | T3 vs H2 | H2 vs stock |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for T in sorted(byT):
        v = byT[T]
        st = v.get("stock", {}).get("us_med")
        h2 = v.get("h2", {}).get("us_med")
        t3 = v.get("t3", {}).get("us_med")
        def pct(a, b):
            if a is None or b is None or b == 0: return "—"
            return f"{100*(b-a)/b:+.1f}%"
        # H2 ne s'applique que sur T==4 : les autres T, mode h2 = stock (T3 off + H2 off)
        if T != 4:
            h2 = None
        lines.append(f"| {T} | {st if st is None else round(st)} | {h2 if h2 is None else round(h2)} | {t3 if t3 is None else round(t3)} | {pct(t3,st)} | {pct(t3,h2)} | {pct(h2,st) if T==4 else '—'} |")
    lines.append("")
    lines.append("Lecture : « T3 vs X » = (X - T3)/X : positif = T3 plus rapide.")
    lines.append("H2 n'est éligible que sur T=4 (rows==4) ; sur les autres T, le mode h2 est identique à stock.")
    open(OUT_SUMMARY, "w").write("\n".join(lines))
    log(f"  · summary -> {OUT_SUMMARY}")

if __name__ == "__main__":
    main()
