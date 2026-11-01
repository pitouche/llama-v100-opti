#!/usr/bin/env python3
"""
mb_fa_t3v2_bench.py — A/B T3 v2 (levier D) : vectorisation 128-bit du load V q8_0, 27/10.

2 lancements du microbench mb_fa (MÊME binaire, géométrie prod MTP-OFF, D=256 KV q8_0),
dlopen du .so choisi par MB_FA_SO, SANS env GGML_CUDA_FATTN_PB (launcher identique prod) :
  REF  : libggml-cuda.so prod (4fd152f9 = T3 + lba4 + cvec : load V en LDG.32 dispersés)
  V2   : libggml-cuda-dual.so (candidat : load V q8_0 = 1-3 LDG.128 alignés + extraction octets)

Seule différence : la façon dont la fenêtre V q8_0 (scale d + quants de la lane, DL=16 B)
est lue dans fattn_vs_v_row (phase 2). La déquantisation est la même (mêmes octets), donc
le résultat attention doit être BIT-IDENTIQUE (maxrel = 0 attendu, tolérance pratique 1e-4).

Lancer via :
  python3 supervisor_bench.py run mb_fa_t3v2_bench.py [iters n_kv] --detach
"""
import json
import os
import signal
import subprocess
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(D, "night_bench.py"), "__name__": "mb_fa_t3v2_harness"}
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
V2_SO = "/home/guile/llama-experiments/v100/dual-arch/lib/libggml-cuda-dual.so"
OUT_JSONL = os.path.join(D, "mb_fa_t3v2_results.jsonl")
OUT_SUMMARY = os.path.join(D, "mb_fa_t3v2_summary.md")

VARIANTS = [
    ("REF", PROD_SO),
    ("V2", V2_SO),
]

def md5p(p):
    try:
        return subprocess.run(["md5sum", p], capture_output=True, text=True).stdout.split()[0][:8]
    except Exception:
        return "?"

def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_kv = int(sys.argv[2]) if len(sys.argv) > 2 else 107351

    signal.signal(signal.SIGINT, ns["on_sig"])
    signal.signal(signal.SIGTERM, ns["on_sig"])

    log(f"BENCH MB_FA_T3V2 D — load V q8_0 vectorisé 128-bit — iters={iters} n_kv={n_kv}")
    if not os.path.isfile(MB):
        sys.exit(f"binaire manquant : {MB}")
    if not os.path.isfile(V2_SO):
        sys.exit(f"so candidat manquant : {V2_SO}")
    if not port_free(PORT):
        sys.exit(f"port {PORT} occupé — abandon")

    prod_md5 = md5p(PROD_SO)
    v2_md5 = md5p(V2_SO)
    log(f"  · prod md5 avant : {prod_md5} · V2 md5 : {v2_md5}")
    if prod_md5 != "4fd152f9":
        log("  · ATTENTION : prod != 4fd152f9 — référence inattendue (on continue)")
    if v2_md5 == prod_md5:
        sys.exit("ABORT : V2 == prod — rien à mesurer")

    if not stop_and_free():
        log("  · V100 pas libérée — on continue quand même (VRAM résiduelle ?)")

    results, ok = [], True
    for name, so in VARIANTS:
        env = os.environ.copy()
        env["MB_FA_SO"] = so
        env["LD_LIBRARY_PATH"] = ("/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12"
                                  + os.pathsep + env.get("LD_LIBRARY_PATH", ""))
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

    # --- validation numérique : REF et V2 doivent donner la même attention (bit-identique attendu) ---
    maxrel = 0.0
    bit_ident = True
    if ok:
        byT = {}
        for r in results:
            byT.setdefault(r["T"], {})[r["variant"]] = r
        for T, v in byT.items():
            if len(v) < 2:
                continue
            for j in range(4):
                ref = v["REF"]["out"][j]
                cv = v["V2"]["out"][j]
                if cv != ref:
                    bit_ident = False
                maxrel = max(maxrel, abs(cv - ref) / (0.5 + abs(ref)))
        ok = maxrel < 1e-4
        log(f"  · VALIDATION numérique : maxrel={maxrel:.2e} (tol 1e-4) -> {'OK' if ok else 'KO'} ; "
            f"bit-identique = {bit_ident}")

    if ok and results:
        with open(OUT_JSONL, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        write_summary(results, n_kv, iters, prod_md5, v2_md5, bit_ident)
    else:
        log("  · bench pas concluant — résultats non écrits")

    restore_service()
    st = svc_state()
    prod_md5b = md5p(PROD_SO)
    log(f"  · FIN T3V2-MB : service={st}, prod md5 après={prod_md5b}, results={len(results)}")
    if st != "active":
        log("  · !! service pas active — Intervention humaine requise")
        sys.exit(1)
    if prod_md5b != prod_md5:
        log("  · !! prod md5 a CHANGÉ — Intervention humaine requise")
        sys.exit(1)
    if not ok:
        sys.exit(2)

def write_summary(results, n_kv, iters, ref_md5, v2_md5, bit_ident):
    byT = {}
    for r in results:
        byT.setdefault(r["T"], {})[r["variant"]] = r
    lines = []
    lines.append("# Microbench FA T3 — T3 v2 (levier D) : load V q8_0 vectorisé 128-bit")
    lines.append("")
    lines.append(f"Géométrie prod : D=256, 24 qh, 4 kvh (gqa 6), KV q8_0, Q f32, n_kv={n_kv} (pad 256). "
                 f"{iters} iters, médiane. MTP-OFF. Launcher identique (pb auto=60 @T=1).")
    lines.append(f"REF = prod ({ref_md5}) · V2 = candidat dual ({v2_md5}). "
                 f"Seule diff = lecture de la fenêtre V q8_0 (scale + quants de la lane) dans fattn_vs_v_row.")
    lines.append(f"Validation : maxrel < 1e-4 requis ; bit-identique = {bit_ident}.")
    lines.append("")
    lines.append("| T | NT | REF µs | V2 µs | V2 vs REF | note |")
    lines.append("|---:|---:|---:|---:|---:|---|")
    for T in sorted(byT):
        v = byT[T]
        ref = v.get("REF", {}).get("us_med")
        c2 = v.get("V2", {}).get("us_med")
        def pct(a, b):
            if a is None or b is None or b == 0:
                return "—"
            return f"{100*(b-a)/b:+.1f}%"
        nt = {1:1, 2:2, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3}.get(T, 1)
        note = "**cas decode prod**" if T == 1 else "MTP verify"
        lines.append(f"| {T} | {nt} | {ref:.0f} | {c2:.0f} | {pct(c2, ref)} | {note} |")
    lines.append("")
    lines.append("Lecture : « V2 vs REF » = (REF − V2)/REF : positif = V2 plus rapide. "
                 "Porte e2e : gain @T=1 ≥ +2 % pour déclencher l'A/B applicatif.")
    open(OUT_SUMMARY, "w").write("\n".join(lines))
    log(f"  · summary -> {OUT_SUMMARY}")

if __name__ == "__main__":
    main()
