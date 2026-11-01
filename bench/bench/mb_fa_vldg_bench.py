#!/usr/bin/env python3
"""
mb_fa_vldg_bench.py — A/B T3 rewrite vldg (ILP phase 2 : double-buffer V) vs prod, 28/10.

2 lancements du microbench mb_fa (MÊME binaire, géométrie prod MTP-OFF, D=256 KV q8_0),
dlopen du .so choisi par MB_FA_SO, SANS env GGML_CUDA_FATTN_PB (launcher identique prod) :
  REF  : libggml-cuda.so prod (4fd152f9 = T3 + levier B + levier C)
  VLDG : libggml-cuda-vldg.so (87f85442 = REF + rewrite phase 2 : les fenêtres V du
         chunk ch (4 mots 32-bit alignés, rangées pv et pv+16) sont chargées à la fin
         de ch-1 (et le chunk 0 avant la boucle), déquant décalé d'un chunk — la
         latence LDG est en vol pendant les mmas du chunk courant. MÊMES mots que la
         baseline, MÊME arithmétique de déquant -> BIT-IDENTIQUE attendu (maxrel=0).
         SASS kernel prod exact (D256 NT=1 q8_0) : REG:128 STACK:0 (4 CTA/SM, levier B
         intact), levier C intact (17 LDG.E.128.CONSTANT.SYS), 3792 instr = baseline.

Objectif : le double-buffer V masque-t-il de la latence long-scoreboard (stall dominant
60,6 % NCU) et donne-t-il un gain mesurable @T=1 ?
Seuil GO : VLDG vs REF >= +2 % @T=1 (et pas de régression T=2..8).

Lancer via :
  python3 supervisor_bench.py run mb_fa_vldg_bench.py [iters n_kv] --detach
"""
import json
import os
import signal
import subprocess
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(D, "night_bench.py"), "__name__": "mb_fa_vldg_harness"}
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
VLDG_SO = "/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib/lib/libggml-cuda-vldg.so"
OUT_JSONL = os.path.join(D, "mb_fa_vldg_results.jsonl")
OUT_SUMMARY = os.path.join(D, "mb_fa_vldg_summary.md")

VARIANTS = [
    ("REF", PROD_SO),
    ("VLDG", VLDG_SO),
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

    log(f"BENCH MB_FA_VLDG — rewrite T3 vldg (double-buffer V phase 2, ILP) — iters={iters} n_kv={n_kv}")
    if not os.path.isfile(MB):
        sys.exit(f"binaire manquant : {MB}")
    if not os.path.isfile(VLDG_SO):
        sys.exit(f"so candidat manquant : {VLDG_SO}")
    if not port_free(PORT):
        sys.exit(f"port {PORT} occupé — abandon")

    prod_md5 = md5p(PROD_SO)
    vldg_md5 = md5p(VLDG_SO)
    log(f"  · prod : {prod_md5} · VLDG : {vldg_md5}")
    if prod_md5 != "4fd152f9":
        log("  · ATTENTION : prod != 4fd152f9 — référence inattendue (on continue)")
    if vldg_md5 == prod_md5:
        sys.exit("ABORT : VLDG == prod — rien à mesurer")

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

    # --- validation numérique : VLDG doit être BIT-IDENTIQUE à REF (mêmes mots, même
    #     arithmétique de déquant, même mma — réordonnancement de l'émission seul) ---
    maxrel_vldg = 0.0
    bit_ident = True
    if ok:
        byT = {}
        for r in results:
            byT.setdefault(r["T"], {})[r["variant"]] = r
        for T, v in byT.items():
            if len(v) < 2 or "REF" not in v or "VLDG" not in v:
                continue
            for j in range(4):
                ref = v["REF"]["out"][j]
                cv = v["VLDG"]["out"][j]
                if cv != ref:
                    bit_ident = False
                rel = abs(cv - ref) / (0.5 + abs(ref))
                maxrel_vldg = max(maxrel_vldg, rel)
        ok = maxrel_vldg < 1e-4
        log(f"  · VALIDATION : VLDG maxrel={maxrel_vldg:.2e}, bit-identique={bit_ident} (attendu True) -> {'OK' if ok else 'KO'}")

    if ok and results:
        with open(OUT_JSONL, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        write_summary(results, n_kv, iters, prod_md5, vldg_md5, bit_ident, maxrel_vldg)
    else:
        log("  · bench pas concluant — résultats non écrits")

    restore_service()
    st = svc_state()
    prod_md5b = md5p(PROD_SO)
    log(f"  · FIN VLDG-MB : service={st}, prod md5 après={prod_md5b}, results={len(results)}")
    if st != "active":
        log("  · !! service pas active — Intervention humaine requise")
        sys.exit(1)
    if prod_md5b != prod_md5:
        log("  · !! prod md5 a CHANGÉ — Intervention humaine requise")
        sys.exit(1)
    if not ok:
        sys.exit(2)

def write_summary(results, n_kv, iters, ref_md5, vldg_md5, bit_ident, maxrel_vldg):
    byT = {}
    for r in results:
        byT.setdefault(r["T"], {})[r["variant"]] = r
    def pct(a, b):
        if a is None or b is None or b == 0:
            return "—"
        return f"{100*(b-a)/b:+.1f}%"
    lines = []
    lines.append("# Microbench FA T3 — rewrite vldg (double-buffer V phase 2, ILP)")
    lines.append("")
    lines.append(f"Géométrie prod : D=256, 24 qh, 4 kvh (gqa 6), KV q8_0, Q f32, n_kv={n_kv} (pad 256). "
                 f"{iters} iters, médiane. MTP-OFF. Launcher identique (pb auto=60 @T=1).")
    lines.append(f"REF = prod ({ref_md5} = B+C) · VLDG = candidat ({vldg_md5} = phase 2 pipelinée : "
                 f"fenêtres V (4 mots 32-bit) du chunk ch+1 en vol pendant les mmas du chunk ch, "
                 f"chunk 0 chargé avant la boucle ; mêmes mots + même déquant que la baseline).")
    lines.append(f"Validation : VLDG maxrel={maxrel_vldg:.2e}, bit-identique={bit_ident} (attendu True).")
    lines.append("")
    lines.append("| T | NT | REF µs | VLDG µs | VLDG vs REF |")
    lines.append("|---:|---:|---:|---:|---:|")
    for T in sorted(byT):
        v = byT[T]
        ref = v.get("REF", {}).get("us_med")
        vldg = v.get("VLDG", {}).get("us_med")
        nt = {1:1, 2:2, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3}.get(T, 1)
        lines.append(f"| {T} | {nt} | {ref:.0f} | {vldg:.0f} | {pct(vldg, ref)} |")
    lines.append("")
    lines.append("Lecture : « VLDG vs REF » = (REF − VLDG)/REF : positif = VLDG plus rapide. "
                 "**Seuil GO : >= +2 % @T=1 sans régression T=2..8.**")
    open(OUT_SUMMARY, "w").write("\n".join(lines))
    log(f"  · summary -> {OUT_SUMMARY}")

if __name__ == "__main__":
    main()
