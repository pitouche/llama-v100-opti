#!/usr/bin/env python3
"""
mb_fa_kpf_bench.py — A/B T3 (levier ILP/prefetch K, CCTL.E.PF2) vs prod, 28/10.

3 lancements du microbench mb_fa (MÊME binaire, géométrie prod MTP-OFF, D=256 KV q8_0),
dlopen du .so choisi par MB_FA_SO, SANS env GGML_CUDA_FATTN_PB (launcher identique prod) :
  REF : libggml-cuda.so prod (4fd152f9 = T3 + levier B + levier C)
  KPF2: libggml-cuda-kpf2.so (1a8ba52e = REF + prefetch.global.L2 des 32 rangées K de
        l'itération SUIVANTE (offset k0+gridDim.y*STEP — le vrai lead time), tête de
        boucle — 3 CCTL.E.PF2/lane, 0 reg, 0 smem, kernel prod REG:128 STACK:0)
  KPF : libggml-cuda-kpf.so (7a87cf78 = REF + prefetch des rangées K de l'itération
        COURANTE (v1, offset à corriger) — référence pour isoler l'effet du lead time)
  NOB : libggml-cuda-nob.so (e19b6112 = REF sans levier B — rappel du coût du revert,
        pour ancrer la grille B × ILP ; n'attend pas de gain ici)

Le kernel prod exact (D256 NT=1 q8_0) reste à REG:128 STACK:0 dans kpf (4 CTA/SM,
levier B intact) : le prefetch ne consomme AUCUN registre (CCTL.E.PF2 = instruction
cache-only, pas de valeur retournée). Validation attendue : BIT-IDENTIQUE (cache only).

Objectif : le prefetch K masque-t-il de la latence long-scoreboard et donne-t-il un
gain mesurable @T=1 ? Seuil GO : KPF vs REF >= +2 % @T=1 (et pas de régression T=2..8).

Lancer via :
  python3 supervisor_bench.py run mb_fa_kpf_bench.py [iters n_kv] --detach
"""
import json
import os
import signal
import subprocess
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(D, "night_bench.py"), "__name__": "mb_fa_kpf_harness"}
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
KPF2_SO = "/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib/lib/libggml-cuda-kpf2.so"
KPF_SO = "/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib/lib/libggml-cuda-kpf.so"
NOB_SO = "/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib/lib/libggml-cuda-nob.so"
OUT_JSONL = os.path.join(D, "mb_fa_kpf2_results.jsonl")
OUT_SUMMARY = os.path.join(D, "mb_fa_kpf2_summary.md")

VARIANTS = [
    ("REF", PROD_SO),
    ("KPF2", KPF2_SO),
    ("KPF", KPF_SO),
    ("NOB", NOB_SO),
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

    log(f"BENCH MB_FA_KPF — levier ILP/prefetch K (CCTL.E.PF2, 0 reg) — iters={iters} n_kv={n_kv}")
    if not os.path.isfile(MB):
        sys.exit(f"binaire manquant : {MB}")
    for so in (KPF2_SO, KPF_SO, NOB_SO):
        if not os.path.isfile(so):
            sys.exit(f"so candidat manquant : {so}")
    if not port_free(PORT):
        sys.exit(f"port {PORT} occupé — abandon")

    prod_md5 = md5p(PROD_SO)
    kpf2_md5 = md5p(KPF2_SO)
    kpf_md5 = md5p(KPF_SO)
    nob_md5 = md5p(NOB_SO)
    log(f"  · prod : {prod_md5} · KPF2 : {kpf2_md5} · KPF(v1) : {kpf_md5} · NOB : {nob_md5}")
    if prod_md5 != "4fd152f9":
        log("  · ATTENTION : prod != 4fd152f9 — référence inattendue (on continue)")
    if kpf2_md5 == prod_md5 or kpf2_md5 == kpf_md5:
        sys.exit("ABORT : KPF2 == prod ou KPF2 == KPF — rien à mesurer")

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

    # --- validation numérique : KPF2/KPF doivent être BIT-IDENTIQUES à REF (cache only, 0 reg) ;
    #     NOB = recette h2lib -use_fast_math -> drift f16 attendu (5e-5), tol 1e-4 ---
    maxrel_kpf2 = 0.0
    maxrel_kpf = 0.0
    maxrel_nob = 0.0
    bit_ident = True
    if ok:
        byT = {}
        for r in results:
            byT.setdefault(r["T"], {})[r["variant"]] = r
        for T, v in byT.items():
            if len(v) < 2 or "REF" not in v:
                continue
            for name, acc in (("KPF2", "kpf2"), ("KPF", "kpf"), ("NOB", "nob")):
                if name not in v:
                    continue
                for j in range(4):
                    ref = v["REF"]["out"][j]
                    cv = v[name]["out"][j]
                    if cv != ref and name in ("KPF2", "KPF"):
                        bit_ident = False
                    rel = abs(cv - ref) / (0.5 + abs(ref))
                    if acc == "kpf2":
                        maxrel_kpf2 = max(maxrel_kpf2, rel)
                    elif acc == "kpf":
                        maxrel_kpf = max(maxrel_kpf, rel)
                    else:
                        maxrel_nob = max(maxrel_nob, rel)
        ok = maxrel_kpf2 < 1e-4 and maxrel_kpf < 1e-4 and maxrel_nob < 1e-4
        log(f"  · VALIDATION : KPF2 maxrel={maxrel_kpf2:.2e} KPF maxrel={maxrel_kpf:.2e} "
            f"bit-identique={bit_ident} ; NOB maxrel={maxrel_nob:.2e} (tol 1e-4) -> {'OK' if ok else 'KO'}")

    if ok and results:
        with open(OUT_JSONL, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        write_summary(results, n_kv, iters, prod_md5, kpf2_md5, kpf_md5, nob_md5, bit_ident, maxrel_kpf2, maxrel_kpf, maxrel_nob)
    else:
        log("  · bench pas concluant — résultats non écrits")

    restore_service()
    st = svc_state()
    prod_md5b = md5p(PROD_SO)
    log(f"  · FIN KPF-MB : service={st}, prod md5 après={prod_md5b}, results={len(results)}")
    if st != "active":
        log("  · !! service pas active — Intervention humaine requise")
        sys.exit(1)
    if prod_md5b != prod_md5:
        log("  · !! prod md5 a CHANGÉ — Intervention humaine requise")
        sys.exit(1)
    if not ok:
        sys.exit(2)

def write_summary(results, n_kv, iters, ref_md5, kpf2_md5, kpf_md5, nob_md5, bit_ident, maxrel_kpf2, maxrel_kpf, maxrel_nob):
    byT = {}
    for r in results:
        byT.setdefault(r["T"], {})[r["variant"]] = r
    def pct(a, b):
        if a is None or b is None or b == 0:
            return "—"
        return f"{100*(b-a)/b:+.1f}%"
    lines = []
    lines.append("# Microbench FA T3 — levier ILP/prefetch K : prefetch.global.L2 (CCTL.E.PF2)")
    lines.append("")
    lines.append(f"Géométrie prod : D=256, 24 qh, 4 kvh (gqa 6), KV q8_0, Q f32, n_kv={n_kv} (pad 256). "
                 f"{iters} iters, médiane. MTP-OFF. Launcher identique (pb auto=60 @T=1).")
    lines.append(f"REF = prod ({ref_md5} = B+C) · KPF2 = candidat offset corrigé ({kpf2_md5} = prefetch des rangées K de "
                 f"l'itération SUIVANTE, 3 CCTL.E.PF2/lane, 0 reg, 0 smem, kernel prod REG:128 STACK:0) · "
                 f"KPF(v1) = {kpf_md5} (prefetch des rangées COURANTES — référence du lead time) · NOB = {nob_md5} (sans B).")
    lines.append(f"Validation : KPF2 maxrel={maxrel_kpf2:.2e}, KPF maxrel={maxrel_kpf:.2e}, bit-identique={bit_ident} "
                 f"(attendu True — cache only) ; NOB maxrel={maxrel_nob:.2e} (drift -use_fast_math, tol 1e-4).")
    lines.append("")
    lines.append("| T | NT | REF µs | KPF2 µs | KPF2 vs REF | KPF(v1) µs | KPF vs REF | NOB µs | NOB vs REF |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for T in sorted(byT):
        v = byT[T]
        ref = v.get("REF", {}).get("us_med")
        kpf2 = v.get("KPF2", {}).get("us_med")
        kpf = v.get("KPF", {}).get("us_med")
        nob = v.get("NOB", {}).get("us_med")
        nt = {1:1, 2:2, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3}.get(T, 1)
        lines.append(f"| {T} | {nt} | {ref:.0f} | {kpf2:.0f} | {pct(kpf2, ref)} | {kpf:.0f} | {pct(kpf, ref)} | {nob:.0f} | {pct(nob, ref)} |")
    lines.append("")
    lines.append("Lecture : « X vs REF » = (REF − X)/REF : positif = X plus rapide. **Seuil GO : >= +2 % @T=1 sans "
                 "régression T=2..8.** KPF(v1) = prefetch sans lead time (lignes de l'itération courante, déjà en vol) ; "
                 "KPF2 = prefetch avec lead time = 1 itération (lignes de l'itération suivante). "
                 "NOB = coût du revert du levier B (−7,1 % @T=1, mesuré 28/10).")
    open(OUT_SUMMARY, "w").write("\n".join(lines))
    log(f"  · summary -> {OUT_SUMMARY}")

if __name__ == "__main__":
    main()
