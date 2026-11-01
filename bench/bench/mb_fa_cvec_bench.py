#!/usr/bin/env python3
"""
mb_fa_cvec_bench.py — A/B levier C : vectorisation 128-bit du load K q8_0 (cvec), 26/10.

2 lancements du microbench mb_fa (MÊME binaire, géométrie prod MTP-OFF, D=256 KV q8_0),
dlopen du .so choisi par MB_FA_SO, SANS env GGML_CUDA_FATTN_PB (launcher identique à la
prod, pb auto) :
  REF  : libggml-cuda-t3lba_lb4.so (feaf05aa = base T3 + lba4, 17 LDG.32 / block-pair K)
  CVEC : libggml-cuda-cvec.so      (feaf05aa + load K vectorisé : 5 LDG.128 / block-pair)

Seule différence : la façon dont la rangée K q8_0 (272 B = 17 uint4, alignée 16 B) est lue
dans fattn_vs_kq_row. NCU 24/10 : global loads T3 à 5,9/32 B par secteur (LDG.32 non
vectorisé) + stall long-scoreboard 60,6 % des cycles = latence-bound. La vectorisation 128-bit
doit doubler les octets par instruction et réduire le long scoreboard.

Note référence : la prod actuelle (b014d026, dual-arch) a la MÊME partie sm_70 que feaf05aa
(ab_dual 25/10 : −0,8 % = bruit) ; REF = feaf05aa mono-arch = A/B le plus strict (même
recette build_t3v2, même 150 objets, seule diff = le load K).

Protocole : bench/t3_lbB_protocol.md. Lancer via
  python3 supervisor_bench.py run mb_fa_cvec_bench.py --detach
Usage : mb_fa_cvec_bench.py [iters] [n_kv]  (défauts 200 / 107351)
"""
import json
import os
import signal
import subprocess
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(D, "night_bench.py"), "__name__": "mb_fa_cvec_harness"}
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
REF_SO = "/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib/lib/libggml-cuda-t3lba_lb4.so"
CVEC_SO = "/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib/lib/libggml-cuda-cvec.so"
OUT_JSONL = os.path.join(D, "mb_fa_cvec_results.jsonl")
OUT_SUMMARY = os.path.join(D, "mb_fa_cvec_summary.md")

VARIANTS = [
    ("REF", REF_SO),
    ("CVEC", CVEC_SO),
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

    log(f"BENCH MB_FA_CVEC C — load K q8_0 vectorisé 128-bit — iters={iters} n_kv={n_kv}")
    if not os.path.isfile(MB):
        sys.exit(f"binaire manquant : {MB}")
    if not os.path.isfile(CVEC_SO):
        sys.exit(f"so candidat manquant : {CVEC_SO}")
    if not port_free(PORT):
        sys.exit(f"port {PORT} occupé — abandon")

    prod_md5 = md5p(PROD_SO)
    ref_md5 = md5p(REF_SO)
    cvec_md5 = md5p(CVEC_SO)
    log(f"  · prod md5 avant : {prod_md5} · REF md5 : {ref_md5} · CVEC md5 : {cvec_md5}")
    if ref_md5 != "feaf05aa":
        sys.exit(f"ABORT : REF {ref_md5} != feaf05aa — base inattendue")
    if cvec_md5 == ref_md5:
        sys.exit("ABORT : CVEC == REF — rien à mesurer")

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

    # --- validation numérique : REF et CVEC doivent donner la même attention ---
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
                maxrel = max(maxrel, abs(v["CVEC"]["out"][j] - ref) / (0.5 + abs(ref)))
        ok = maxrel < 1e-4
        log(f"  · VALIDATION numérique : maxrel={maxrel:.2e} (tol 1e-4) -> {'OK' if ok else 'KO'}")

    if ok and results:
        with open(OUT_JSONL, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        write_summary(results, n_kv, iters, ref_md5, cvec_md5)
    else:
        log("  · bench pas concluant — résultats non écrits")

    restore_service()
    st = svc_state()
    prod_md5b = md5p(PROD_SO)
    log(f"  · FIN C : service={st}, prod md5 après={prod_md5b}, results={len(results)}")
    if st != "active":
        log("  · !! service pas active — Intervention humaine requise")
        sys.exit(1)
    if prod_md5b != prod_md5:
        log("  · !! prod md5 a CHANGÉ — Intervention humaine requise")
        sys.exit(1)
    if not ok:
        sys.exit(2)

def write_summary(results, n_kv, iters, ref_md5, cvec_md5):
    byT = {}
    for r in results:
        byT.setdefault(r["T"], {})[r["variant"]] = r
    lines = []
    lines.append("# Microbench FA T3 — levier C : load K q8_0 vectorisé 128-bit (cvec)")
    lines.append("")
    lines.append(f"Géométrie prod : D=256, 24 qh, 4 kvh (gqa 6), KV q8_0, Q f32, n_kv={n_kv} (pad 256). "
                 f"{iters} iters, médiane. MTP-OFF. Launcher identique (pb auto=60 @T=1).")
    lines.append(f"REF = t3lba_lb4 ({ref_md5} : 17 LDG.32/block-pair) · CVEC = ({cvec_md5} : 5 LDG.128/block-pair).")
    lines.append("Seule diff = lecture de la rangée K q8_0 (272 B = 17 uint4 alignés 16 B). "
                 "T=1 = le cas decode prod ; T=2..8 : le load K est partagé par tile → gain attendu partout, "
                 "relatif variable.")
    lines.append("")
    lines.append("| T | NT | REF µs | CVEC µs | CVEC vs REF | note |")
    lines.append("|---:|---:|---:|---:|---:|---|")
    for T in sorted(byT):
        v = byT[T]
        ref = v.get("REF", {}).get("us_med")
        cvec = v.get("CVEC", {}).get("us_med")
        def pct(a, b):
            if a is None or b is None or b == 0:
                return "—"
            return f"{100*(b-a)/b:+.1f}%"
        nt = {1:1, 2:2, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3}.get(T, 1)   # ntiles: (T·6+7)/8, pack GQA=6
        note = "**cas decode prod**" if T == 1 else "MTP verify"
        lines.append(f"| {T} | {nt} | {ref:.0f} | {cvec:.0f} | {pct(cvec, ref)} | {note} |")
    lines.append("")
    lines.append("Lecture : « CVEC vs REF » = (REF − CVEC)/REF : positif = CVEC plus rapide. "
                 "Porte e2e : ≥ +3 % @T=1. Validation numérique maxrel < 1e-4 requise.")
    open(OUT_SUMMARY, "w").write("\n".join(lines))
    log(f"  · summary -> {OUT_SUMMARY}")

if __name__ == "__main__":
    main()
