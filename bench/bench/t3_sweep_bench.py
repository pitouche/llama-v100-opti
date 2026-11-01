#!/usr/bin/env python3
"""
t3_sweep_bench.py — Microbench T3 sans MTP : T=1/2 × n_kv, via dlopen du .so prod.

Contexte (22/10) : le microbench rows 1-8 a mesuré T3 vs stock à n_kv=107 351. Le cas
DOMINANT sans MTP (decode classique) est rows=1 — jamais mesuré hors 107k. Ce bench
trace T3 vs stock (wmma vec/tile) le long du n_kv pour T=1 et T=2, et valide que
T3 est actif (T3 < stock partout) : si T3 ≈ stock, c'est que le dispatch est tombé
sur stock (gate K->ne[1] % 256 == 0 non satisfait, ou gate autre).

Géométrie : D=256, 24 qh, 4 kvh (gqa 6), KV q8_0, Q f32, mask f16=0 — prod exacte.
Le .so est le même que la prod (d2df07b8) : la différence mesure le kernel.

Points n_kv (sémantique app : get_n_kv() = GGML_PAD(used, max(n_pad, 256)), donc
toujours multiple de 256 ; le n_kv non-paddé de 107 351 est conservé comme repère) :
  4096, 16384, 32768, 65536, 107520 (= 107 351 paddé), 131072, 155648, 176128.

Routes :
  t3    : T3 on,  H2 off   (GGML_CUDA_FA_NO_VOLTA_SMALL=0, GGML_CUDA_VOLTA_Q8_FATTN_TC=0)
  stock : T3 off, H2 off   (vec rows 1-2 / tile rows 3-8) — référence
H2 : OFF partout (prod, et non éligible sauf rows==4).

Iters : 200 (n_kv ≤ 32768), 100 (65 536 / 107 520), 50 (≥ 131 072) — ~10 min.
Le bench stoppe/remonte le service lui-même (contrainte README superviseur).

Lancement : cd /home/guile/llama-experiments/bench
           python3 supervisor_bench.py run t3_sweep_bench.py sweep --detach
Sorties : bench/t3_sweep_results.jsonl + bench/t3_sweep_summary.md
"""
import json
import os
import signal
import subprocess
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(D, "night_bench.py"), "__name__": "t3_sweep_harness"}
src = open(os.path.join(D, "night_bench.py")).read().split('if __name__ == "__main__":')[0]
exec(src, ns)

log = ns["log"]
restore_service = ns["restore_service"]
stop_and_free = ns["stop_and_free"]
port_free = ns["port_free"]
svc_state = ns["svc_state"]
PORT = ns["PORT"]

MB = os.path.join(D, "mb_fa")
OUT_JSONL = os.path.join(D, "t3_sweep_results.jsonl")
OUT_SUMMARY = os.path.join(D, "t3_sweep_summary.md")

# (n_kv, iters)
POINTS = [
    (4096, 200),
    (16384, 200),
    (32768, 200),
    (65536, 100),
    (107520, 100),
    (131072, 50),
    (155648, 50),
    (176128, 50),
]
T_LIST = [1, 2]
MODES = ["t3", "stock"]

def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if phase != "sweep":
        sys.exit(f"phase inconnue : {phase} (seul 'sweep' est prévu)")

    signal.signal(signal.SIGINT, ns["on_sig"])
    signal.signal(signal.SIGTERM, ns["on_sig"])

    log(f"BENCH T3_SWEEP — {len(POINTS)} points n_kv × T={T_LIST} × modes {MODES}")
    if not os.path.isfile(MB):
        sys.exit(f"binaire manquant : {MB}")
    if not port_free(PORT):
        sys.exit(f"port {PORT} occupé — abandon")

    if not stop_and_free():
        log("  · V100 pas libérée — on continue quand même (VRAM résiduelle ?)")

    results = []
    ok = True
    for n_kv, iters in POINTS:
        for mode in MODES:
            t0 = time.time()
            # mb_fa boucle T=1..8 en interne : on ne garde que T=1/2 (lignes T hors liste
            # sont jetées) ; iters imposé, n_kv imposé.
            p = subprocess.run([MB, mode, str(iters), str(n_kv)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if p.returncode != 0:
                log(f"  · n_kv={n_kv} {mode} : rc={p.returncode}\n---- stderr ----\n{p.stderr[-2500:]}")
                ok = False
                break
            rows = []
            for line in p.stdout.splitlines():
                line = line.strip()
                if line.startswith("{"):
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("T") in T_LIST:
                        r["mode"] = mode
                        rows.append(r)
            if len(rows) != len(T_LIST):
                log(f"  · n_kv={n_kv} {mode} : lignes inattendues ({len(rows)}/{len(T_LIST)})")
                ok = False
            else:
                for r in rows:
                    results.append(r)
                t3 = next((r for r in rows if r["T"] == 1 and r["mode"] == "t3"), None)
                st = next((r for r in rows if r["T"] == 1 and r["mode"] == "stock"), None)
                ratio = (f"{st['us_med']/t3['us_med']:.2f}x" if (t3 and st and t3["us_med"]) else "?")
                log(f"  · n_kv={n_kv:7d} : {len(rows)} lignes en {time.time()-t0:5.1f}s "
                    f"(T=1 : t3 vs stock = {ratio})")
            if p.stderr and "CUDA err" in p.stderr:
                log("    !! stderr CUDA : " + p.stderr.strip().replace("\n", " | ")[:400])
                ok = False
        if not ok:
            break

    # --- validation : T3 doit être strictement plus rapide que stock (sinon = pas dispatché) ---
    if ok and results:
        bad = []
        bykey = {}
        for r in results:
            bykey.setdefault((r["n_kv"], r["T"]), {})[r["mode"]] = r
        for (n_kv, T), v in sorted(bykey.items()):
            t3 = v.get("t3", {}).get("us_med")
            st = v.get("stock", {}).get("us_med")
            if t3 and st and t3 >= st:
                bad.append((n_kv, T, t3, st))
        if bad:
            log(f"  · !! T3 >= stock sur {len(bad)} points — dispatch T3 en doute : {bad[:6]}")
            ok = False
        else:
            log("  · VALIDATION : T3 < stock sur tous les points (T3 bien dispatché)")

    # --- sortie ---
    if ok and results:
        with open(OUT_JSONL, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        write_summary(results)
    else:
        log("  · sweep non concluant — pas de summary, on remonte le service")

    restore_service()
    st = svc_state()
    log(f"  · FIN sweep : service={st}, results={len(results)}")
    if st != "active":
        log("  · !! service pas active — Intervention humaine requise")
        sys.exit(1)
    if not ok:
        sys.exit(2)

def write_summary(results):
    # table propre à partir du jsonl : (T, n_kv) -> {mode: us_med}
    bykey = {}
    for r in results:
        bykey.setdefault(r["T"], {}).setdefault(r["n_kv"], {})[r["mode"]] = r["us_med"]
    lines = []
    lines.append("# Microbench T3 sans MTP — T=1/2 × n_kv (22/10)")
    lines.append("")
    lines.append("Géométrie prod : D=256, 24 qh, 4 kvh (gqa 6), KV q8_0, Q f32, mask f16=0.")
    lines.append(".so prod T3 (d2df07b8), dispatch réel via dlopen, médiane (iters 200/100/50 selon n_kv).")
    lines.append("H2 OFF partout (prod). « T3 vs stock » = (stock − T3)/stock : positif = T3 plus rapide.")
    lines.append("« vs stock@4k » = rapport des µs à 4k (T3) et à n_kv (T3) — croissance le long du ctx.")
    lines.append("")
    nks = sorted({r["n_kv"] for r in results})
    for T in sorted(bykey):
        lines.append(f"## T = {T}")
        lines.append("")
        lines.append("| n_kv | stock (µs) | T3 (µs) | T3 vs stock | T3 µs/100k keys |")
        lines.append("|---:|---:|---:|---:|---:|")
        for n_kv in nks:
            v = bykey[T].get(n_kv, {})
            st, t3 = v.get("stock"), v.get("t3")
            pct = f"{100*(st-t3)/st:+.1f}%" if (st and t3) else "—"
            per100k = f"{t3*100000.0/n_kv:.0f}" if t3 else "—"
            lines.append(f"| {n_kv} | {round(st) if st else '—'} | {round(t3) if t3 else '—'} | {pct} | {per100k} |")
        lines.append("")
    open(OUT_SUMMARY, "w").write("\n".join(lines))
    log(f"  · summary -> {OUT_SUMMARY}")

if __name__ == "__main__":
    main()
