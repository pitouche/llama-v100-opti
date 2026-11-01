#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_gen.py — BENCH GENERATION (decode) — tg en fonction du contexte KV existant

Objectif : à contexte DÉJÀ pré-rempli (2k → ~196k), combien de tokens/seconde
le modèle GÉNÈRE-t-il (decode) ? Le préfill n'est PAS la métrique : le contexte
est d'abord entièrement pré-rempli, la mesure ne porte que sur les tokens
générés après.

Métrique : `eval time = X ms / Y tokens (Z t/s)` du log du runner (c'est LA
ligne decode — la regex RE_EVAL exclut la ligne `prompt eval time`) →
tg = Y / (X/1000). Exclu du calcul par construction : le préfill, la
restauration de checkpoint, le chargement du modèle, le réseau/API, le temps
avant le premier token.

Même setup que le run prefill (importé de bench_direct.py — même build_env,
mêmes SERVER_ARGS, même GGUF) : Qwen3.8-27B Q6_K + MTP draft=3 + vision,
V100-32GB, KV q8_0, flash-attn, -c 204800, -b 512 -ub 512, 1 slot,
context-shift, keep 4.

Points : contexte 2k, 16k, 32k, 64k, 96k, 128k, 160k, 196k × 300 tok générés
(≥256 demandé). Le contexte est rempli par la CHAÎNE WARM (corpus B, préfixes
croissants, R_WARM) : chaque point restaure le checkpoint du précédent et ne
prefille que la diff → total préfill ≈ un seul 196k. Le KV des tokens générés
au point N est ignoré au point N+1 (restore sur le préfixe commun) → aucun
contaminant.

La métrique MTP (acceptance / mean len) est rapportée EN AUXILIAIRE dans le
tableau — aucune conclusion MTP n'est tirée de ce bench (la métrique centrale
est le débit de génération observé).

SORTIES (séparées du bench prefill, à ne pas mélanger) :
  bench_gen_results.jsonl  une ligne JSON par point (brut)
  bench_gen_summary.md     tableau + courbe tg vs contexte

USAGE :
  python3 bench_gen.py             # 8 points (~10-15 min : préfill ~8 min + gén)
  python3 bench_gen.py --quick     # 2 points (2k, 32k) — valider la mécanique
  python3 bench_gen.py --sizes 2048 65536 196608
  python3 bench_gen.py --leave-server
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_direct as B

GEN_N_PRED = 300       # tokens générés par point (spéc : ≥256)
GEN_MIN_VALID = 256    # un point avec moins de tokens générés = invalide (stop précoce)
GEN_SIZES = [2048, 16384, 32768, 65536, 98304, 131072, 163840, 196608]
GEN_QUICK = [2048, 32768]

D = os.path.dirname(os.path.abspath(__file__))
OUT_GEN_JSONL = os.path.join(D, "bench_gen_results.jsonl")
OUT_GEN_SUMMARY = os.path.join(D, "bench_gen_summary.md")


def log(msg):
    print(msg, flush=True)


def fnum(v, fmt="{:.1f}"):
    return fmt.format(v) if isinstance(v, (int, float)) else "·"


# ------------------- requête de génération -------------------------------------
def req_gen(text):
    payload = {
        "model": B._state["model_id"],
        "messages": [{"role": "user", "content": text}],
        "max_tokens": GEN_N_PRED,
        "stream": False,
        "temperature": 0.0,   # texte dégénéré déterministe (bench bande passante,
        "top_k": 1,           # pas de mesure de qualité)
        "top_p": 1.0,
        # CRITIQUE (bug du 1er run) : sans ça, le modèle émet l'EOS (fin de tour)
        # après "OK-BENCH." et stoppe à ~57 tok < 256 → point INVALID, rejou identique
        # (temp=0 déterministe). Le build b11081 copie les champs inconnus du
        # body OAI dans llama_params (server-common.cpp "Copy remaining
        # properties") et ignore_eos est une clé reconnue (server-task.cpp).
        "ignore_eos": True,
    }
    d, wall = B.http_json(f"http://127.0.0.1:{B.PORT_DIRECT}/v1/chat/completions",
                          payload, B.TIMEOUT_S)
    u = d.get("usage", {})
    return {
        "prompt_tokens": u.get("prompt_tokens"),
        "gen_tokens": u.get("completion_tokens"),
        "wall_s": round(wall, 2),
    }


def wait_decode_line(offset, budget=30.0):
    """Attend la ligne DECODE `eval time = ...` (RE_EVAL exclut `prompt eval
    time` par lookbehind) — elle s'écrit à la fin de la génération."""
    deadline = time.time() + budget
    while True:
        lines = B.server_log_chunk(offset)
        if B.RE_EVAL.search("\n".join(lines)):
            return lines
        if time.time() >= deadline:
            return lines
        time.sleep(0.2)


# ------------------- un point de mesure ------------------------------------------
def run_point(tag, c, c_prev):
    text = B.warm_text(c)
    row = None
    for attempt in (1, 2):
        t0 = time.time()
        offset = os.path.getsize(B.OUT_SERVER_LOG) if os.path.exists(B.OUT_SERVER_LOG) else 0
        api = req_gen(text)
        lines = wait_decode_line(offset)
        jr = B.extract_run(list(enumerate(lines)))
        gen_tok = jr.get("eval_tokens")
        gen_ms = jr.get("eval_ms")
        api_gen = api.get("gen_tokens") or 0
        valid = (gen_tok is not None and gen_ms
                 and gen_tok >= GEN_MIN_VALID and api_gen >= GEN_MIN_VALID)
        if valid or attempt == 2:
            break
        log(f"  · {tag}: gén={api_gen} < {GEN_MIN_VALID} (stop précoce ?) — 2e essai")
        time.sleep(1)
    tps = round(gen_tok / (gen_ms / 1000), 1) if (gen_tok and gen_ms) else None
    row = {
        "ts": int(t0), "tag": tag, "C_target_tok": c, "C_prev_tok": c_prev,
        # contexte RÉEL avant génération (log du runner) + ce qui a été restauré
        "context_real_tok": jr.get("task_n_tokens"),
        "restored_n_past": jr.get("restored_n_past"),
        # préfill : INFORMATIONNEL UNIQUEMENT (exclu de la métrique)
        "prefill_ms": jr.get("prompt_eval_ms"),
        "prefill_tokens": jr.get("prompt_eval_tokens"),
        # MÉTRIQUE : decode (generation eval time / tokens generated)
        "gen_tokens": gen_tok,
        "gen_eval_ms": gen_ms,
        "gen_tps": tps,
        "ms_per_tok": round(gen_ms / gen_tok, 3) if (gen_tok and gen_ms) else None,
        # MTP : auxiliaire, pas de conclusion
        "acceptance": jr.get("acceptance"),
        "mean_len": jr.get("mean_len"),
        "api_gen_tokens": api_gen,
        "vram_mib": B.vram_mib(),
        "api": api,
    }
    if not valid:
        row["note"] = (f"INVALID: gen={api_gen} < {GEN_MIN_VALID} (stop précoce) — "
                       "point à rejouer")
    with open(OUT_GEN_JSONL, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"  [{tag}] contexte réel={row['context_real_tok']} restore={row['restored_n_past']} "
        f"gen={gen_tok} tok en {fnum((gen_ms or 0) / 1000, '{:.2f}')}s → {tps} t/s"
        + (f" | {row['note']}" if row.get("note") else ""))
    return row


# ------------------- synthèse -----------------------------------------------------
def write_summary(rows, quick, mtp=True, volta=True):
    L = []
    L.append("# Analyse — bench GENERATION (decode tg) vs longueur du contexte")
    L.append(f"Généré le {time.strftime('%Y-%m-%d %H:%M:%S')} par bench_gen.py")
    L.append("")
    if mtp:
        L.append("Setup : Qwen3.8-27B Q6_K + MTP draft=3 + vision — V100-PCIE-32GB — "
                 "llama-server direct, KV q8_0, flash-attn, -c 204800, -b 512 -ub 512, "
                 "1 slot, context-shift, keep 4 (mêmes args que le run prefill).")
    else:
        L.append("Setup : Qwen3.8-27B Q6_K + vision — **MTP DÉSACTIVÉ** (vanilla decode) — "
                 "V100-PCIE-32GB — llama-server direct, KV q8_0, flash-attn, -c 204800, "
                 "-b 512 -ub 512, 1 slot, context-shift, keep 4 (mêmes args que le run "
                 "prefill, sans les 3 args --spec-*).")
    if not volta:
        L.append("**Kernels V100 optimisés DÉSACTIVÉS** (env : VOLTA_Q6_W4R4=0, Q5_X4=0, "
                 "Q8_FATTN_TC=0, GQA8_NCOLS2 unset, ALLREDUCE unset) — même binaire, "
                 "même GGUF, mêmes args : la seule différence est le gating env des "
                 "kernels Volta.")
    L.append("")
    L.append("Métrique : `eval time` du log (DECODE) / tokens générés (tg).")
    L.append("Exclus du calcul : préfill (prompt eval time), restauration du "
             "checkpoint, chargement du modèle, réseau/API, temps avant 1er token.")
    L.append("")
    L.append("## Tableau")
    L.append("| contexte cible | contexte réel (log) | restored | générés (tok) | gen time (s) | **tg (t/s)** | ms/tok | MTP acc (aux) | mean len (aux) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(
            f"| {r['C_target_tok']} | {r.get('context_real_tok')} | {r.get('restored_n_past')} "
            f"| {r.get('gen_tokens')} | {fnum((r.get('gen_eval_ms') or 0) / 1000, '{:.2f}')} "
            f"| **{fnum(r.get('gen_tps'), '{:.0f}')}** | {fnum(r.get('ms_per_tok'), '{:.2f}')} "
            f"| {fnum(r.get('acceptance'), '{:.3f}')} | {fnum(r.get('mean_len'), '{:.1f}')} "
            + (" |" if not r.get("note") else f" — {r['note']} |"))
    L.append("")
    L.append("## Courbe — tg (t/s) vs contexte (n_past)")
    L.append("```")
    valid = [r for r in rows if r.get("gen_tps")]
    for r in valid:
        bar = "█" * round(r["gen_tps"] / 2)
        L.append(f"{r['gen_tps']:5.0f} |{bar:<30}| contexte {r.get('context_real_tok')}")
    L.append("```")
    L.append("(1 █ = 2 t/s)")
    L.append("")
    L.append("## Réponse à la question")
    L.append("")
    ans = []
    for c in (2048, 16384, 32768, 65536, 98304, 131072, 163840, 196608):
        r = next((x for x in valid if x["C_target_tok"] == c), None)
        if r:
            ans.append(f"{c // 1024}k → **{r['gen_tps']:.0f} t/s**")
    if ans:
        L.append(f"Qwen3.8-27B Q6_K sur cette V100 — génération : {', '.join(ans)}.")
    else:
        L.append("(aucun point valide)")
    L.append("")
    L.append("## Méthode & limites")
    L.append(f"- {GEN_N_PRED} tokens générés par point (spéc : ≥{GEN_MIN_VALID}) ; un point "
             f"avec gen < {GEN_MIN_VALID} (stop précoce/EOS) est marqué INVALID et rejoué 1x automatiquement.")
    L.append("- tg est calculé sur l'ensemble des tokens générés : le contexte "
             "croît de +300 tok pendant la génération (2k → 2,3k, etc.) — la "
             "vitesse moyenne du point en reflète l'évolution interne.")
    L.append("- Le contexte est rempli par la chaîne WARM (corpus B) : le préfill "
             "de chaque point est une diff restaurée (colonne 'restored'), EXCLU "
             "de la métrique. Le KV des tokens générés au point N est ignoré au "
             "point N+1 (restore sur le préfixe commun) → pas de contaminant.")
    L.append("- Texte : corpus warm + temp=0 top_k=1 + ignore_eos (déterministe) : "
             "mesure de bande passante du decode, pas de mesure de qualité. "
             "ignore_eos est obligatoire : sans, le modèle émet l'EOS après "
             "\"OK-BENCH.\" et stoppe à ~57 tok (< 256). Après l'EOS, la suite "
             "générée est du texte dégénéré répéteur — sans incidence sur la "
             "bande passante mesurée.")
    if mtp:
        L.append("- MTP : acceptance/mean len rapportés en auxiliaire uniquement — "
                 "AUCUNE conclusion MTP n'est tirée de ce bench (sur texte dégénéré "
                 "l'acceptance est biaisée).")
    else:
        L.append("- MTP DÉSACTIVÉ pour ce run : colonne acc = — (aucune ligne draft "
                 "acceptance dans le log) ; c'est le run de comparaison vanilla.")
    L.append("- Les mesures de préfill restent dans bench_direct_results.jsonl / "
             "bench_direct_summary.md — rien n'est mélangé ici.")
    with open(OUT_GEN_SUMMARY, "w") as f:
        f.write("\n".join(L) + "\n")


# ------------------- main -----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="bench generation (decode tg) vs contexte")
    ap.add_argument("--quick", action="store_true", help="2 points (2k, 32k) — valider la mécanique")
    ap.add_argument("--sizes", type=int, nargs="*", help="cibles custom, CROISSANTES (en tokens)")
    ap.add_argument("--no-mtp", action="store_true",
                    help="DÉSACTIVER le MTP (--spec-type draft-mtp retiré) — run vanilla de comparaison")
    ap.add_argument("--no-volta", action="store_true",
                    help="DÉSACTIVER les opti kernels V100 (env Volta =0) — PAS de rebuild : le .so "
                         "de prod est env-gated à l'exécution (vérifié strings + source)")
    ap.add_argument("--binary", metavar="PATH",
                    help="utiliser UN AUTRE binaire llama-server (ex : build vanilla sans "
                         "kernels V100) — le backend CUDA (plugin GGML_BACKEND_PATH) est "
                         "pris à côté du binaire")
    ap.add_argument("--no-unload", action="store_true", help="ne pas décharger le modèle ollama avant (risque OOM)")
    ap.add_argument("--leave-server", action="store_true", help="ne pas tuer le llama-server à la fin")
    args = ap.parse_args()

    B._state["gpu_idx"], B._state["gpu_cuda_idx"], B._state["gpu_name"] = B.find_gpu()
    sizes = args.sizes or (GEN_QUICK if args.quick else GEN_SIZES)
    if not all(a <= b for a, b in zip(sizes, sizes[1:])):
        sys.exit("--sizes doit être croissant (chaîne de préfixes warm)")

    # Suffix de sortie : combine les flags (rien ne s'écrase entre runs)
    nova = bool(args.binary) or args.no_volta
    suffix = "".join(["_nomtp" if args.no_mtp else "", "_novolta" if nova else ""])
    if suffix:
        global OUT_GEN_SUMMARY
        OUT_GEN_SUMMARY = os.path.join(D, f"bench_gen_summary{suffix}.md")
        # archive du log du run précédent (timestampé — jamais d'écrasement)
        if os.path.exists(B.OUT_SERVER_LOG):
            os.replace(B.OUT_SERVER_LOG,
                       f"{B.OUT_SERVER_LOG}.{time.strftime('%Y%m%d-%H%M%S')}.bak")
        B.OUT_SERVER_LOG = os.path.join(D, f"bench_gen_server{suffix}.log")

    # --binary : binaire alternatif (ex : build vanilla b11081, sans kernels V100).
    # Le backend CUDA est un PLUGIN chargé via GGML_BACKEND_PATH (bug du 02/10 :
    # sans ça = CPU silencieux) → il faut pointer le .so de CE build, et ajouter
    # son dir bin à LD_LIBRARY_PATH (les libggml-*.so de la build y sont).
    if args.binary:
        B.LLAMA_SERVER = os.path.abspath(args.binary)
        bdir = os.path.dirname(B.LLAMA_SERVER)
        bin_backend = os.path.join(bdir, "libggml-cuda.so")
        for f in (B.LLAMA_SERVER, bin_backend):
            if not os.path.exists(f):
                sys.exit(f"--binary : introuvable {f}")
        if bdir not in B.CUDA_LIBS:
            B.CUDA_LIBS.append(bdir)
        orig_build_env = B.build_env
        def _build_env_binary(gpu_cuda_idx):
            env = orig_build_env(gpu_cuda_idx)
            # le backend CUDA est un PLUGIN : pointer le .so de CE build, sinon
            # le binaire vanilla chargerait le .so de prod (kernels V100 inclus)
            env["GGML_BACKEND_PATH"] = bin_backend
            return env
        B.build_env = _build_env_binary

    # --no-mtp : args spec retirés du SERVER_ARGS (les 3 args --spec-*)
    if args.no_mtp:
        with_val = {"--spec-type", "--spec-draft-n-max"}
        keep, skip = [], 0
        for a in B.SERVER_ARGS:
            if skip:
                skip = 0
                continue
            if a in with_val:
                skip = 1          # saute l'arg ET sa valeur
                continue
            if a == "--spec-draft-backend-sampling":
                continue          # flag sans valeur
            keep.append(a)
        B.SERVER_ARGS = keep

    # --no-volta (sans --binary) : mêmes BINAIRE/GGUF/args, mais les kernels Volta
    # optimisés du .so de prod sont désactivés par ENV (vérifié dans strings du
    # .so prod et la source llama-volta-merged) :
    #   Q6_W4R4  : Q6_K matvec rows_per_block=4 (gate : unset ou !=0 → ON) → =0
    #   Q5_X4    : Q5_K matvec (absent de ce modèle mais on le zéro pour la propreté)
    #   Q8_FATTN_TC : flash-attn q8 tensor-core (gate : ON par défaut, =0 rollback)
    #   GQA8_NCOLS2 : géométrie attention alternée (lu SEULEMENT si défini) → unset
    #   ALLREDUCE / AR_COPY_THRESHOLD : tuning multi-GPU → unset (défaut)
    # LLAMA_ARG_FIT_TARGET est conservé (layout mémoire, pas un kernel).
    # (inutile avec --binary vanilla : ce .so ne connaît pas ces variables)
    if args.no_volta and not args.binary:
        orig_build_env = B.build_env
        def _build_env_vanilla(gpu_cuda_idx):
            env = orig_build_env(gpu_cuda_idx)
            env["GGML_CUDA_VOLTA_Q6_W4R4"] = "0"
            env["GGML_CUDA_VOLTA_Q5_X4"] = "0"
            env["GGML_CUDA_VOLTA_Q8_FATTN_TC"] = "0"
            env.pop("GGML_CUDA_VOLTA_GQA8_NCOLS2", None)
            env.pop("GGML_CUDA_ALLREDUCE", None)
            env.pop("GGML_CUDA_AR_COPY_THRESHOLD", None)
            return env
        B.build_env = _build_env_vanilla
        B.OUT_SERVER_LOG = os.path.join(D, f"bench_gen_server{suffix}.log")

    log(f"Bench GEN — {B._state['gpu_name']} (CUDA dev {B._state['gpu_cuda_idx']}) — "
        f"{len(sizes)} points × {GEN_N_PRED} tok générés"
        + ("" if not args.no_mtp else " — **MTP DÉSACTIVÉ** (vanilla decode)")
        + ("" if not args.no_volta else " — **kernels V100 désactivés** (env Volta=0)")
        + ("" if not args.binary else f" — **binaire alternatif** : {os.path.basename(args.binary)}"))
    log(f"  · points : {sizes}")

    if args.no_unload:
        log("  · --no-unload : pas de déchargement ollama (risque OOM sur la V100)")
    else:
        B.unload_ollama_model()
        time.sleep(3)
    B.start_server()

    t_all = time.time()
    B._state["run_start_ts"] = int(t_all)
    with open(OUT_GEN_JSONL, "a") as f:
        f.write(json.dumps({"tag": "run_start", "ts": B._state["run_start_ts"],
                            "quick": args.quick, "sizes": sizes,
                            "gen_n_pred": GEN_N_PRED,
                            "mtp": not args.no_mtp,
                            "volta": not args.no_volta}) + "\n")

    prev = None
    for c in sizes:
        run_point(f"gen_{c}", c, prev)
        prev = c
        time.sleep(1)

    if not args.leave_server:
        B.stop_server()

    rows = []
    with open(OUT_GEN_JSONL) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if (r.get("ts", 0) >= B._state["run_start_ts"]
                    and str(r.get("tag", "")).startswith("gen_")):
                rows.append(r)
    write_summary(rows, args.quick, mtp=not args.no_mtp,
                  volta=not (args.no_volta or args.binary))
    log(f"\nFIN en {time.time() - t_all:.0f}s — {len(rows)} points")
    log(f"  synthèse : {OUT_GEN_SUMMARY}")
    log(f"  brut     : {OUT_GEN_JSONL}")
    p = B._state.get("proc")
    if args.leave_server and p and p.poll() is None:
        log(f"  llama-server LASSÉ en place (port {B.PORT_DIRECT}) — arrêt manuel : kill {p.pid}")


if __name__ == "__main__":
    main()
