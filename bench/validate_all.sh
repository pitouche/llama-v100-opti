#!/usr/bin/env bash
# =============================================================================
# validate_all.sh — orchestrateur de re-measurement de la pile d'optis V100
# Voir bench/VALIDATION.md pour le protocole complet et les references a battre.
#
# Usage:
#   ./validate_all.sh L0     # audit des datas brutes (sans GPU, sans service)
#   ./validate_all.sh L1     # microbenches (V2 V3 V5 V7 V9) — machine de reference
#   ./validate_all.sh L2     # e2e A/B (V1 V4 V6 V8 V10 V12) — via superviseur, ~4-5 h
#   ./validate_all.sh L3     # runs figes nsys (V11) — seulement si L2 ecart > 3 %
#   ./validate_all.sh ALL    # L0 puis L1 puis L2 (L3 est conditionnel, lance a part)
#
# Garde-fous:
#   - 1 bench a la fois (flock /tmp/kolbo-supervisor-bench.lock, comme le superviseur)
#   - le superviseur stoppe/restore ollama-v100 lui-meme (jamais de restart manuel)
#   - GPU par UUID (gpu_uuid.py) — jamais d'index en dur
#   - les nouveaux resultats vont dans bench/validation/<date>-<tag>/ (jamais
#     par-dessus les resultats publies des repos)
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"        # .../llama-v100-opti/bench
ROOT="$(dirname "$HERE")"                     # .../llama-v100-opti
SUPER="$HERE/bench/supervisor_bench.py"
LEVEL="${1:-L0}"
TAG="$(date -u +%Y%m%d-%H%M%S)"
OUT="$HERE/validation/$TAG-$LEVEL"
mkdir -p "$OUT"

echo "[validate_all] level=$LEVEL tag=$TAG"
echo "[validate_all] sortie: $OUT"

check_so(){ # check_so <fichier-sofiles> — le .so est la et le md5 matche le suffixe
  local f="$1" md5got
  [ -f "$ROOT/$f" ] || { echo "ERREUR: .so manquant $f"; exit 2; }
  md5got=$(md5sum "$ROOT/$f" | cut -d' ' -f1 | cut -c1-8)
  case "$f" in
    *"$md5got") echo "  OK $f (md5 $md5got)" ;;
    *) echo "  AVERT: md5 $md5got != suffixe dans $f — verifier avant de bench" ;;
  esac
}

l0(){ # audit des datas brutes (pas de GPU, pas de service)
  echo "=== L0: audit des datas brutes ==="
  for o in t3-volta-small fa16-compact t1-mmsmt mmvq-clos b2048 mtp-nmax h2; do
    # h2: les bruts sont a la racine du repo (nightrun8/ab_run9*, ab_results_*, ...)
    d="$ROOT/repos/$o"; [ "$o" != h2 ] && d="$d/results"
    n=$(find "$d" -name '*.jsonl' 2>/dev/null | wc -l)
    echo "  $o: $n fichiers jsonl"
    bad=0
    for j in $(find "$d" -name '*.jsonl' 2>/dev/null); do
      python3 -c 'import json,sys; [json.loads(l) for l in open(sys.argv[1]) if l.strip()]' "$j" 2>/dev/null || { echo "  JSONL INVALIDE: $j"; bad=1; }
    done
  done
  echo "  L0 termine (jsonl invalides listes ci-dessus, sinon vide = OK)"
}

l1(){ # microbenches — pas de service, mais la V100 doit etre libre
  echo "=== L1: microbenches ==="
  check_so sofiles/libggml-cuda-t3.so.d2df07b8
  check_so sofiles/libggml-cuda-dual.so.d4954502
  check_so sofiles/libggml-cuda-dual.so.h3-20261029
  exec 9>/tmp/kolbo-supervisor-bench.lock
  flock 9
  cd "$ROOT"
  # V2: T3 kernel FA decode (rows 1/4/8 @98k) vs baseline
  python3 repos/t3-volta-small/bench/mb_fa_bench.py \
    --so sofiles/libggml-cuda-t3.so.d2df07b8 --rows 1 4 8 --ctx 98304 \
    > "$OUT/v2_t3_fa.jsonl" 2> "$OUT/v2_t3_fa.err"
  # V3: variants t3v2 (lba4/cvec/vldg) rows1 @98k
  python3 repos/t3-volta-small/bench/mb_fa_t3v2_bench.py \
    > "$OUT/v3_t3v2.jsonl" 2> "$OUT/v3_t3v2.err"
  # V5: FA16 prefill (f16) @98/131/196k
  python3 repos/fa16-compact/bench/bench_fa16_ncu.py \
    --so sofiles/libggml-cuda-dual.so.d4954502 \
    > "$OUT/v5_fa16_pf.jsonl" 2> "$OUT/v5_fa16_pf.err"
  # V7: FA16 decode — COOLDOWN 60 s d'abord (thermal, cf VALIDATION.md §1.6)
  sleep 60
  python3 repos/fa16-compact/bench/bench_fa16_decode_mb.py \
    > "$OUT/v7_fa16_dec.jsonl" 2> "$OUT/v7_fa16_dec.err"
  # V9: H3 GEMV Q6_K (dlopen du .so h3)
  python3 repos/mmvq-clos/bench/ab_mmvq2.py \
    --so sofiles/libggml-cuda-dual.so.h3-20261029 --shapes q6_f17408x5120 \
    > "$OUT/v9_h3_mmvq.jsonl" 2> "$OUT/v9_h3_mmvq.err"
  echo "  L1 termine — comparer $OUT/*.jsonl aux references (VALIDATION.md §2)"
}

l2(){ # e2e A/B — via superviseur, consomme le service
  echo "=== L2: e2e A/B (service stoppe/restaure par le bench) ==="
  exec 9>/tmp/kolbo-supervisor-bench.lock
  flock 9
  cd "$ROOT"
  # V1: T3 e2e MTP OFF (t3_bench.py)
  python3 "$SUPER" run repos/t3-volta-small/bench/t3_bench.py --detach
  # V4: H2 ON/OFF (t3_h2_bench.py, toggle GGML_CUDA_VOLTA_Q8_FATTN_TC)
  python3 "$SUPER" run repos/t3-volta-small/bench/t3_h2_bench.py --detach
  # V6: FA16 e2e prod (rollout_fa16.py, TMAX ON/OFF par drop-in)
  python3 "$SUPER" run repos/fa16-compact/bench/rollout_fa16.py --detach
  # V8: T1 e2e (t1_bench.py, 3 legs x 2 reps x 2 ctx)
  python3 "$SUPER" run repos/t1-mmsmt/bench/t1_bench.py --detach
  # V10: b2048 e2e (ab_batch_t3.py, A-b2048/B-b512 x 2 ctx x 3 reps)
  python3 "$SUPER" run repos/b2048/bench/ab_batch_t3.py --detach
  # verification post-L2: le service est bien restaure
  if curl -s -m 5 http://127.0.0.1:11434/api/version > /dev/null 2>&1; then
    echo "  service prod: RESTAURE (port 11434 repond)"
  else
    echo "  ATTENTION: le port 11434 ne repond PAS — verifier ollama-v100"
  fi
  echo "  L2 termine — comparer les logs dans $OUT et le log superviseur"
}

case "$LEVEL" in
  L0)  l0 ;;
  L1)  l1 ;;
  L2)  l2 ;;
  L3)  echo "L3: re-lancer V11 (n_max 2/3/4) — voir VALIDATION.md §2 ligne V11."
       echo "     (3 x ~17 min via superviseur, nsys CUPTI 12.9 requis)" ;;
  ALL) l0; l1; l2 ;;
  *)   echo "Usage: $0 {L0|L1|L2|L3|ALL}"; exit 1 ;;
esac
echo "[validate_all] fini — resultats dans $OUT"
