#!/bin/bash
# Sweep FATTN_VS_STEP sur l'instance FA prod dominante : NT=3 (= T=4 mb_fa), q8_0, D=256.
# Prod = fa3e75c2 (STEP=128), cand = fa_step256 / fa_step512.
# Sortie : bench/mb_fa_step_sweep.jsonl
set -u
cd /home/guile/llama-experiments/bench
export LD_LIBRARY_PATH=/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib
export CUDA_VISIBLE_DEVICES=$(python3 /home/guile/llama-experiments/bench/gpu_uuid.py --cvt-ordinal 2>/dev/null || echo 0)  # UUID (jamais d'index en dur — GPU_UUID_NSYS_CUPTI_20261030.md)
H2LIB=/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib/lib
OUT=mb_fa_step_sweep.jsonl
: > "$OUT"

for so in "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so" \
          "$H2LIB/libggml-cuda-fa_step256.so" \
          "$H2LIB/libggml-cuda-fa_step512.so"; do
  v=$(basename "$so" .so | sed 's/^libggml-cuda-//')
  [ "$v" = "libggml-cuda" ] && v="prod"
  for nkv in 16384 32768 36312 65536 107351; do
    line=$(MB_FA_SO="$so" ./mb_fa t3 100 "$nkv" 2>/dev/null | grep '"T":4')
    [ -n "$line" ] && echo "{\"variant\":\"$v\",\"n_kv_req\":$nkv,$(echo "$line" | sed 's/^{//;s/}$//')}" >> "$OUT"
  done
  echo "  $v : $(grep -c "$v" "$OUT")/5 lignes"
done
echo "---"
column -s, -t < <(sed 's/[{}"]//g' "$OUT") 2>/dev/null || cat "$OUT"
