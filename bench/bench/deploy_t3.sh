#!/usr/bin/env bash
# deploy_t3.sh — 18/10 — replace libggml-cuda.so (a4b6f440) with libggml-cuda-t3.so
# T3 validé bench 18/10 : +13.07% decode @114688, +28.52% @196608, prefill ±0.1%,
# VRAM identique, C-t3off≈A-prod (méthodologie propre). MTP ON en prod n'interagit
# pas avec le dispatch FA decode.
# ROLLBACK : sudo cp <BAK> $SO_PROD && sudo systemctl restart ollama-v100
set -euo pipefail
SVC=ollama-v100
SO_PROD=/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so
SO_T3=/home/guile/llama-experiments/h2/cupti129-test/e2/h2lib/lib/libggml-cuda-t3.so
MODEL=qwen3.8-27b-q6-mtp-vision:latest
TS=$(date +%Y%m%d-%H%M%S)
BAK=/var/backups/llama-experiments/libggml-cuda.so.prod-a4b6f440-$TS

echo "== 0) preconditions =="
[ "$(md5sum "$SO_PROD" | cut -d' ' -f1 | cut -c1-8)" = "a4b6f440" ] || { echo "ABORT: md5 prod != a4b6f440"; exit 2; }
[ -f "$SO_T3" ] || { echo "ABORT: .so T3 manquant"; exit 2; }
echo "prod md5  : $(md5sum "$SO_PROD" | cut -d' ' -f1)"
echo "T3  md5  : $(md5sum "$SO_T3" | cut -d' ' -f1)"

echo "== 1) attendre generation idle =="
for i in $(seq 1 120); do
  TPS=$(curl -s --max-time 5 http://127.0.0.1:9444/metrics 2>/dev/null | grep '^ollama_generation_tokens_per_second{gpu="V100"}' | awk '{print $2}')
  if [ "${TPS:-x}" = "0.0" ]; then echo "idle (tps=$TPS) apres ${i}x5s"; break; fi
  echo "  tps=$TPS — j'attends (i=$i)"; sleep 5
done
TPS=$(curl -s --max-time 5 http://127.0.0.1:9444/metrics 2>/dev/null | grep '^ollama_generation_tokens_per_second{gpu="V100"}' | awk '{print $2}')
[ "${TPS:-x}" = "0.0" ] || { echo "ABORT: generation active (tps=$TPS) — rien fait."; exit 2; }

echo "== 2) backup .so prod =="
sudo cp "$SO_PROD" "$BAK"
echo "backup: $BAK"
md5sum "$BAK" | cut -d' ' -f1

echo "== 3) stop service =="
sudo systemctl stop "$SVC"
sleep 5

echo "== 4) V100 liberee =="
for i in $(seq 1 60); do
  M=$(nvidia-smi --id="$(python3 /home/guile/llama-experiments/bench/gpu_uuid.py --smi-index 2>/dev/null || echo 1)" --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "$M" -lt 1024 ]; then echo "V100 liberee ($M MiB)"; break; fi
  sleep 3
done
M=$(nvidia-smi --id="$(python3 /home/guile/llama-experiments/bench/gpu_uuid.py --smi-index 2>/dev/null || echo 1)" --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$M" -lt 1024 ] || echo "WARN: V100 pas totalement liberee ($M MiB)"

echo "== 5) installer T3 =="
sudo cp "$SO_T3" "$SO_PROD"
sudo chown root:root "$SO_PROD"
sudo chmod 755 "$SO_PROD"
echo "nouveau md5: $(md5sum "$SO_PROD" | cut -d' ' -f1)"

echo "== 6) start service =="
sudo systemctl start "$SVC"
for i in $(seq 1 60); do
  [ "$(systemctl is-active "$SVC")" = "active" ] && break
  sleep 2
done
echo "service: $(systemctl is-active "$SVC")"

echo "== 7) piege prunage (blobs) =="
sleep 8
sudo journalctl -u "$SVC" --since "3 min ago" --no-pager 2>/dev/null | grep -E "unused blobs|NOPRUNE" | tail -4 || true

echo "== 8) sanity : declencher le runner + petite generation =="
curl -s --max-time 600 http://127.0.0.1:11434/api/generate \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"Dis OK en un mot.\",\"stream\":false,\"options\":{\"num_predict\":12,\"temperature\":0}}" \
  | head -c 500
echo
echo "== 9) etat final =="
nvidia-smi --id="$(python3 /home/guile/llama-experiments/bench/gpu_uuid.py --smi-index 2>/dev/null || echo 1)" --query-gpu=memory.used --format=csv,noheader
ps aux | grep "[o]llama-v100/lib/ollama/llama-server" | grep -oE "\-b [0-9]+ -ub [0-9]+|spec-type draft-mtp|spec-draft-n-max [0-9]+" | head -4
echo "MD5 .so prod: $(md5sum "$SO_PROD" | cut -d' ' -f1)"
echo "FIN deploy_t3 — T3 actif en prod."
echo "ROLLBACK : sudo cp $BAK $SO_PROD && sudo systemctl restart $SVC"
