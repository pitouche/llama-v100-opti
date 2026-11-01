#!/usr/bin/env bash
# nsys_cuda.sh — wrapper nsys 2025.5.2 + CUPTI 12.9 (la seule combo qui capture CUDA sur ce driver 580)
#
# POURQUOI CE WRAPPER (voir GPU_UUID_NSYS_CUPTI_20261030.md) :
#   - nsys tel quel charge sa libcupti.so.13.1 embarquée → CUPTI_ERROR_INVALID_DEVICE,
#     0 événement CUDA sur driver 580.178.04 (Volta V100). Reprouvé 29/10, 3× (kapp, nsystest, bench nso).
#   - NSYS_CUPTI_LIBRARY_PATH pointe nsys vers une lib CUPTI 12.9 → capture COMPLETE :
#     kernels (noms/demangling, grid/block, streams), API runtime/driver, CUDA graphs (node),
#     memcpys, syncs, OSRT. Vérifié le 30/10 sur kapp (20 KER + 27 RT nommés) et nsystest
#     (12 004 KER + 3 016 RT + graph nodes).
#   - Ne PAS mélanger avec le shim LD_PRELOAD libcupath (double inscription CUPTI → les
#     deux échouent : test T4 du 30/10).
#
# Usage (identique à nsys, on passe simplement par le wrapper) :
#   bench/nsys_cuda.sh profile --trace=cuda,osrt --osrt-threshold=2000 -o /tmp/x -f true ./app
#   bench/nsys_cuda.sh launch --session-new=S --trace=cuda,osrt --cuda-graph-trace=node <app>
#   ...puis : nsys start/stop --session=S  (les commandes de contrôle passent par nsys direct)
#
# Le wrapper :
#   1. résout le V100 par UUID (jamais d'index en dur) ; si CUDA_VISIBLE_DEVICES n'est pas
#      déjà défini, il le fixe à l'ordinal CUDA du V100 ;
#   2. fixe NSYS_CUPTI_LIBRARY_PATH vers bench/nsys-cupti129/ ;
#   3. loggue une ligne de contrôle dans le .rep (DIAGNOSTIC_EVENT) : vérifier après coup
#      que le .rep a bien chargé /.../nsys-cupti129/libcupti.so.12.9.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NSYS=/opt/nvidia/nsight-systems/2025.5.2/target-linux-x64/nsys
CUPTI_DIR="$HERE/nsys-cupti129"

[ -x "$NSYS" ] || { echo "nsys introuvable : $NSYS" >&2; exit 1; }
[ -f "$CUPTI_DIR/libcupti.so.12.9" ] || { echo "CUPTI 12.9 absent : $CUPTI_DIR/libcupti.so.12.9" >&2; exit 1; }

# sélection GPU par UUID (le helper aborte si le V100 n'est pas là)
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  V="$(python3 "$HERE/gpu_uuid.py" --cvt-ordinal 2>/dev/null || true)"
  if [ -n "$V" ]; then
    export CUDA_VISIBLE_DEVICES="$V"
  else
    echo "[nsys_cuda] ATTENTION : gpu_uuid.py indisponible — CUDA_VISIBLE_DEVICES non réglé" >&2
  fi
fi
export NSYS_CUPTI_LIBRARY_PATH="$CUPTI_DIR"

echo "[nsys_cuda] NSYS_CUPTI_LIBRARY_PATH=$CUPTI_DIR CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
exec "$NSYS" "$@"
