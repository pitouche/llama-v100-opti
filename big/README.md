# big/ — artefacts lourds (hors git, sha256-manifesté)

- nsys-prof-prd/ : traces nsys (.nsys-rep + .sqlite) de la décision n_max (30/10)
  et du profiling PRD (4 fenêtres). ~10-11 Mo par .nsys-rep. Ouvrir avec nsys
  (CUPTI 12.9 — voir nsys-cupti129/) ou sqlite3 directement.
- nsys-cupti129/libcupti.so.12.9 : binaire CUPTI 12.9 (NVIDIA, licence NVIDIA,
  non redistribuable librement) requis pour nsys sur Volta + driver 580
  (CUPTI 13.x inopérant — cf bench/docs/GPU_UUID_NSYS_CUPTI_20261030.md).
- libggml-cuda-t3v2-attempt1-0653dc50.so : artefact de build historique (t3v2).

Vérification : `sha256sum -c big.sha256`.
