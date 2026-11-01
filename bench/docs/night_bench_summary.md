# NIGHT BENCH — 18/10 — MTP OFF, même .so par exp, interleave

date : 2026-10-18 02:31 UTC
reps : 2 · service restauré : OUI

## EXP1-H3 — decode/prefill H2-prod vs H3 (42d9581) — -b 2048
| ctx | rep | A-prodH2 dec | A-prodH2 pre | B-H3-42d dec | B-H3-42d pre |
|---|---|---|---|---|---|
| 114688 | 1 | — | — | — | — |
| 114688 | 2 | — | — | — | — |
| 196608 | 1 | — | — | — | — |
| 196608 | 2 | — | — | — | — |

Δ decode vs A-prodH2 (médiane par leg) :

## EXP2-BATCH — decode/prefill batch 2048/4096/8192 — .so prod
| ctx | rep | A-b2048 dec | A-b2048 pre | B-b4096 dec | B-b4096 pre | C-b8192 dec | C-b8192 pre |
|---|---|---|---|---|---|---|---|
| 114688 | 1 | — | — | — | — | — | — |
| 114688 | 2 | — | — | — | — | — | — |
| 196608 | 1 | — | — | — | — | — | — |
| 196608 | 2 | — | — | — | — | — | — |

Δ decode vs A-b2048 (médiane par leg) :

## EXP3-NT — decode/prefill -t 6 vs 12 — .so prod, -b 2048
| ctx | rep | A-t6 dec | A-t6 pre | B-t12 dec | B-t12 pre |
|---|---|---|---|---|---|
| 114688 | 1 | — | — | — | — |
| 114688 | 2 | — | — | — | — |

Δ decode vs A-t6 (médiane par leg) :

## VRAM pic par leg (MiB)
| leg | max VRAM |
|---|---|
| EXP1-H3 | 30988 |
| EXP2-BATCH | 32477 |
| EXP3-NT | 30988 |
