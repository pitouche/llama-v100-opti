# DEPouilLEMENT NIGHT BENCH (fix du summary — donnees brutes identiques)

22 points, 2 reps, MTP OFF, -np 1, KV q8_0, FA auto, .so H2 prod / H3 (42d9581).

## EXP1-H3
| ctx | A-prodH2 | B-H3-42d |
|---|---|---|
| 114688 | 17.69 | 17.59 |
| 196608 | 14.14 | 14.06 |

| ctx | A-prodH2 | B-H3-42d |  (prefill t/s)
|---|---|---|
| 114688 | 619.7 | 619.8 |
| 196608 | 343.3 | 343.3 |

decode @ 114688 vs A-prodH2: B-H3-42d: -0.6%
decode @ 196608 vs A-prodH2: B-H3-42d: -0.6%

## EXP2-BATCH
| ctx | A-b2048 | B-b4096 | C-b8192 |
|---|---|---|---|
| 114688 | 17.69 | 17.68 | - |
| 196608 | 14.13 | 14.13 | - |

| ctx | A-b2048 | B-b4096 | C-b8192 |  (prefill t/s)
|---|---|---|---|
| 114688 | 619.8 | 636.4 | - |
| 196608 | 343.6 | 354.9 | - |

decode @ 114688 vs A-b2048: B-b4096: -0.1%
decode @ 196608 vs A-b2048: B-b4096: +0.0%

## EXP3-NT
| ctx | A-t6 | B-t12 |
|---|---|---|
| 114688 | 17.69 | 17.61 |

| ctx | A-t6 | B-t12 |  (prefill t/s)
|---|---|---|
| 114688 | 619.5 | 619.1 |

decode @ 114688 vs A-t6: B-t12: -0.5%

## VRAM pic par leg (MiB)
- A-b2048: 30988
- A-prodH2: 30988
- A-t6: 30988
- B-H3-42d: 30988
- B-b4096: 32144
- B-t12: 30988
- C-b8192: 32477
