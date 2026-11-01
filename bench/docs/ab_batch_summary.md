# AB BATCH PREFILL (-b/-ub 512/1024/2048/4096) — MTP OFF, même .so prod

date : 2026-10-17 20:14 UTC
service restauré : NON

## PREFILL (prompt t/s) — la cible du levier -b
| point | A b512 | B b1024 | C b2048 | D b4096 |
|---|---|---|---|---|
| cold 16384 | 874.4 | 1003.2 | 1047.2 | 1068.5 |
| warm 32768 | 809.0 | 905.8 | 946.5 | 965.3 |
| warm 65536 | 684.9 | 753.2 | 781.7 | 797.6 |
| warm 98304 | 589.0 | 642.9 | 665.0 | 682.2 |
| warm 114688 | 546.8 | 599.0 | 619.8 | 635.8 |
| warm 131072 | 516.5 | 562.1 | 581.4 | 597.4 |
| warm 163840 | 460.6 | 499.4 | 515.7 | 530.5 |
| warm 196608 | 418.7 | 450.5 | 464.2 | 477.7 |

## DECODE (t/s) — contrôle, doit rester stable (batch 1, -b n'a pas d'effet direct)
| point | A b512 | B b1024 | C b2048 | D b4096 |
|---|---|---|---|---|
| cold 16384 | 24.68 | 24.68 | 24.68 | 24.66 |
| warm 32768 | 23.06 | 23.09 | 23.08 | 23.08 |
| warm 65536 | 18.41 | 20.32 | 20.39 | 20.41 |
| warm 98304 | 12.06 | 17.91 | 18.25 | 16.52 |
| warm 114688 | 12.13 | 16.95 | 17.1 | 16.33 |
| warm 131072 | 11.15 | 16.27 | 16.3 | 14.77 |
| warm 163840 | 11.02 | 15.28 | 15.09 | 14.53 |
| warm 196608 | 10.42 | 14.13 | 14.13 | 13.74 |

Gain prefill fiable si > ~5% et constant sur plusieurs points. Chaque leg = serveur frais (interleave).
