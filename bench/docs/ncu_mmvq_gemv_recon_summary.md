# NCU recon GEMV T=4 — .so prod 8035caf4 (mb_mmvq, 5 lances/shape)

date : 2026-10-29 09:13 UTC
service restauré : OUI

Cible : LECTURE DES POIDS (GEMV). T=4 (verify MTP). 6 shapes FFN réelles 27B.

| shape | type | grid | block | dur µs | DRAM_r MB | BW GB/s | %pic DRAM | occ % | L2miss MB | théo poids MB | DRAM/théo | GHz |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q5_f17408x5120 | Q5_K | 8704,1,1 | 32,4,1 | 137 | 61.4 | 0 | 53.3 | 28.1 | 56.5 | 61.3 | 1.00 | 1.23 |
| q5_f5120x17408 | Q5_K | 2560,1,1 | 32,4,1 | 126 | 61.4 | 1 | 58.8 | 29.0 | 97.2 | 61.3 | 1.00 | 1.22 |
| q6_f17408x5120 | Q6_K | 4352,1,1 | 32,4,1 | 149 | 73.2 | 1 | 59.0 | 26.8 | 73.1 | 73.1 | 1.00 | 1.23 |
| q6_f5120x17408 | Q6_K | 1280,1,1 | 32,4,1 | 145 | 73.6 | 1 | 61.0 | 27.5 | 111.3 | 73.1 | 1.01 | 1.22 |
| q8_f17408x5120 | Q8_0 | 8704,1,1 | 32,4,1 | 108 | 11.9 | 0 | 14.4 | 39.1 | 52.9 | 94.7 | 0.13 | 1.22 |
| q8_f5120x17408 | Q8_0 | 2560,1,1 | 32,4,1 | 108 | 11.9 | 0 | 14.3 | 40.2 | 143.0 | 94.7 | 0.13 | 1.22 |

Référence : pic HBM2 V100 ≈ 900 GB/s. Durée non profilée (mb_mmvq) : Q6_K ~133-135 µs, Q8_0 ~100 µs, Q5_K ~115-128 µs (T=4).
Le ratio durée_NCU / durée_non_profilée = surcoût du profiler (≈7 % sur le KOLBO E1).

Artefacts : /tmp/ncu_mmvq_recon/ncu_<shape>.raw.txt + ncu_mmvq_recon.jsonl
