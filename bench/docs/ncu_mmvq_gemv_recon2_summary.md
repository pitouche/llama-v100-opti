# NCU recon2 (sections) GEMV T=4 — .so prod 8035caf4

date : 2026-10-29 10:00 UTC
service restauré : OUI

1 ligne par shape (1er lance). Cible : LECTURE DES POIDS, T=4.

| shape | dur µs | BW GB/s | DRAM % | SM % | occ theo | occ ach | lim R/S/W | regs | smem KB | WCPI | NoElig % | waves |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| q6_f17408x5120 | 148.26 | 505.67 | 59.30 | 65.96 | 31.25 | 26.64 | 5.00/5.00/16.00 | 96.00 | 6.14 | 6.34 | 32.63 | 10.88 |
| q6_f5120x17408 | 145.66 | 516.30 | 60.73 | 63.08 | 31.25 | 27.39 | 5.00/5.00/16.00 | 96.00 | 6.14 | 6.59 | 33.64 | 3.20 |
| q8_f17408x5120 | 107.33 | 128.34 | 14.54 | 63.00 | 43.75 | 39.04 | 7.00/10.00/16.00 | 72.00 | 3.07 | 9.67 | 35.29 | 15.54 |
| q8_f5120x17408 | 108.16 | 127.11 | 14.57 | 55.17 | 43.75 | 40.20 | 7.00/10.00/16.00 | 72.00 | 3.07 | 10.94 | 41.15 | 4.57 |
| q5_f17408x5120 | 137.82 | 458.02 | 53.51 | 64.45 | 31.25 | 28.00 | 5.00/5.00/16.00 | 96.00 | 3.07 | 6.80 | 34.27 | 21.76 |
| q5_f5120x17408 | 127.04 | 496.14 | 58.66 | 64.41 | 31.25 | 28.95 | 5.00/5.00/16.00 | 96.00 | 3.07 | 6.95 | 33.19 | 6.40 |

Legende : BW = MemWA Gbyte/s ; DRAM/SM = % pic ; lim R/S/W = block limit regs/smem/warps ; WCPI = Warp Cycles Per Issued Instruction (haut = latence non masquee) ; NoElig = % scheduler sans warp elig.

Artefacts : /tmp/ncu_mmvq_recon2/ncu_<shape>.full.txt (toutes sections) + ncu_mmvq_recon2.jsonl
