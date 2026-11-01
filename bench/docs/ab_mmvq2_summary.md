# AB SÛR : prod vs 42d9581 (nwarps T=1 K-quant 4->2)

date : 2026-10-17 13:09 UTC
service restauré : NON

| point | A prefill | B prefill | A decode | B decode | Δdecode | A acc/len | B acc/len |
|---|---|---|---|---|---|---|---|
| cold 16384 | 872.9 | 870.6 | 24.67 | 24.43 | -1.0% | ?/? | ?/? |
| warm 32768 | 732.1 | 729.4 | 23.12 | 21.61 | -6.5% | ?/? | ?/? |
| warm 65536 | 588.0 | 584.2 | 16.55 | 12.32 | -25.6% | ?/? | ?/? |
| warm 98304 | 461.1 | 454.2 | 11.82 | 11.49 | -2.8% | ?/? | ?/? |
| warm 131072 | 377.6 | 374.2 | 11.13 | 11.0 | -1.2% | ?/? | ?/? |
| warm 163840 | 321.1 | 319.5 | 10.79 | 10.76 | -0.3% | ?/? | ?/? |
| warm 196608 | 279.6 | 278.8 | 10.24 | 10.22 | -0.2% | ?/? | ?/? |

Interprétation : Δdecode > 0 avec prefill stable et acc/len identiques = patch gagnant et sûr.
