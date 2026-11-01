# A/B applicatif e2e — levier B : lba4 (launch_bounds NT=1 -> 4 blocs/SM) vs lba, MTP OFF

date : 2026-10-24 20:21 UTC
service restaure : OUI
.so prod (feaf05aa, touche jamais) md5 : feaf05aa
A-lba4 md5 : feaf05aa  ·  B-ref md5 : fa3e75c2

A-lba4 = nouveau prod (GGML_BACKEND_PATH=lba4) · B-ref = ancien prod (lba, backup fa3e75c2).
Meme kernel T3, meme smem/launcher/pb auto : seule diff = minblocks NT=1 (4 vs 3). -b 2048, MTP OFF.
Decode MTP-OFF = single-token = T=1 = le cas ameliore par lba4 (microbench +6,8 %).
Delta = (A-lba4 - B-ref)/B-ref : positif = lba4 plus rapide.
Gain FA dilue par mmvq (73,6 % du busy) : FA = 14,2 % -> attente ~+0,7 a +1 % du pas.

## DECODE (t/s) par leg
| contexte | rep | A-lba4 | B-ref | Delta |
|---|---|---|---|---|
| 32768 | 1 | 24.92 | 24.91 | +0.0% |
| 32768 | 2 | 24.92 | 24.93 | -0.0% |
| 65536 | 1 | 23.73 | 23.48 | +1.1% |
| 65536 | 2 | 23.73 | 23.48 | +1.1% |
| 98304 | 1 | 22.56 | 22.18 | +1.7% |
| 98304 | 2 | 22.56 | 22.19 | +1.7% |

## DECODE — medianes par leg
| contexte | A-lba4 | B-ref | Delta |
|---|---|---|---|
| 32768 | 24.92 | 24.93 | -0.0% |
| 65536 | 23.73 | 23.48 | +1.1% |
| 98304 | 22.56 | 22.19 | +1.7% |

## WALL (s par requete, prompt+64 gen)
| contexte | A-lba4 | B-ref | Delta s |
|---|---|---|---|
| 32768 | 38.1 | 37.9 | +0.2 |
| 65536 | 88.3 | 88.3 | +0.0 |
| 98304 | 154.1 | 154.2 | -0.1 |

