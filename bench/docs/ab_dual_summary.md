# A/B applicatif e2e — dual-arch (sm_70+sm_61) vs prod mono-arch (sm_70), MTP OFF

date : 2026-10-25 00:14 UTC
service restaure : OUI
.so prod (feaf05aa, touche jamais) md5 : feaf05aa
A-dual md5 : b014d026  ·  B-prod md5 : feaf05aa

A-dual = rebuild dual-arch (b014d026) · B-prod = prod actuelle (feaf05aa, lecture seule).
Meme code source (T3 + lba4 + 42d9581), memes flags : seule diff = ELF sm_61 supplementaires. -b 2048, MTP OFF.
Seule diff binaire = presence des 150 ELF sm_61 : le chemin V100 doit etre strictement identique.
Delta = (A-dual - B-prod)/B-prod : attendu ~0 (|delta| <= 0,5 % = bruit).
Verdict : NEUTRE si |delta| <= 0,5 % partout ; investiguer si > 1 %.

## DECODE (t/s) par leg
| contexte | rep | A-dual | B-prod | Delta |
|---|---|---|---|---|
| 32768 | 1 | 23.96 | 24.95 | -4.0% |
| 32768 | 2 | 24.72 | 24.94 | -0.9% |
| 65536 | 1 | 23.53 | 23.72 | -0.8% |
| 65536 | 2 | 23.49 | 23.7 | -0.9% |
| 98304 | 1 | 22.37 | 22.55 | -0.8% |
| 98304 | 2 | 22.38 | 22.55 | -0.8% |

## DECODE — medianes par leg
| contexte | A-dual | B-prod | Delta |
|---|---|---|---|
| 32768 | 24.72 | 24.95 | -0.9% |
| 65536 | 23.53 | 23.72 | -0.8% |
| 98304 | 22.38 | 22.55 | -0.8% |

## WALL (s par requete, prompt+64 gen)
| contexte | A-dual | B-prod | Delta s |
|---|---|---|---|
| 32768 | 38.3 | 37.9 | +0.4 |
| 65536 | 88.3 | 88.3 | +0.0 |
| 98304 | 154.2 | 154.2 | +0.0 |

