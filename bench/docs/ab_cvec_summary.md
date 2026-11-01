# A/B applicatif e2e — levier C : load K q8_0 vectorisé 128-bit (cvec) vs prod, MTP OFF

date : 2026-10-26 20:56 UTC
service restaure : OUI
.so prod (b014d026, touche jamais) md5 : b014d026
A-cvec md5 : f89418fc  ·  B-prod md5 : b014d026

A-cvec = t3lba_lb4 + load K q8_0 vectorisé (5 LDG.128/block-pair, 123-128 regs, 0 spill).
B-prod = prod actuelle b014d026 (dual-arch, partie sm_70 = feaf05aa : ab_dual 25/10 −0,8 % = bruit).
Meme kernel T3, meme smem/launcher/pb auto : seule diff = lecture de la rangée K q8_0 (272 B).
Decode MTP-OFF = single-token = T=1 = le cas ameliore par la vectorisation du load K.
Delta = (A-cvec - B-prod)/B-prod : positif = cvec plus rapide.
Gain FA dilue par mmvq (73,6 % du busy) : FA = 14,2 % du pas decode.

## DECODE (t/s) par leg
| contexte | rep | A-cvec | B-prod | Delta |
|---|---|---|---|---|
| 32768 | 1 | 25.2 | 24.7 | +2.0% |
| 32768 | 2 | 25.18 | 24.72 | +1.9% |
| 65536 | 1 | 24.14 | 23.52 | +2.6% |
| 65536 | 2 | 24.15 | 23.53 | +2.6% |
| 98304 | 1 | 23.17 | 22.37 | +3.6% |
| 98304 | 2 | 23.18 | 22.39 | +3.5% |

## DECODE — medianes par leg
| contexte | A-cvec | B-prod | Delta |
|---|---|---|---|
| 32768 | 25.2 | 24.72 | +1.9% |
| 65536 | 24.15 | 23.53 | +2.6% |
| 98304 | 23.18 | 22.39 | +3.5% |

## WALL (s par requete, prompt+64 gen)
| contexte | A-cvec | B-prod | Delta s |
|---|---|---|---|
| 32768 | 38.0 | 37.9 | +0.1 |
| 65536 | 88.0 | 88.1 | -0.1 |
| 98304 | 153.7 | 153.8 | -0.1 |

