# sofiles/ — .so de chaque état prod (hors git, sha256-manifesté)

Nom = libggml-cuda-<famille>.so.<md5-court> où le md5-court (8 hex) = les 8 premiers
charactères du md5 du fichier (vérifiable par `md5sum`). La table complète :

| fichier | md5 | état prod |
|---|---|---|
| libggml-cuda-dual.so.d4954502 | d4954502 | **prod actuel** (dual sm_70+sm_61, T3+lba4+cvec+vldg+FA16) |
| libggml-cuda-dual.so.8035caf4 | 8035caf4 | T3+lba4+cvec+vldg (H2 OFF, sans FA16) |
| libggml-cuda-dual.so.4fd152f9 | 4fd152f9 | T3+lba4+cvec+vldg (déploié 28/10, dernier avant FA16) |
| libggml-cuda-dual.so.b014d026 | b014d026 | T3+lba4+cvec (déploié 26/10) |
| libggml-cuda-dual.so.feaf05aa | feaf05aa | T3+lba4 (premier déploiement prod, 25/10) |
| libggml-cuda-dual.so.fa3e75c2 | fa3e75c2 | T3+lba4 (rebuild historique 24/10) |
| libggml-cuda-t3.so.d2df07b8 | d2df07b8 | T3 de base (port anyei #153, 22-26/10) |
| libggml-cuda-h2.so.a4b6f440 | a4b6f440 | H2 seul (baseline « mission Kolbo », 18/10) |
| libggml-cuda-t1.so.ecc70a9f | ecc70a9f | T1 seul (diagnostic mmsmt) |
| libggml-cuda-t1t3.so.04f41b70 | 04f41b70 | T1+T3 (rejeté) |
| libggml-cuda-cvec.so.723c0016 | 723c0016 | T3+lba4+cvec (candidate, non-déployé) |
| libggml-cuda-vldg.so.87f85442 | 87f85442 | T3+lba4+cvec+vldg (candidate, non-déployé) |
| libggml-cuda-dual.so.h3-20261029 | (voir sha256) | H3 candidat (NO-GO) |
| libggml-cuda-dual.so.w2-20261029 | (voir sha256) | W2R4 candidat (NO-GO) |

Vérification d'intégrité : `sha256sum -c sofiles.sha256` (le manifest est généré
à la publication ; le md5 court est dans le nom de fichier).
