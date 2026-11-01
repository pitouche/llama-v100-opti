# Audit PRMT (byte_perm) sur les kernels CUDA de production — 2026-10-30

Objectif : évaluer si des manipulations byte-level dans les kernels réellement
utilisés pourraient être réécrites avec `PRMT` / `__byte_perm` (1 instruction au
lieu de 2-3 `SHF`+`LOP3`), et si ça vaudrait le coup.

Corpus analysé : SASS `sm_70` complet du `libggml-cuda.so` de production
(ollama-v100, `cuda_v12/`), 300 cubins extraits, 150 `sm_70`,
7310 fonctions, ~6,02 M d'instructions (`/tmp/prmt-audit/sass_all.txt`,
disassemblage nvdisasm 12.6.77 — le CUDA 13.1 système refuse le sm_70).
Croisé avec les 4 profils nsys du run PRD `prof-prd-20261030-114948`
(prefill/decode, MTP OFF/ON) pour hiérarchiser par temps GPU réel.

## 1. Réponse courte

- **Le compilateur émet déjà massivement des `PRMT`** : 103 682 occurrences
  dans 1828 fonctions sur les 7310. La question n'est pas « en mettre »,
  c'est « en mettre encore ».
- **Là où ça compte (decode `mul_mat_vec_q` Q6_K, 2,14 s / fenêtre decode ON),
  le gain extractible est ≈ 0** : les champs déquantisés sont sub-byte (Q6_K =
  4+2 bits), PRMT ne travaille qu'à la granularité de l'octet, et le
  compilateur est déjà à ~1 instruction/byte, le minimum théorique (vérifié à
  la source, §3.2).
- **Pas de réécriture PRMT utile** : les `PRMT 0x5410` existants dans
  `flash_attn_ext_f16` (prefill, 87,8 s GPU) et
  `flash_attn_ext_mma_volta_small` (decode, 556-582 ms) sont des
  packings F32→F16 pairs déjà optimaux en 1 instruction.

## 2. Ce que le compilateur fait déjà (patterns observés)

### 2.1 `PRMT 0x5410` = pack 2×F32 → 1×F16x2 (le gros morceau)
Présent en série dans :
- `flash_attn_ext_f16<256,256,32,2,...>` (LE kernel prefill du PRD) :
  **69 PRMT**, tous `0x5410`, 130 `F2F.F16.F32`, 4608 `HMMA`.
- `flash_attn_ext_mma_volta_small` (toutes instances) : jusqu'à **288 PRMT**,
  pattern `F2F.F16.F32 R9` / `F2F.F16.F32 R10` → `PRMT R16, R8, 0x5410, R9`
  (select 0x5410 : d = [Ra.b0, Ra.b1, Rb.b0, Rb.b1] = les 2 demi-mots F16
  des registres de sortie F2F empilés en 1 registre),
  immédiatement avant `STS.128`.

C'est exactement le pattern « packing avant tensor cores » qu'on cherche à
optimiser — **déjà fait, et en 1 instruction par paire**. Sur sm_70 il n'y a
pas `F2FP` (post-Volta), donc `F2F`+`PRMT` est déjà la forme la plus courte.
**Rien à gagner ici ; ne pas toucher.**

### 2.2 `dequantize_block_q8_0_f16` (prefill, 1,38 s GPU)
96 PRMT par instance, pattern par chunk :
```
PRMT R10, R2,  0x9991, RZ ;   // [b3,b3,b3,b0]
PRMT R9,  R2,  0x8880, RZ ;   // [b2,b2,b2,b2]
PRMT R10, R9,  0x5410, R10;   // entrelacement 4 valeurs
```
Dequant q8_0 → f16 vectorisé, déjà compact.

### 2.3 `mul_mat_vec_q` par type — qui a déjà des PRMT
(plus grosse instance NR, `perfn.tsv` v2 token-based)

| type | PRMT | POPC | LOP3 | SHF | lecture |
|---|---:|---:|---:|---:|---|
| Q4_0 | 4-16 | 0 | 9-37 | 38-101 | déquant par LOP3/SHF |
| Q4_1 | **0** | 0 | 9-37 | 36-98 | idem, sans PRMT |
| Q5_0 | 6-24 | 0 | 41-165 | 54-169 | 6 PRMT pour les 5 bits hauts |
| Q5_1 | **0** | 0 | 41-165 | 57-181 | idem, sans PRMT |
| Q8_0 | 4-16 | 0 | 1-5 | 34-97 | peu de LOP3 |
| Q2_K/Q3_K | 0-40 | 0 | 26-172 | 53-197 | Q3_K a des PRMT (scales) |
| Q4_K/Q5_K | **0** | 0 | 30-72 | 48-175 | **aucun PRMT** |
| **Q6_K** | **16-32** | 0 | 31-61 | 42-144 | 16 PRMT (pack + interleave scales) |
| IQ2_XXS/IQ2_XS/IQ3_XXS | 18-72 | 8-32 | 117-493 | 45-154 | POPC (codebook index) + PRMT |
| IQ4_NL | 36-144 | 0 | 18-69 | 33-97 | très PRMT-dense |

À noter : le code source ne contient `__byte_perm` **nulle part**
(à part `pxq4.cu`, format PXQ hors scope) — les 103 k PRMT sont 100 % émis
par nvcc.

## 3. Le kernel chaud du decode : `mul_mat_vec_q<(ggml_type)14, 8>` (Q6_K, NR=8)

C'est LE kernel du PRD : 2,14 s (decode ON) / 3,16 s (decode OFF) sur la
fenêtre de ~4-6 s. Corps de boucle (~114 instructions, extrait complet dans
`/tmp/prmt-audit/q6k_mmvec.sass`) :

```
FADD.FTZ 96   SHFL.BFLY 80   IADD3 68   IMAD 55   FFMA.FTZ 48
IADD3.X 38    IDP.4A 32      I2F 32     LOP3.LUT 31  LDG.U16 26
LEA 21        MOV 19         HADD2.F32 18  LDG 16   STS/LDS/STG 16+16+16
PRMT 16       SHF 15 (L×1,2,3,4 / R.HI×4,8)
```

### 3.1 Ce qui est déjà optimal
- Les **16 PRMT existants** (sélecteurs `0xba98` ×8, `0x5410` ×4, `0x8880` ×4)
  font **exactement** la rotation d'octets du mot qh entre quarts de la
  déquant Q6_K : `PRMT R104, R104, 0xba98, R103` équivaut à
  `(qh_word << 8) | (prev_word >> 24)` en **1 instruction** au lieu de
  `SHF.L 8` + `SHF.R 24` + `LOP3`. C'est la forme déjà optimale de l'accès
  glissant au mot qh (1 octet qh couvre 4 éléments, les quarts successifs sont
  décalés de 8 bits). Les `0x8880` font le répétabilité d'octets (signes/scales),
  les `0x5410` le packing pair.
- Les **32 `IDP.4A.S8.S8`** font le cœur du dot-product 4-octets — instruction
  dédiée, rien à réécrire.
- Le `FADD` (96) et `FFMA` (48) : c'est de l'arithmétique, pas du byte.

### 3.2 Les clusters candidats restants — vérifiés à la source, **aucun n'est gagnable**

Source du cœur de boucle (`vecdotq.cuh`, `vec_dot_q6_K_q8_1_impl_mmvq`) :

```cpp
const int vil = (vl >> (4*i)) & 0x0F0F0F0F;      // 4 bits bas de ql
const int vih = ((vh >> (4*i)) << 4) & 0x30303030; // 2 bits HAUTS de qh, replacés en 4-5
const int vi  = __vsubss4((vil | vih), 0x20202020);
sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc);
```

Le SASS généré (corps de boucle, x4 itérations dépliées) :
```
LOP3.LUT R46, R50, 0x30303030, RZ, 0xc0 ;   // vh & 0x30
LOP3.LUT R41, R46, 0xf0f0f0f,  R41, 0xf8 ;  // | (vl & 0x0f0f...)
IADD3    R26, R26, -0x20202020, RZ ;        // - 32
```
sans shift pour i=0 (2 LOP3 + 1 IADD3), + 1 `SHF.R 4/8` pour i=1..3.

**Point clé** : le champ « hauts bits » de qh fait **2 bits** (bits 4i..4i+1
dans l'octet qh, 1 octet qh couvrant 4 éléments) → c'est un champ
**sub-byte**. `PRMT` ne travaille qu'à la granularité de l'octet complet : elle
ne peut ni extraire 2 bits ni les positionner en 4-5, et ne peut pas non plus
expanser un nibble (il faudrait un AND avant). La décomposition du compilateur
(AND + AND + OR + SUB) est à **~1 instruction par byte de valeur déquantifiée**,
soit le minimum théorique pour des champs sub-byte. **Aucun gain PRMT possible
ici.** C'est exactement le cas où l'hypothèse « SHR/AND/OR → PRMT » échoue :
les K-quants stockent leurs 6 bits en 4+2, et le champ de 2 bits est
inaccessible à une instruction byte-oriented.

Les 15 `SHF` restants : montants 1,2,3,4,8, dont ~la moitié des `SHF.L` de
décalage inter-word (non byte-alignés, non remplaçables) et des `R.HI 4/8`
alimentant les index — là encore, sub-byte ou arithmétique.

**Bilan honnête Q6_K** : **0 instruction gagnable par PRMT** dans le corps de
boucle (et non ~5-8 comme une première lecture l'espérait). Ce qui est là est
déjà la forme minimale. Le kernel reste memory-bound (poids streamés) ;
le gain wall-clock d'une éventuelle réécriture : **0 %**.

## 4. Les autres hot kernels

| kernel (temps PRD) | PRMT | verdict |
|---|---:|---|
| `flash_attn_ext_f16<256,256,32,2>` (prefill, 87,8 s) | 69 × `0x5410` | déjà optimal (packing F32→F16x2). **Ne pas toucher.** |
| `flash_attn_ext_mma_volta_small` (decode, 0,56-0,58 s) | 28-288 | `0x5410` avant `STS.128` déjà en place. Les 98 `SHF.R.S32.HI 0x1f` (extraction du bit de signe, shift arithmétique) + LOP3 `0x33/0x3c` sont de l'arith/bit-level non byte-aligned → pas de PRMT possible. Les ~155 LOP3 « big-imm » échantillonnés sont des `AND/OR` de bits de flags → non remplaçables. |
| `Kernel2` = cutlass s884gemm_f16_128x128 (prefill, 65 s) | n/a | binaire cutlass, pas notre code. Hors scope. |
| `gated_delta_net_cuda_128x4_volta` (prefill 7,6 s / decode ON 54 ms) | 0 | **float32 pur** (SSM). Aucune manipulation byte-level à faire. Les 57 SHF sont des `<<` d'indexing. PRMT n'a rien à y chercher. |
| `convert_unary` (prefill, 3,7 s) | 1 | F32→F16 par élément : 1 `F2F.F16.F32` par élément, pas de pairing. Le pairing 2×F2F+PRMT est possible **seulement** si on réécrit le kernel en version vectorisée (2 éléments par thread) — gain = 1 instruction par 2 éléments, mais le kernel est bandwidth-bound (lecture f32 / écriture f16) → gain wall ≈ 0. |
| `dequantize_block_q6_K_grouped` (prefill, 3,2 s, x12 320) | 0 | 100 lignes, déquant q6_K → f16 du même chemin 4+2 bits : même contrainte sub-byte (§3.2), les LOP3 0xf/0x30 + SHF sont déjà la forme minimale. Kernel minuscule et memory-bound (13 µs de call moyen, la moitié en latence de launch) → gain wall ≈ 0. |
| `quantize_q8_1` (decode, 74-108 ms) | 1 | 1 `SHF.R.S32.HI 0x1f` (rounding) + LOP3 flags — **pas de byte-perm possible**, c'est de l'arith. |
| `rms_norm_f32` / `add_rms_norm_mul_f32` / `cpy_scalar` / `unary_gated` | 0-1 | float32, aucun byte-level. Hors sujet PRMT. |
| `mul_mat_vec_q_mapped_rows<(ggml_type)8, 8>` (decode ON, 136 ms) | 4 | le chemin Q8_0 « mapped rows » (shortlist MTP) : 4 PRMT déjà, le reste LOP3 de déquant q8_0 byte-aligned mais déjà en forme minimale → gain ≈ 0, kernel memory-bound. |

## 5. Synthèse par zone (vos 4 hypothèses)

| zone | conclusion |
|---|---|
| packing/unpacking FP16-BF16 | **déjà fait par nvcc** (`F2F`+`PRMT 0x5410`), 1 insn/paire, optimal sur Volta (pas de F2FP). `convert_unary` est vectorisable en 2-elt/thread mais bandwidth-bound → gain ≈ 0. |
| extraction de champs quantifiés | C'est là que le pattern SHR/AND/OR existe encore — **mais vérifié à la source : les champs sont sub-byte** (Q6_K = 4+2 bits, Q4_K/Q5_K = nibbles + bits hauts non alignés). PRMT ne travaille qu'à la granularité de l'octet → **aucun remplacement possible**. Le compilateur est déjà à ~1 instruction/byte déquantifiée, le minimum théorique. |
| réorganisation avant tensor cores | **déjà fait** (PRMT avant STS.128 dans les 2 flash-attn). |
| réduction d'instructions dans les boucles critiques | **≈ 0 extractible** dans le corps du mmvq Q6_K (structure sub-byte 4+2, §3.2). Le vrai bottleneck du decode reste le **goulot GPU global** (95,9 % busy decode OFF) documenté dans le rapport PRD, pas le nombre d'instructions entières. |

## 6. Risques / contre-indications

- **Réécrire un `__byte_perm` à la main = le compilateur le fera pareil**, et
  risque d'interférer avec son scheduling (chaînes de dépendance, réutilisation
  de registres). Sur Volta, PRMT et LOP3/SHF sont tous les deux 1 cycle
  d'émission en ALU ; la seule différence qui compte = **pression registre**
  (PRMT consomme 3-4 regs) et la **longueur de chaîne**.
- `0x1f`/`0x5`/index-based SHF ne sont **pas** remplacables par PRMT (pas
  byte-aligned). Ne pas chercher à les forcer.
- Tout patch dans `mmvq.cu`/`common.h` doit repasser par la chaîne
  build + rebuild du `.so` + re-run nsys pour valider — le coût de validation
  dépasse de loin le gain attendu (~0 %).

## 7. Recommandation

1. **Ne pas faire de patch PRMT pour un gain de bench.** Le ROI est négatif
   vu les chiffres du run 114948 (decode memory-bound, GPU 96 % busy).
2. Si on veut quand même un nettoyage : il n'y a **plus de micro-réécriture
   PRMT auto-justifiable** identifiée (le candidat Q6_K de §3.2 s'est avéré
   sans gain — champ sub-byte). La seule zone où le compilateur n'émet
   pas de PRMT malgré des données byte-alignées est `convert_unary` (F32→F16
   sans pairing) : vectorisation 2-elt/thread = 1 PRMT + 1 F2F pour 2
   éléments, mais kernel bandwidth-bound → gain ≈ 0, à faire seulement si
   fusionné dans un pass plus large.
3. Les leviers réels de perf decode restent ceux identifiés dans le rapport
   PRD : acceptation MTP (0,711), `n_max` (test 2 en préparation), et
   éventuellement la fusion du prefill (bandwidth), pas l'arithmétique
   entière.
4. `pxq4.cu` utilise déjà `__byte_perm` en source — si la famille PXQ
   revient en scope, le pattern est déjà éprouvé là-bas.

## 8. Artefacts

- `/tmp/prmt-audit/sass_all.txt` — SASS sm_70 complet (390 Mo, 6,02 M insn)
- `/tmp/prmt-audit/perfn.tsv` — v2 **fiable** (token-based ; v1 sous-comptait
  PRMT/POPC car son regex `@?!?P?\d*` mangeait le `P` de `PRMT`/`POPC`),
  7310 fonctions × 19 compteurs
- `/tmp/prmt-audit/parse2.py`, `agg2.py` — parseurs/agrégateurs
- `/tmp/prmt-audit/q6k_mmvec.sass` — corps Q6_K NR=8 (extrait)
- `/tmp/prmt-audit/q4k_mmvec.sass`, `q50` (via masks.py), `k01..k17_*.sass`
  — hot kernels extraits
- `/tmp/prmt-audit/fa_prefill.sass` — `flash_attn_ext_f16<256,256,32,2>`
- `/tmp/prmt-audit/q6k_grouped.sass` — `dequantize_block_q6_K_grouped`
- Toolchain : nvdisasm 12.6.77 extrait dans
  `/home/guile/llama-experiments/tools/nvdisasm-12.6/` (CUDA 13.1 système
  refuse le sm_70)
