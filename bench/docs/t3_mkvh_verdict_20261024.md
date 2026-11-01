# Verdict — « multi-kvhead » (240 → 960 blocks) — 24/10 17:30 — **HYPOTHÈSE RÉFUTÉE**

## L'erreur de lecture de la grille

Le grid observé en prod @T=1 est **(1, 60, 4)** = 240 blocks. L'interprétation initiale
« 60 = 1 kvhead × 60 KV-tiles, 24 qheads = 1 block » était **fausse**. La chaîne source
réelle (fattn.cu + fattn-common.cuh, `.so` prod `fa3e75c2` = variante `lba`) :

- `ggml_cuda_fattn_volta_small_geometry` : gqa_opt = (gqa=6 ≥ 2, mask, max_bias=0,
  n_kv%512=0) → **pack les heads GQA** : c=32, nz=ceil(6/32)=1, pack=ceil(6/1)=6, n·pack=6≤32 → ncols2=32, **pack=6, nrows=6**, ntiles=1.
- Grille `launch_fattn` : `(ntiles_x=1, parallel_blocks=60, ntiles_z_gqa·K->ne[2]=4)`.
  - **x=1** : 1 x-tile (les 6 qheads GQA du kvhead sont packés dans le block : `pack=6`, nrows=6, 1 warp de 8 rows).
  - **y=60 = `parallel_blocks` = split-KV** (60 de 841 KV-tiles, STEP=128) — ce n'est PAS le kvhead.
  - **z=4 = les 4 kvheads.**
- Le KV (220 Mo q8_0) est donc **déjà lu ~1×** : chaque (x-tile, kvhead) a 60 blocks qui
  se partagent 841 KV-tiles, 1 tile par block, pas de relecture entre blocks d'un même
  (x, kvhead). NCU confirmé : DRAM 53 % du pic ≈ 220 Mo en 515 µs.

## Pourquoi le « multi-kvhead ×4 » ne donne pas 960 blocks utiles

1. **Le ×4 kvhead est déjà dans le grid** (z=4). Il n'y a pas 240 blocks « à multiplier »
   par kvhead : les 4 kvheads sont déjà 4 colonnes de blocks distinctes.
2. **Dé-packer les qheads (1 block = 1 qhead au lieu de 6) ne crée pas de parallélisme
   utile** : les 6 qheads d'un même kvhead partagent le *même* KV. Les séparer en 6 blocks
   → le même KV-tile serait lu 6× (relecture) au lieu d'être packé dans 1 block
   (c'est exactement le rôle de `pack`). On passerait de 240 à 960 blocks **en relisant
   le KV 6×** → +5× le trafic L1/DRAM sur un kernel déjà à 53 % du pic DRAM.
3. **L'occupancy résidente ne bouge pas** : le plafond est **3 blocs/SM** (NCU :
   `Block Limit Registers = 3` avec 162 regs **ET** `Block Limit Shared Mem = 3` avec
   smem/block tel que 32,77/8,61→3). 960 blocks sur 80 SM = 12/SM en *total*, mais seulement
   **3 résidents à la fois** → 4 vagues au lieu de 1, pas 16 warps résidents au lieu de 12.
   Le masquage de latence (le vrai problème, 60,6 % long-scoreboard, 12 warps/SM) est
   plafonné par les regs + smem **par block**, pas par le nombre de blocks totaux.

## Ce que ça change dans la suite

- Le levier « multi-kvhead / 960 blocks » est **retiré** de la liste.
- Le vrai levier reste **l'occupancy résidente 3 blocs/SM**, plafonnée *simultanément* par
  les registres (162) et la shared memory. Pour monter à 4+ blocs/SM il faut réduire
  **les deux** (regs par thread **et** smem par block) — ou accepter 3 blocs et augmenter
  le warps/block (impossible : block=128 threads=4 warps, ceiling warps/SM=64 → 16 blocs
  max, mais regs 162×128×16 = 331 776 > 65 536 regs/SM → impossible).
- Ré-ouverture du **levier A** (parallel_blocks) : avec le grid réel, `parallel_blocks=60`
  @T=1 est déjà la config « 1 wave » (240 blocks = 80 SM × 3). Tester pb=120/240 mesure
  si 2/4 waves améliorent le masquage — à faire, mais on doit s'attendre à ce que le
  combine `flash_attn_combine_results` (déjà lancé à pb=60) pèse plus, et que le gain
  soit limité par le même plafond d'occupancy résidente.

**Décision** : passer à A (pb forcé 120/240 vs 60), puis B (launch_bounds, si A neutre).
Le « multi-kvhead » n'est **pas** une réécriture de kernel à engager.
