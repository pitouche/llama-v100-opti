# RUN-NCU-FA-T3 — Verdict NCU sur le kernel FA/T3 (24/10 12:23 UTC)

**Question posée (24/10)** : FA = 14,2 % du busy decode @90k (run9e) mais le kernel T3 ne tourne
qu'à ~45 % du roofline *lecture KV* → **au mur mémoire, ou marge compute/latence** ?

**Méthode** : NCU 2024.1.1 `--set full` (24 passes × 24 lancements) sur
`flash_attn_ext_mma_volta_small<256, NT=1, q8_0>` (T3, .so prod `fa3e75c2`), harnais `mb_fa`
route `t3` (dlopen prod = même dispatch), géométrie prod D=256/24qh/4kvh, n_kv=107 520.
Clocks verrouillés base (SM 1,23 GHz, DRAM 854 MHz → pic ~900 GB/s).
Sous superviseur, service stoppé pendant la capture, restauré après (rc=0, 24 s).
Artefacts : `ncu_fa_t3_20261024-122311.txt` (380 Ko), `ncu_fa_t3_extract.md`.

## Chiffres clés (blocs timed, T=1..8)

| T | grid | DUR (µs) | DRAM % | SM % | occTheo | No Eligible % |
|---|---|---|---|---|---|---|
| 1 | 240 blk | 515 | **53,1** | 21,2 | 18,75 | 77,7 |
| 2 | 240 blk | 567 | 49,3 | 24,5 | 18,75 | 74,5 |
| 3–5 | 160 blk | 733–874 | 37,6–31,9 | ~22 | 12,50 | ~77 |
| 6–8 | 160 blk (1x20x8) | ~1330 | ~20 | ~24 | 12,50 | ~75 |

T=1 (timed) : **464 GB/s** effectifs sur ~220 Mo de KV q8_0 lue une fois, L1 hit 84,5 %,
L2 hit 38,5 %. Stall dominant = **long scoreboard (attente L1TEX/DRAM) 8,1/13,35 cyc/warp = 60,6 %**
des cycles entre deux instructions. 162 regs/thread + 32 Ko static shared → **3 blocks/SM max**
= 12 warps actifs/SM (occ 18,75 %), 0,27 warp éligible/cycle/sched.

## VERDICT : T3 est **latence-bound (low-occupancy latency-bound)**, ni au mur DRAM, ni compute-bound

1. **Ce n'est PAS un mur mémoire** : 53 % du pic DRAM @T=1, et le pic DRAM est atteint une seule
   fois dans la boucle (1 seul chargement par block) — le reste du temps le DRAM dort.
2. **Ce n'est PAS compute-bound** : SM = 21–24 % du pic, IPC 0,89, 60 % des cycles = stall mémoire.
3. **C'est le coût de la géométrie** : 128 threads × 240 blocks (T=1), 1,5 wave max,
   12 warps/SM pour masquer une latence DRAM ~600 ns → masquage insuffisant par construction
   (regs 162/thr + shared 32 Ko plafonnent l'occupancy à 18,75 %).
   À T≥6 le dispatch bascule en (1,20,8) = 160 blocks et le temps double → la forme
   « 1 block = 1 (qhead, kvhead) » ne scale pas au-delà de T=2.

### Conséquence sur le CLOS « FA = déjà T3 »
Le CLOS **tient sur le plan du choix de kernel** (T3 reste le meilleur kernel FA connu,
+50–72 % vs stock — rien d'autre de disponible), mais il est **réqualifié** :
> T3 @ n_kv ~100k est plafonné à ~53 % du pic DRAM par son occupancy (12 warps/SM),
> pas par la bande passante.

### La piste qui en découle (réouverture légitime, une seule)
**T3 multi-kvhead** (ou split par qhead) : 4 kvhead/layer → 4× les blocks (240 → 960 à T=1),
4× les warps/SM (occupancy ~40 %), 4× le masquage en latence.
- Cible : T=1–2, qui est le régime de la prod MTP-OFF (1 fwd/tok) et le verify MTP.
- Plafond théorique : T=1 à ~85–90 % du pic DRAM = **~2,9–3,1 ms/couche vs 515 µs**
  → ~2,5× sur FA, soit **~3–4 ms/tok récupérés sur le decode @100k** = **~7–9 % du pas decode**.
  (En MTP-ON, le verify MTP utilise la même géométrie → même ordre sur ce terme-là.)
- Coût : réécriture du dispatch `fattn_mma_volta_small` + validation numérique (harnais `mb_fa`
  et probe inter-kernel déjà en place). Risque : shared mem par block (le buffer q8_0 de K/V
  est re-lu par les 4 blocks d'une même qhead — +3× la lecture L1/DRAM de KV si pas de care,
  ce qui peut manger une partie du gain → **c'est exactement ce qu'il faut mesurer avant d'écrire**).
- Si la relecture ×3 casse le gain : variante avec split par **qhead** (24 blocks) — même
  principe, relecture du KV par 24 blocks mais le KV est partagé (L2 38 % → monterait).

### Ce qui ne change pas
- Le verdict global du tour MTP-OFF (run9e) : mmvq 73,6 % = mur HBM2, inchangé.
- KV q4_0 : passe de « seule échappatoire » à « alternative au multi-kvhead ».
  Si le multi-kvhead ne donne pas ≥ 2×, le KV q4_0 (2× moins d'octets, même roofline)
  reste la brique suivante — hors périmètre actuel (risque qualité).
- Aucun bench MTP lancé (consigne user toujours en vigueur).

## Invariants post-bench (vérifié 12:39 UTC)
service `ollama-v100` active · .so prod `fa3e75c2` intact · V100 = 2721 MiB (modèle servi seul)
· A100 = 30524 MiB · util 0 % · aucun process bench résiduel.
