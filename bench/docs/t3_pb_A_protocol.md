# Protocole A/B — levier A : parallel_blocks forcé (2/4 waves) — 24/10

**Décision user** : A et B séparés. A d'abord : pb=120 puis pb=240, témoin = config actuelle.
**Question** : en latence-bound (NCU : 60,6 % long-scoreboard, 1 wave = 240 slots), des
vagues additionnelles (2 puis 4) améliorent-elles le masquage de latence ?

## Variantes (3 lancements `mb_fa`, même binaire, même géométrie, MTP-OFF)

| variante | .so (MB_FA_SO) | env | pb effectif attendu @T=1 |
|---|---|---|---|
| REF | prod `fa3e75c2` (ollama-v100) | — | 60 (heuristique 1 wave) |
| PB120 | `h2lib/lib/libggml-cuda-t3lba_pb.so` (`4863c97f`) | `GGML_CUDA_FATTN_PB=120` | 120 (2 waves) |
| PB240 | idem | `GGML_CUDA_FATTN_PB=240` | 240 (4 waves) |

- pb forcé = override **exact** dans `launch_fattn` (patch bench-only de `fattn-common.cuh`,
  absent de la prod) : `pb = min(max(pbf,1), ntiles_KV)` ; `ntiles_KV = ceil(n_kv/STEP) = 841`
  @107520 → 120/240 toujours valides.
- @T=1..2 : ntiles_dst=4 → pb 60/120/240 = 1/2/4 waves (240/480/960 blocks).
- @T=3..5 : ntiles_dst=4 (col2=128, pack=6) → pb 40 auto ; 120/240 = 1/2 waves (480/960 blk).
- @T=6..8 : ntiles_dst=8 (col2=64, pack=3) → pb 20 auto ; 120/240 = 1/2 waves (960/1920 blk).
- Combine : `flash_attn_combine_results` relancé à pb>1 (déjà le cas REF @60) → même coût fixe.
- **Risque connu** : pb>60 à T=1 = 480/960 blocks > 240 slots → la 2ᵉ/3ᵉ/4ᵉ vague démarre
  quand la précédente finit (pas de pipeline parfait) ; si le temps monte, c'est le verdict.

## Conditions

- Harnais `mb_fa` (dlopen .so, dispatch réel), géométrie prod : D=256, 24 qh, 4 kvh,
  KV q8_0, n_kv=107351 (pad 107520), T=1..8, **200 iters, médiane**.
- V100 seule (CVD=0), service `ollama-v100` stoppé par le harnais pendant la mesure
  (chaîne de supervision : `supervisor_bench.py run mb_fa_pb_bench.py --detach`),
  restauration + vérification avant sortie du bench.
- AUCUN MTP (consigne 24/10). AUCUN changement prod.
- Validation numérique : `out` (4 premiers floats) des 3 variantes — tolérance relative
  1e-4 (différence d'ordre de somme des partiels float ; REF pb=60 vs pb=120/240).

## Grille de lecture

- PB120 ou PB240 < REF de **≥ 3 %** @T=1-2 (et pas >5 % pire ailleurs) → levier A validé,
  on documente et on envisage la prod.
- Neutre (±2 %) → les vagues ne masquent pas mieux : le mur est bien les 12 warps/SM
  (regs), pas le dispatch → on passe à B.
- Pire → la queue de vague (tail effect) + combine pèse plus que le gain d'overlap.
- Le T=6-8 (grille (1,20,8) auto) est le contrôle : si pb forcé aide là-bas, c'est cohérent.

## Artefacts
- Script : `bench/mb_fa_pb_bench.py` ; sortie `mb_fa_pb_results.jsonl` + `mb_fa_pb_summary.md`.
- Build : `h2/cupti129-test/e2/t3v2/build_t3v2.sh t3lba_pb '' '' fattn-mma-volta-small.lba.cuh`
  (source `fattn-common.cuh.pb120-240` sauvegardée, `.bak-pristine` = fork original).

## Invariants post-bench
service `ollama-v100` active · `.so` prod `fa3e75c2` intact · V100 libérée · aucun bench MTP.
