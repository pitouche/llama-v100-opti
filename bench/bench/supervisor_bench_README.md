# supervisor_bench.py — superviseur de run V100 avec reprise d'inférence

**V1 — documentée le 16/10 sur la base du code réel (`supervisor_bench.py`, 564 lignes,
5 fonctions de reprise + `resolve_terminal_id`).** Ce README est l'autorité : si un autre
document (note, protocole, mémoire) décrit un mécanisme différent, c'est lui qui a tort.
À contexte vide, ce seul fichier suffit pour lancer, vérifier et déboguer un cycle.

## À quoi ça sert

Un bench V100 stoppe le service `ollama-v100` (port 11434) pendant la mesure. Si l'agent
OWUI est en train de générer au moment du stop, le tir échoue
(« Ollama: Cannot connect to host localhost:11434 … ») et le nœud assistant courant du
chat reste en erreur : **l'agent est bloqué** tant qu'on ne relance pas la génération.

Le superviseur s'occupe de la **fin de cycle** :

1. **Lance** le bench comme enfant Python (`python3 <bench> [args]`), dans son propre
   groupe de process (`setsid`), cwd = `bench/`.
2. **Attend** sa sortie **sans timeout** (les timeouts sont dans le script du bench).
   Renvoie le code de sortie **tel quel** (transparence).
3. **Filet de sécurité** : `killpg` (SIGTERM puis SIGKILL) sur le groupe de l'enfant s'il
   reste des résidus (crash du bench). No-op si l'enfant est propre.
4. **Vérifie passivement** le service (`systemctl is-active ollama-v100`), pause 10 s,
   puis **reprise OWUI** si besoin. **Il ne stoppe JAMAIS le service et ne le redémarre
   JAMAIS** : si le service n'est pas `active`, il loggue `Intervention humaine requise`
   et n'y touche pas (config éventuellement pourrie laissée par le bench). La reprise OWUI
   est **tout de même** tirée ensuite — voulu : la 2ᵉ erreur dans l'historique est la preuve
   que le superviseur a tourné.
5. **S'arrête.** Après le run et la reprise éventuelle.

Le bench reste responsable de **sa propre séquence V100** (stop → mesure → restauration du
service avant de sortir, typiquement via `trap EXIT`). Si le bench sort sans avoir restauré,
le superviseur le constatera dans le log et demandera une intervention humaine — il ne
corrige pas.

## Contrainte d'interface (à retenir)

Le superviseur exécute **`python3 <bench> [args]`** : le fichier en argument doit être un
**script Python**. Un `.sh` transmis direct échoue en 0 s avec `SyntaxError`. Si le bench
est shell, écrire un wrapper Python : `subprocess.run(["bash", "run.sh", *args])` et
retourner le rc (précédent : `h2/cupti129-test/nightrun8/run8_bench.py`).

## Usage

```bash
cd /home/guile/llama-experiments/bench

# Depuis un terminal :
python3 supervisor_bench.py run <bench.py> [args du bench]

# Depuis l'agent (obligatoire : le tir en cours est coupé par le stop du service,
# le superviseur doit survivre à la fin du message agent) :
python3 supervisor_bench.py run <bench.py> [args] --detach
```

`--detach` relance le superviseur en nouvelle session : il survit à l'interruption.
Affichage : `Superviseur détaché. Log : …` ; tout se lit ensuite dans `supervisor_bench.log`.
Un verrou `flock` limite à **1 superviseur à la fois** (2ᵉ instance → sortie immédiate,
message dans le log).

**Vérification après le run** (le superviseur ne fait que *tirer* la reprise ; il ne
confirme pas la fin de la génération) :

```bash
tail -8 supervisor_bench.log                 # "=== FIN du run : rc=N, … ==="
cat /tmp/kolbo-supervisor-bench.last-rc      # "rc note"
systemctl is-active ollama-v100              # doit être active
```

puis confirmation que le nœud de reprise a fini (`done=True`, `error` vide, au moins un
tool_call si la session repasse par le terminal) — extrait Python prêt à l'emploi en fin de
ce README.

## Ce que fait la reprise (mécanique exacte)

**Chat visé.** `OWUI_CHAT_ID` si défini, sinon découverte dans `GET /api/v1/chats/?page=1` :
chat `active=True` d'abord, repli = le plus récent daté.

**Condition de reprise.** Le nœud `currentId` du chat est un assistant **avec `error` non
vide** (l'erreur et `done=True` peuvent coexister). Sinon : log « rien à reprendre », rien
d'autre. (Un nœud `done=False` orphelin sans erreur — stream mort — n'est **pas** repris.)

**Les deux nœuds de reprise — mécanisme d'IDs (important).**

```python
user_id      = str(uuid.uuid5(uuid.NAMESPACE_URL, f"kolbo-resume:{message_id}"))
assistant_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"kolbo-resume-a:{message_id}"))
# message_id = ID du nœud assistant EN ERREUR (celui que le superviseur a trouvé)
```

Garanties qui en découlent :

- **Déterministe** : les IDs ne dépendent QUE de `message_id` (pas du temps, pas de compteur).
- **Idempotent** : si la reprise est tirée deux fois sur le même nœud en erreur (relance du
  superviseur, double cycle), on recalcule **les mêmes deux IDs** → le commit OWUI fusionne
  (même clé dans `history.messages`), **pas de jumeau** dans le graphe.
- **Branchement** : les deux nœuds sont branchés **sur** le nœud en erreur, qui est
  **conservé** (il reste dans l'historique, visible dans l'UI).
  `user.parentId = message_id`, `assistant.parentId = user_id`,
  `history.currentId` est déplacé sur le nœud assistant de reprise.
- Le commit est un `POST /api/v1/chats/<chat_id>` **partiel** : seuls les deux nouveaux
  nœuds sont envoyés (pas le parent) — le backend OWUI fusionne dans l'existant. Vérifié
  dans le code : `commit_nodes()` + `verify_node()` (relecture du chat après commit, erreur
  explicite si le nœud assistant a disparu).

**Message de reprise.** Texte unique figé dans le code (`RESUME_PROMPT`, « REPRISE
AUTOMATIQUE (superviseur bench) : … »), identique que le service soit up ou down. C'est la
signature pour trouver les nœuds de reprise dans le graphe (extrait de vérification ci-dessous).

**Réinjection du terminal (fix 16/10).** Le tir de reprise est un `POST /api/chat/completions`
**sans frontend** : le store navigateur `selectedTerminalId` (mémoire seule, jamais
persisté) ne peut pas fournir le terminal, et le backend OWUI 0.12.2 n'a **aucun repli** sur
`model.info.meta.terminalId` (`utils/middleware.py` : les outils terminal ne se résolvent que
si le payload porte `terminal_id`). Sans réinjection, le modèle repart **sans l'outil
terminal** (les outils builtin — notes/KB/chat/calendar — passent, eux, via `session_id`).
Le superviseur corrige ça :

```python
terminal_id = resolve_terminal_id(model)   # 1. env OWUI_TERMINAL_ID (override)
                                             # 2. sinon GET /api/v1/models → modèle du
                                             #    nœud en erreur → info.meta.terminalId
                                             #    (None + warning si introuvable)
if terminal_id:
    payload["terminal_id"] = terminal_id
```

Log attendu : `Reprise : terminal_id=<uuid> réinjecté (meta.terminalId du modèle).`

⚠️ **Ne JAMAIS ajouter de clé `tools` au payload du tir** : si le payload contient `tools`
(explicite, même `[]`), le backend court-circuite **toute** la résolution serveur d'outils
(`utils/middleware.py:3025`) → le modèle repart sans AUCUN outil, terminal compris.

**Le tir (non bloquant).** `POST /api/chat/completions` en stream, fermé après
**acceptation** (attente max `BENCH_FIRE_WAIT_S`, défaut 45 s), puis le superviseur quitte :
OWUI prend la main et attend à sa place. Payload exact :

```python
{
    "chat_id": ...,          # chat agent
    "id": assistant_id,      # nœud assistant de reprise (uuid5 ci-dessus)
    "parent_id": message_id, # nœud assistant EN ERREUR (branchement)
    "user_message": {...},   # RESUME_PROMPT
    "model": ...,            # modèle du nœud en erreur
    "messages": [...],       # chemin racine→nœud courant (contenus vides filtrés)
                             #            + RESUME_PROMPT en dernier (rôle user)
    "stream": True,
    "session_id": str(uuid4()),  # uuid4 FRAIS à chaque tir (nécessaire aux builtin tools)
    "terminal_id": ...,     # réinjection 16/10 (absent si non résolu → warning log)
}
```

## Règles V100 / profiling à connaître pour ÉCRIRE un bench (30/10)

Source de détail : **`bench/GPU_UUID_NSYS_CUPTI_20261030.md`** (note de référence).

1. **GPU par UUID, jamais d'index en dur.** Les numérotations sont inversées sur cette
   machine (état 30/10 : V100 = CUDA ordinal **0** mais `nvidia-smi --id=1` ; 1070 = l'inverse)
   et peuvent bouger au reboot. Helpers : `bench/gpu_uuid.py` (CLI, abort si V100 absent) et
   `bench/gpu_uuid_select.py` (Python : `v100_cuda_ordinal()` pour `CUDA_VISIBLE_DEVICES`,
   `v100_smi_index()` pour `nvidia-smi --id=`). Scripts volontairement multi-GPU (tests 1070) :
   cibler « l'unique GPU qui n'est PAS le V100 » par UUID — ne pas les patcher vers la V100.
2. **nsys CUDA** : nsys 2025.5.2 tel quel charge sa CUPTI 13.1 → **0 événement CUDA** sur le
   driver 580.178.04 (Volta non supportée ≥ Nsight 2025.4). La capture CUDA exige
   `NSYS_CUPTI_LIBRARY_PATH=bench/nsys-cupti129` — un répertoire contenant **une seule**
   `libcupti.so.12.9` (md5 `9bc91feae571aa505069c0640e85c26c`) ; un dir multi-versions fait
   re-choisir la 13.1 (cause du faux négatif de la matrice 09/10, `h2/ETAT.md`).
   Wrapper prêt : **`bench/nsys_cuda.sh`** (fixe l'env + résout le V100 par UUID).
   Toujours `--cuda-graph-trace=node` pour llama-server (sinon les kernels de graph ne sont pas
   exposés individuellement). Valider chaque `.rep` : `bench/inspect_nsdb.py <f.nsys-rep>`
   (tables + compteurs) ou `nsys stats --report cuda_gpu_kern_sum`.
3. **sudo + env** : `sudo -E` est IGNORÉ ici (« preserving the entire environment is not
   supported ») → passer l'env explicitement `sudo env K=V… <cmd>` (pattern
   `nso_orchestrator_b2048.py`). nsys tourne root : le `.rep` est root-owned → `mv` +
   `chown guile` en fin de bench.
4. **Ne jamais mélanger** le shim `LD_PRELOAD=libcupath*.so` (trace CUPTI 12.9 direct) et nsys
   dans le même process : double inscription CUPTI, les deux échouent.
5. **Le bench restaure `ollama-v100` avant de sortir** (trap EXIT) ; le superviseur ne fait que
   vérifier passivement et tirer la reprise OWUI — voir ci-dessous.

## Fichiers et variables

| Chemin / variable | Rôle |
|---|---|
| `bench/supervisor_bench.log` | log du superviseur (tous les cycles, format `date LEVEL [pid] msg`) |
| `/tmp/kolbo-supervisor-bench.lock` | verrou `flock` EX, 1 superviseur à la fois |
| `/tmp/kolbo-supervisor-bench.last-rc` | dernière fin : `rc note` (utile en `--detach`) |
| `bench/terminal_test_bench.py` | bench de test ~30 s (stop/start `ollama-v100`) pour vérifier un cycle de reprise sans vraie mesure |
| `~/.config/kolbo-supervisor/token` | token OWUI (0600) ; override `OWUI_SUPERVISOR_TOKEN_FILE`, sinon env `OWUI_API_KEY` / `OWUI_TOKEN`. Rien en dur dans le code. |
| `OWUI_URL` | défaut `http://127.0.0.1:8080` |
| `OWUI_CHAT_ID` | forcer le chat agent (sinon découverte) |
| `OWUI_TERMINAL_ID` | override de l'ID terminal réinjecté à la reprise (défaut : `meta.terminalId` du modèle, lu via `/api/v1/models`) |
| `BENCH_OLLAMA_SERVICE` | service vérifié en fin de run, défaut `ollama-v100` (**vérification seule, jamais d'action**) |
| `BENCH_POST_RUN_PAUSE_S` | pause avant vérification/reprise, défaut 10 s |
| `BENCH_FIRE_WAIT_S` | attente max d'acceptation du tir de reprise, défaut 45 s |
| `LOG_LEVEL` | défaut `INFO` |

## Limites et comportements à connaître

- **Le superviseur ne restaure jamais le service** : bench sorti avec `ollama-v100` inactif
  → log `Intervention humaine requise` (mais la reprise OWUI est quand même tirée — voulu).
- **La reprise est un tir non bloquant** : le superviseur quitte sans vérifier que la
  génération a fini → confirmer séparément (extrait ci-dessous).
- **Le rc renvoyé est celui du bench** : un échec d'analyse *post-capture* dans le script
  (`rc=1` avec trace valide et service restauré) ne rend pas la mesure invalide. Toujours
  distinguer la validité de la mesure de la rc du pipeline (consulter le log + les artefacts).
- **Pas de timeout sur l'enfant** : un bench bloqué bloque le superviseur (les timeouts
  doivent être dans le script du bench).
- Le chat est choisi sur la **page 1** de l'API OWUI (suffisant ici).
- Ne jamais lancer un bench au milieu d'un tour d'inférence actif si on veut éviter le
  doublon d'erreur : le bench qui stoppe le service coupe le tir en cours (c'est le cas
  normal géré ici), mais deux interruptions successives = deux nœuds en erreur empilés.

## Vérifier la reprise via l'API OWUI (extrait prêt à l'emploi)

```python
import json, urllib.request, pathlib
tok = pathlib.Path.home().joinpath(".config/kolbo-supervisor/token").read_text().strip()
def get(p):
    r = urllib.request.Request("http://127.0.0.1:8080" + p,
                               headers={"Authorization": f"Bearer {tok}"})
    return json.load(urllib.request.urlopen(r, timeout=60))
CID = ""            # chat agent : laisser "" pour découvrir le chat actif
if not CID:
    CID = next(c["id"] for c in get("/api/v1/chats/?page=1") if c.get("active"))
msgs = get(f"/api/v1/chats/{CID}")["chat"]["history"]["messages"]
for nid, m in msgs.items():
    if isinstance(m.get("content"), str) and m["content"].startswith("REPRISE AUTOMATIQUE"):
        aid = (m.get("childrenIds") or [None])[0]     # nœud assistant de reprise
        a = msgs.get(aid, {})
        tools = [i.get("name") for i in (a.get("output") or [])
                 if isinstance(i, dict) and i.get("type") == "function_call"]
        print("reprise user", nid[:8], "→ assistant", str(aid)[:8],
              "done:", a.get("done"), "error:", bool(a.get("error")),
              "tools:", tools[:5])
```

Attendu en fin de cycle : `done: True`, `error: False`, et — si la session de reprise a
réellement l'outil terminal — des `run_command` (ou équivalent) dans `tools`.

## Ce qui n'existe PAS dans ce code (ne pas chercher)

`bench_supervised.sh`, `--watch`, les markers `/tmp/kolbo-bench-run.{start,end}`,
`bench_status()`/état `crashed` : des mécanismes d'anciens brouillons, **absents de
`supervisor_bench.py`**. La détection de fin de run est directe : le superviseur *est* le
parent du bench, il attend sa sortie. Le seul mode est `run`.

## chat_watchdog.py — garde de veille (cron 5 min, depuis le 16/10)

Le superviseur ne reprend que **après un run** (et son tir de reprise est **non
bloquant** : il quitte sans vérifier que la génération a fini). Entre les runs, et
sur la zone résiduelle juste après un tir, une fin de génération peut échouer pour
des raisons qu'il ne gère pas. Le watchdog couvre trois catégories d'échecs
(17/10 et 27/10) :

- **l'erreur de dépassement de contexte**
  (« request (N tokens) exceeds the available context size … » — typiquement un
  appel outils qui renvoie trop de données) : la catégorie **pas relançable telle
  quelle** (le tir redonnerait exactement le même prompt trop long) → compacter
  puis relancer ;
- **l'erreur transitoire** (17/10) : runner de modèle arrêté à chaud (« model
  runner has unexpectedly stopped … »), « Ollama: Cannot connect … », etc. —
  **relançable telle quelle**, si le service est `active` et si ce n'est pas une
  reprise morte (disjoncteur ci-dessous) ;
- **le plafond silencieux** (27/10, cas C) : nœud `done=True` **SANS aucune
  erreur**, mais le dernier sous-appel a claqué pile au plafond du contexte du
  runner. Le stream se termine « proprement » en fin de contexte : **rien n'est
  stocké dans `error`** → A et B ne voient rien, alors que le travail est
  tronqué (cas réel : chat `fae260af`, tour T3 v2, 127 998/128 000 tokens,
  arrêt en plein `function_call`). Signal **quantitatif**, pas textuel :
  `usage.prompt_tokens + usage.completion_tokens` du **dernier sous-appel**
  (l'usage du nœud est celui du dernier sous-appel — **pas** `input_tokens`/
  `output_tokens` qui sont des **cumuls**) `>= context_length(runner live) − 4`
  (marge MTP : un pas peut émettre 2+ tokens, le total rapporté peut atterrir
  quelques tokens sous le plafond — un strict `== Cmax` aurait manqué le cas
  réel). Le ctx live se lit en **lecture passive** : modèle du nœud →
  `base_model_id` (`/api/v1/models`) → runner (`/api/ps` direct sur ollama,
  **pas** via OWUI qui n'a pas de tel endpoint). Action (depuis 27/10) :
  **re-tir DIRECT en SELF-CHECK, sans compaction forcée** : le prompt
  demande au modèle de vérifier lui-même si la tâche était interrompue ; si
  elle était terminée, il le dit et s'arrête (le faux positif — tour naturel
  fini dans `[Cmax−4, Cmax]` — coûte un mini tour, et reste visible dans le
  chat). La compaction est déléguée à OWUI : chaque tir passe par
  `compact_messages_for_request` dans `process_chat_payload` (middleware du
  chat), qui, si le seuil `chat.context_compaction.token_threshold` est
  dépassé, résume le contexte **avant** la génération — c'est la même
  compaction on-request que fait l'UI. Validé 27/10 sur `fae260af` : tir
  direct à 16:00 → compaction on-request 16:03 (`dropped=3`) → le modèle a
  repris pile au point d'interruption ; un continue manuel 4 h plus tard a
  été compacté de même (20:33, `dropped=87`) et le run est allé au bout.
  Le `/compact` explicite a, lui, échoué en cascade sur ce chat (500 MTP «
  failed to flush deferred speculative prompt prefix » → timeout 180 s →
  `too_short` → faux « Compaction effective ») — il générait en plus un
  prompt de ~127k tokens sur le MÊME runner à 128k.

**Périmètre exact (tout le reste → inaction, log explicite)** :

1. le superviseur n'est PAS en cours → sinon sortie immédiate ;
2. le nœud `currentId` du chat agent est un assistant **`done=True`** (une
   génération en cours n'est jamais touchée) ;
3. l'erreur du nœud est de **contexte** (signatures `exceed_context_size_error`,
   « exceeds the available context size », « context window », …) → actions 1+2
   ci-dessous ; **ou transitoire** (tout le reste des erreurs) → action 3, sous
   les deux portes : service `ollama-v100` = `active` (lecture passive) et nœud
   ≠ reprise morte ; **ou il n'y a PAS d'erreur** → action 4, si le dernier
   sous-appel est `>= ctx live − marge` (plafond silencieux) et nœud ≠ reprise
   morte.

Ordre d'évaluation : `done=True` → role assistant → **disjoncteur** →
A/B/C (le disjoncteur passe avant tout, y compris le plafond sans erreur).

**Le disjoncteur — borne stricte des chaînes d'auto-reprise (17/10, étendu à
C le 27/10).**

Si le nœud est une **reprise** — son parent est un nœud user commençant par
« REPRISE AUTOMATIQUE » (superviseur **ou** watchdog) — le watchdog ne fait
**RIEN** (log `REPRISE MORTE … disjoncteur`, intervention humaine). La chaîne
est bornée : erreur d'origine → **1 auto-reprise max** → si cette reprise
meurt → humain. C'est l'anti-boucle sur instabilité persistante (ex. `.so`
corrompu qui re-crash à chaque re-tir — incident 25/10 : re-tirer aurait mort
de nouveau, il fallait la restauration du `.so`). Règle **globale (A, B et C)** :
elle s'applique aussi à A (une erreur de contexte sur une reprise morte n'est
pas re-traitée non plus) **et à C** : si le self-check est lui-même long et
recolle au plafond, le nouveau nœud est une reprise → pas de 2ᵉ self-check.
Détection par la signature du contenu du nœud parent (même signature que
l'extrait de vérification), pas par l'ID des nœuds (l'idempotence des uuid5 ne
suffit pas : chaque reprise morte porte un **nouvel** ID).

**Les actions quand le cas se présente :**

1. **(A) Compaction forcée** : `POST /api/v1/chats/<id>/compact` — c'est le
   mécanisme interne du « /COMPACT » de l'UI, appelé côté serveur (timeout 180 s :
   le résumé est généré par le modèle). OWUI 0.12.2 écrit le résumé de la branche
   courante sur le nœud courant sous `contextSummary` ; à la prochaine
   génération, le backend replacera le contexte par résumé + messages récents
   (`_apply_latest_summary_checkpoint`, `utils/context_compaction.py`). Le
   watchdog exige **`compacted: true` dans la réponse** ET un `contextSummary`
   sur le nœud courant avant de relancer — sinon pas de tir. (Le 27/10, la
   vérification ne regardait que le `contextSummary` hérité d'une tentative
   antérieure : `compacted: false, reason: too_short` + résumé déjà présent →
   faux « Compaction effective » → tir inutile recollé au plafond. La
   compaction forcée reste en A parce que l'erreur de contexte du tir
   précédent prouve que le prompt tel quel dépasse le plafond : sans résumé
   préalable, le re-tir retomberait sur la même erreur. Le cas C, lui, ne
   compacte PLUS — voir 4.)
2. **(A) Relance** : exactement la mécanique du superviseur (importé, pas
   dupliquée) — commit de 2 nœuds déterministes + tir non bloquant +
   réinjection `terminal_id` + jamais de clé `tools`. Préfixe d'IDs
   **`kolbo-wd:`** / **`kolbo-wd-a:`** (distinct de `kolbo-resume:` → aucune
   collision de jumeau possible entre les deux outils) et message proprio
   (`WD_PROMPT`, signature « REPRISE AUTOMATIQUE (watchdog) »).
3. **(B) Re-tir tel quel** (17/10) : même mécanique qu'en 2 — commit de 2 nœuds
   déterministes `kolbo-wd:` / `kolbo-wd-a:` + tir non bloquant + réinjection
   `terminal_id` + jamais de clé `tools` — **sans compaction**, message proprio
   `WD_TRANSIT_PROMPT` (« REPRISE AUTOMATIQUE (watchdog) : … erreur transitoire
   du serveur … »). Le service n'est **jamais** stoppé ni redémarré (lecture
   passive `systemctl is-active`) : si le service n'est pas `active`, RIEN —
   intervention humaine (même philosophie que le superviseur).
4. **(C) Re-tir direct + SELF-CHECK** (27/10, modifié le jour même) : **pas de
   compaction forcée** — même mécanique qu'en 2, avec le message
   `WD_CEILING_PROMPT` (« REPRISE AUTOMATIQUE (watchdog) : … s'est terminée
   pile à la limite du contexte maximum — elle a pu être interrompue …
   vérifie : la tâche en cours était-elle réellement terminée ? … »). La
   compaction est faite par OWUI au moment du tir (compaction on-request,
   cf. catégorie « plafond silencieux »). Si la compaction on-request ne
   peut pas assez réduire (nœud géant sans checkpoint interne — le tir
   repart à ~127k et recolle au plafond), le nœud de re-tir est une reprise
   → **disjoncteur** → humain (même borne qu'avant, sans la cascade de
   `/compact`). Si le runner n'est pas chargé dans `/api/ps` (pas de ctx
   live connu) ou si l'usage du nœud est incomplet → **pas de déclenchement**
   (on ne devine pas un plafond).

**Anti-collision avec le superviseur (les deux ne peuvent pas entrer en collision) :**

- **Même verrou** `/tmp/kolbo-supervisor-bench.lock`. Le superviseur l'acquiert au
  démarrage (tolérance `BENCH_LOCK_TIMEOUT_S`, défaut 120 s, puis le tient jusqu'à la fin
  de la reprise) ; le watchdog le tente **une seule fois, non bloquant** : tenu → sortie
  sans rien faire. Comme le watchdog ne s'occupe que des fins de génération **entre** les
  runs et que le superviseur est verrouillé **pendant** les runs, il n'y a pas d'intervalle
  où les deux agissent en même temps. Les IDs distincts (kolbo-wd vs kolbo-resume) sont la
  2ᵉ barrière : même nœud en erreur traité par les deux → deux branches distinctes, jamais
  de doublon de nœud.
- Le watchdog tient aussi le verrou pendant SA compaction+relance (un superviseur lancé
  pendant ces ~2-3 min attend dans sa tolérance, sans avorter).
- Safety global : `WATCHDOG_MAX_RUN_S` (défaut 300 s, SIGALRM).

**Installation (cron user guile, système cron actif) :**

```cron
*/5 * * * * cd /home/guile/llama-experiments/bench && /home/guile/venv.3.12.13/bin/python3.12 chat_watchdog.py >> /home/guile/llama-experiments/bench/chat_watchdog.cron.log 2>&1
```

| Fichier / variable | Rôle |
|---|---|
| `bench/chat_watchdog.log` | log du watchdog (tous les passages — la plupart sont « rien à faire ») |
| `bench/chat_watchdog.cron.log` | miroir du stdout pour le cron |
| `bench/watchdog_test_fixture.py` | crée un chat test + nœud FINI selon `--error ctx\|transient\|dead_resume\|ceiling\|ceiling_low\|ceiling_dead` |
| `OWUI_CHAT_ID` | override du chat observé (sinon découverte active, comme le superviseur) |
| `OWUI_TERMINAL_ID` | idem superviseur |
| `WATCHDOG_MAX_RUN_S` | safety globale, défaut 300 |
| `WD_CEILING_MARGIN` | marge du plafond silencieux (tokens), défaut 4 (marge MTP) |
| `WD_OLLAMA_URL` | URL ollama pour la lecture passive `/api/ps`, défaut `http://127.0.0.1:11434` |

Comportements à connaître :
- **Le watchdog est le filet sur la zone résiduelle du superviseur** (17/10) : le tir
  de reprise du superviseur est non bloquant — si cette reprise échoue PENDANT la
  génération (runner arrêté à chaud après la fin de run), c'est le prochain tick du
  watchdog qui la rattrape (cas 17:49 du 25/10). Si le re-tir du watchdog échoue à son
  tour → **disjoncteur** → humain (la chaîne est bornée à 1 auto-reprise par origine
  d'erreur).
- **Le watchdog ne vérifie pas la fin de la génération qu'il relance** (tir non bloquant,
  comme le superviseur) — la confirmation se fait via l'extrait API ci-dessus.
- Le tir de relance repart **sans la clé `tools`** (piège du README, identique).
- Cas B : le watchdog **ne stoppe jamais le service et ne le redémarre jamais**
  (lecture passive) — service non `active` → RIEN, intervention humaine.
- Cas C (plafond silencieux) : le watchdog ne lit le contexte du runner qu'en
  **lecture passive** (`/api/ps` direct sur ollama) — runner non chargé ou
  usage incomplet → pas de déclenchement (on ne devine pas un plafond).
  Faux positif possible (tour naturel fini dans `[Cmax−4, Cmax]`) → coûte un
  mini-tour visible : le self-check répond « terminé » et s'arrête. Si le
  self-check est lui-même long et recolle au plafond → **disjoncteur** (1
  auto-reprise max par origine).
- Un cas « chat empoisonné » (ex. « Failed to tokenize prompt », sortie
  terminal corrompue — erreur 400 **non-retryable**, le re-tir ne la guérit
  pas et peut la propager) est **remisé** (17/10, à remettre sur la table
  après expérimentation du cas B) : solution conçue = extraire le contenu
  propre → log d'audit → réinjecter sans le poison → tronquer ; le
  disjoncteur bornait la chaîne entre-temps.
- Le cas C ne s'applique qu'au **chat suivi** (`active=True`, ou override
  `OWUI_CHAT_ID`) : un chat non suivi au plafond n'est pas vu (limites
  connues : multi-chats en suspens — à trancher après expérimentation).
- Une erreur de contexte sur un nœud `done=False` (stream mort à mi-erreur) n'est PAS
  traitée : c'est un cas dégénère, on préfère l'intervention humaine.
- Le « /COMPACT » de l'UI : c'est bien ce `POST /{id}/compact` (rien d'autre) ; le
  threshold de compaction automatique (`chat.context_compaction.token_threshold`, défaut
  80 000, config admin OWUI) est **déclenché côté serveur** sur chaque tir
  (`compact_messages_for_request` dans le middleware) — mais la compaction forcée du
  watchdog ne dépend pas de ce seuil : elle compacte la branche entière quoi qu'il arrive.
