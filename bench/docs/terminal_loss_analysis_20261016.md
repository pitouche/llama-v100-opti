# Perte de l'outil « open terminal » après reprise superviseur — analyse 16/10

Chat visé : `6bc8a6fa-599d-40f9-8147-c24e186545ea` (« Protocole Exploratoire V100 », modèle `kolbo5`).
OWUI : open-webui **0.12.2** (pip, venv `/home/guile/venv.3.12.13`, service user `open-webui`, cwd `/home/guile`).
DB : `/home/guile/venv.3.12.13/lib/python3.12/site-packages/open_webui/data/webui.db`.

## 1. Rappel du mécanisme (constaté dans le code, pas supposé)

### Côté frontend (normal)
- `src/lib/stores/index.ts:170` : `selectedTerminalId: Writable<string|null> = writable(null)`
  → **mémoire vive uniquement, jamais persistée** (ni localStorage, ni profil utilisateur).
- `Chat.svelte:2543-2565` : à chaque send, le payload `/api/chat/completions` contient
  `terminal_id: model.info?.meta?.capabilities?.terminal ? ($selectedTerminalId ?? undefined) : undefined`.
- `Chat.svelte:503-509` (`setDefaults()`, `let settingDefaults = false` → s'exécute **une seule fois
  par session de navigateur**) : si `model.info.meta.terminalId` est défini ET que le terminal existe
  → `selectedTerminalId.set(tid)`.
- `Chat.svelte:945-948` (onMount) : si `selectedTerminalId` pointe vers un terminal inexistant → reset à null.

### Côté modèle (base de données, vérifié)
- `model.meta['kolbo5']` contient : `terminalId = ec51e8a8-ffc7-4361-b4c3-dc9b0d4f753f` (« full-access »,
  `http://127.0.0.1:8000`), `capabilities.terminal = true`.
- L'API `/api/v1/models` **expose bien** `info.meta.terminalId` (vérifié en direct le 16/10).
- Le frontend ne lit `terminalId` QUE dans `setDefaults()` (une fois / navigateur).
  **La reprise API ne lit jamais `model.info.meta.terminalId`** — rien n'injecte le terminal
  dans le payload. (`utils/automations.py:175` le lit pour les automations, pas pour /api/chat/completions.)

### Côté backend (0.12.2)
- `main.py:1144` `chat_completion` : `form_data` passe tel quel dans `metadata`/`process_chat_payload`.
- `utils/middleware.py:2788` : `terminal_id = form_data.pop('terminal_id', None)`.
  **Pas de repli sur `model.info.meta.terminalId`.**
- `utils/middleware.py:3130-3149` : les outils terminal sont résolus uniquement si
  `terminal_id and terminal_capability and terminal_id in terminal_connection_ids`.
- `utils/middleware.py:2453-2459` : `use_builtin_tools` (notes/KB/chat/calendar/time/…) exige
  `metadata['session_id']` non vide. Le superviseur envoie un `session_id` (uuid4) → les builtin
  passent. **Le terminal ne fait pas partie des builtin** (`get_builtin_tools`, `utils/tools.py:601+`) :
  il dépend uniquement de `terminal_id` — d'où le motif exact observé :
  notes/KB/chat/calendar oui, terminal non.
- `utils/middleware.py:3025-3027` : si le payload contient une clé `tools` explicite,
  **toute la résolution serveur est court-circuitée** (le `tools: []` explicite = opt-out).
  Le superviseur n'envoie pas de clé `tools` (à ne pas ajouter « par précaution »).

## 2. Chaîne causale exacte

1. Bench → stop `ollama-v100` → le tir en cours échoue (« Ollama: Cannot connect… 11434 »)
   → nœud assistant en erreur.
2. `supervisor_bench.py` : `check_and_resume()` → `send_resume()` → `fire_completion()` :
   `POST /api/chat/completions` avec payload `chat_completions_payload()`
   (chat_id, id, parent_id, user_message, model, messages, stream, session_id).
   **Aucun `terminal_id`, aucun `tool_servers`, aucun `model_item`.**
3. Frontend hors de la boucle : `selectedTerminalId` (mémoire) est null pour ce tir ;
   le payload ne le contient pas.
4. Backend 0.12.2 : `terminal_id = None` → `get_terminal_tools()` non appelé →
   le modèle voit le terminal dans l'historique (messages tool_call/tool) mais plus aucun outil
   terminal dans sa liste → il dit « Bloquant identifié : la session de reprise n'a pas d'outil
   terminal/shell ».

## 3. Pourquoi c'est systématique (et pas un bug aléatoire)

- Le tir de reprise est **toujours** généré par `supervisor_bench.py` (Python, sans navigateur) →
  `terminal_id` absent **à chaque reprise**.
- Même si l'onglet OWUI est ouvert au moment de la reprise : la génération est prise en charge par
  le POST direct (stream pris en main côté OWUI, mais le payload d'origine reste celui du POST).
  Le store navigateur n'intervient pas.
- Conséquence identique pour toute reprise API (superviseur, script, curl) : le terminal est perdu
  jusqu'au prochain rechargement de page (ou la re-sélection manuelle du terminal dans le menu).

## 4. Pistes / solutions (au choix)

### A. Fix superviseur (recommandé, local, sans patch OWUI)
Dans `supervisor_bench.py` :
1. Lire l'ID du terminal : `GET /api/v1/models` → modèle `kolbo5` → `info.meta.terminalId`
   (généralisable : `OWUI_MODEL_ID` env, défaut `kolbo5`) ; vérifier qu'il est présent dans
   `terminal_server.connections` (via `GET /api/v1/configs/terminal_server` si exposé, sinon
   confiance + le log OWUI signalera).
2. L'ajouter au payload de `fire_completion()` : `"terminal_id": <id>`.
3. Ne jamais envoyer de clé `tools` (côt-circuit backend, middleware.py:3025).
Risque : nul côté données (le tir reste idempotent, UUID déterministes inchangés).

### B. Patch backend OWUI (réparation à la racine, pour TOUS les appels API)
Dans `process_chat_payload` (middleware.py, vers 2788) : quand `terminal_id` est absent,
repli sur `model.info.meta.terminalId` (à la manière de `_resolve_model_defaults` d'`automations.py:167-177`).
Implémentation ~10 lignes dans site-packages (ou config.json / sitecustomize), **mais** :
- écrasé par `pip install -U open-webui` (documenter + ré-appliquer) ;
- change le comportement global (un appel API « sans terminal » pourrait en recevoir un —
  à peser, ici c'est l'effet voulu).

### C. Améliorer le message de reprise (symptôme, pas la cause)
Modifier `RESUME_PROMPT` pour dire au modèle : « si un outil a disparu après la reprise
(terminal), signale-le et demande sa réactivation plutôt que de bloquer ». Utile en attendant A/B,
n'enlève pas le blocage effectif (les vérifs post-run restent impossibles).

### D. Persistance frontend du terminal (amont du bug)
`selectedTerminalId` en mémoire seule + `setDefaults()` une-fois-par-session expliquent aussi que
toute recharge de page sans terminal configuré sur le modèle perd la sélection. OWUI a déjà
la moitié de la solution (`model.meta.terminalId`) ; un PR amont « persist selectedTerminalId /
restore per-chat » supprimerait l'asymétrie. Non bloquant pour nous (A/B suffisent).

### E. À exclure (vérifié)
- `function_calling='legacy'` : le modèle `kolbo5` n'a pas ce param (`params = {system,
  custom_params}`) → `function_calling` résout à `'native'` (main.py:1355-1358).
- `use_builtin_tools`/`session_id` : le superviseur envoie un `session_id` → les builtin passent ;
  le terminal n'est de toute façon pas dans les builtin.
- Connexion terminal désactivée : `full-access` est `enabled: true` dans `terminal_server.connections`
  (config DB, vérifié).
- Capacité `terminal` du modèle : `true` (DB, vérifié).
- KB « vide » : volontaire (guile) — hors périmètre, confirmé.

## 5. Test de reproductibilité demandé (statut)

Bench de ~20 s écrit (stop/start `ollama-v100` via sudo sans mot de passe — vérifié fonctionnel,
comportement identique aux run9*). **Non lancé** : j'ai arrêté juste avant le stop (le tir en
cours est protégé de toutes les autres commandes ; une interruption n'aurait pas de sens).
Le service `ollama-v100` est `active` et le nœud courant du chat est sain → aucun nœud d'erreur
créé. Si tu veux la preuve dure (nœud de reprise sans terminal, reproductible à l'identique),
relance le superviseur :
```
cd /home/guile/llama-experiments/bench
python3 supervisor_bench.py run <petit_bench.py> --detach
```
Attendu : erreur « Cannot connect… » sur le nœud courant → reprise automatique → message
assistant avec le même « Bloquant identifié » (pas d'outil terminal) que le 15/10.

## 6. Résumé

- Cause racine : le tir de reprise `POST /api/chat/completions` du superviseur ne porte pas de
  `terminal_id` (le frontend est hors de la boucle et son store est en mémoire seule), et le
  backend 0.12.2 n'a aucun repli sur `model.info.meta.terminalId`.
- Motif observé (notes/KB/chat/calendar présents, terminal absent) = exactement la frontière
  builtin-tools (session_id présent) / terminal (terminal_id absent) de `process_chat_payload`.
- Fix minimal : ajouter `terminal_id` (lu depuis `model.info.meta.terminalId`) au payload du
  superviseur (option A). Fix structurel : patch repli côté backend (option B) et/ou PR amont (D).

## 7. Test 16/10 — fix A implémenté + reproductibilité

- **Fix A appliqué** dans `supervisor_bench.py` : nouvelle fonction `resolve_terminal_id(model)`
  (override env `OWUI_TERMINAL_ID`, sinon `GET /api/v1/models` → `info.meta.terminalId`) ;
  `fire_completion()` ajoute `terminal_id` au payload avec log
  (« Reprise : terminal_id=… réinjecté (meta.terminalId du modèle). »).
  README mis à jour. Aucun ajout de clé `tools` (court-circuit backend).
- `terminal_test_bench.py` : sleep 8 s → `sudo systemctl stop ollama-v100` → coupure 20 s →
  `start` → attente `active` (90 s max). Durée ~35 s. rc=0 si service actif. Séquence
  systemctl identique à run9c3_bench.py, sans aucune autre action sur la V100.
- Lancement : `python3 supervisor_bench.py run terminal_test_bench.py --detach` (exécuté
  16/10 à 13:51) — **VALIDÉ** :
  1. ✅ Erreur « Ollama: Cannot connect… 11434 » sur le nœud assistant courant (93ff536c).
  2. ✅ Log : `Reprise : terminal_id=ec51e8a8-ffc7-4361-b4c3-dc9b0d4f753f réinjecté (meta.terminalId du modèle).`
  3. ✅ Reprise envoyée (nœud assistant ad08b0e8).
  4. ✅ Session de reprise : 4 `tool_calls` `run_command` dans le nœud de reprise — l'outil
     terminal était disponible (sans fix : blocant « n'a pas d'outil terminal/shell »).
  - Note : le chat actif au moment du test était `e4e14b0f-b0eb-41cd-b40b-82b3f25b5d5a`
    (« Perte terminal superviseur ») — la découverte `active=True` a bien visé le bon chat.
- Vérification a posteriori : `grep -n "terminal_id" supervisor_bench.log` ; nœud de reprise
  `done=True` sans erreur, avec tool_calls terminal. (Fait le 16/10.)
