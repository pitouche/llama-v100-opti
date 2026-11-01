#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
supervisor_bench.py — superviseur de run V100 avec reprise d'inférence.

USAGE (mode unique) :
  python3 supervisor_bench.py run <bench.py> [args du bench] [--detach]

Principe (figé avec guile le 11/10) :
  - Le superviseur est EXTERNE au bench : il le lance comme enfant (propre
    groupe de process), attend sa sortie, et c'est tout en amont.
  - Le bench se charge de SA séquence V100 (stop/reconfig/restart du
    service ollama-v100 s'il en a besoin). Le superviseur ne stoppe JAMAIS
    le service.
  - À la fin du run (quel que soit le code de sortie), le superviseur :
      1. killpg filet de sécurité sur les résidus de l'enfant (crash),
      2. pause ~10 s,
      3. vérifie que le PROCESSUS ollama-v100 tourne (systemctl is-active).
         Si non : il ne touche RIEN (le script peut avoir laissé une config
         pourrie — le risque est d'empirer) → log clair, intervention
         humaine. La reprise est tout de même envoyée (voir 4) : c'est
         voulu, la 2e erreur dans l'historique prouve que le superviseur
         a fonctionné.
      4. vérifie le nœud assistant courant de l'agent dans OWUI :
         - erreur sur le nœud → envoi du message de reprise (commit des
           nœuds déterministes + tir, avec réinjection de terminal_id
           depuis meta.terminalId du modèle). Le tir est NON BLOQUANT :
           OWUI prend la main, c'est OWUI qui attend ; le superviseur
           quitte.
         - pas d'erreur → rien du tout.
      5. renvoie le code de sortie du bench tel quel (transparence).

  Pas de recovery "blind" du service, pas de dévination, pas de timeout
  sur l'enfant (les timeouts sont gérés dans les scripts du bench).

LOG : /home/guile/llama-experiments/bench/supervisor_bench.log
VERROU : /tmp/kolbo-supervisor-bench.lock (1 superviseur à la fois)
RC DERNIER RUN : /tmp/kolbo-supervisor-bench.last-rc (utile en --detach)
"""
import argparse
import fcntl
import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

OWUI_URL = os.getenv("OWUI_URL", "http://127.0.0.1:8080").rstrip("/")

# Token : fichier 0600 (rien en dur dans le code).
TOKEN_FILE = Path(
    os.getenv("OWUI_SUPERVISOR_TOKEN_FILE",
              str(Path.home() / ".config/kolbo-supervisor/token"))
)
OWUI_TOKEN = ""

D = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(D, "supervisor_bench.log")
# Verrou : 1 superviseur à la fois. Le watchdog (chat_watchdog.py, cron)
# tente ce même verrou en NON-BLOQUANT : s'il est tenu, il ne fait rien.
LOCK_FILE = Path("/tmp/kolbo-supervisor-bench.lock")
LOCK_TIMEOUT_S = int(os.getenv("BENCH_LOCK_TIMEOUT_S", "120"))  # tolérance au démarrage
LAST_RC_FILE = Path("/tmp/kolbo-supervisor-bench.last-rc")

# Chat de l'agent : override explicite, sinon découverte (active=True,
# repli : le plus récent daté).
OVERRIDE_CHAT_ID = os.getenv("OWUI_CHAT_ID", "").strip()

# Service systemd à VÉRIFIER (jamais à stopper/restart) après le run.
OLLAMA_SERVICE = os.getenv("BENCH_OLLAMA_SERVICE", "ollama-v100")

# Pause après la fin du bench : 10 s suffisent pour que le processus
# ollama reparte ; le modèle se chargera à l'envoi du message de reprise.
POST_RUN_PAUSE_S = int(os.getenv("BENCH_POST_RUN_PAUSE_S", "10"))

CONNECT_TIMEOUT = 10
# Le tir de reprise est non bloquant : on attend juste l'acceptation du
# POST (le service down peut retarder la 1re réponse d'OWUI), puis on ferme.
FIRE_WAIT_S = int(os.getenv("BENCH_FIRE_WAIT_S", "45"))

# Message de reprise : UNIQUE, identique que le service soit up ou down
# (guile : pas de message spécial — la 2e erreur est la preuve de vie).
RESUME_PROMPT = (
    "REPRISE AUTOMATIQUE (superviseur bench) : ton inférence a été interrompue "
    "par un run de test V100. Le run est terminé. Reprends EXACTEMENT "
    "l'action en cours là où elle s'est arrêtée — ne rejoue PAS les étapes "
    "déjà accomplies, ne redemande PAS d'information, et continue sans "
    "préambule sur l'interruption."
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(process)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("supervisor-bench")

session = requests.Session()
session.headers.update({"Accept": "application/json",
                        "Content-Type": "application/json"})


def load_token():
    global OWUI_TOKEN
    try:
        tok = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if tok:
            OWUI_TOKEN = tok
            return True
    except (OSError, ValueError):
        pass
    OWUI_TOKEN = os.getenv("OWUI_API_KEY") or os.getenv("OWUI_TOKEN") or ""
    return bool(OWUI_TOKEN)


# ---------------------------------------------------------------------
# HTTP OWUI
# ---------------------------------------------------------------------

def api_get(path):
    r = session.get(OWUI_URL + path,
                    timeout=(CONNECT_TIMEOUT, 120))
    r.raise_for_status()
    return r.json()


def api_post(path, payload):
    r = session.post(OWUI_URL + path, json=payload,
                     timeout=(CONNECT_TIMEOUT, 120))
    r.raise_for_status()
    return r


# ---------------------------------------------------------------------
# Chat de l'agent
# ---------------------------------------------------------------------

def list_chats():
    data = api_get("/api/v1/chats/?page=1")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("items", "chats", "data"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError("Format inattendu pour GET /api/v1/chats/?page=1")


def discover_chat():
    """active=True d'abord, repli : le plus récent daté."""
    chats = list_chats()
    active, dated = None, []
    for item in chats:
        if not isinstance(item, dict):
            continue
        cid = item.get("id")
        if not cid:
            continue
        if item.get("active") is True:
            active = str(cid)
        ts = (item.get("updated_at") or item.get("updatedAt")
              or item.get("timestamp") or item.get("created_at") or 0)
        try:
            ts = float(ts)
        except (TypeError, ValueError):
            ts = 0
        dated.append((ts, str(cid)))
    if active:
        return active
    dated = [d for d in dated if d[0] > 0]
    if not dated:
        return None
    dated.sort(reverse=True)
    return dated[0][1]


def select_chat():
    if OVERRIDE_CHAT_ID:
        return OVERRIDE_CHAT_ID
    return discover_chat()


def extract_current(chat):
    chat_data = chat.get("chat")
    if not isinstance(chat_data, dict):
        raise ValueError("Champ racine 'chat' invalide.")
    history = chat_data.get("history")
    if not isinstance(history, dict):
        raise ValueError("chat['chat']['history'] absent.")
    messages = history.get("messages")
    if not isinstance(messages, dict):
        raise ValueError("history['messages'] absent.")
    current_id = history.get("currentId") or chat.get("current_message_id")
    if not current_id:
        raise ValueError("Message courant introuvable.")
    msg = messages.get(str(current_id))
    if not isinstance(msg, dict):
        raise ValueError(f"Message {current_id} absent du graphe.")
    return str(current_id), msg


def has_error(msg):
    """error non vide (dict/str non vide, ou bool True) — même si
    done=True (done=True et error coexistent : constat v4)."""
    error = msg.get("error")
    if error in (None, False, ""):
        return False
    if isinstance(error, (dict, list, str)):
        return len(error) > 0
    return True


def resolve_terminal_id(model):
    """Terminal à réinjecter dans le tir de reprise. Le tir ne passe pas
    par le frontend : le store navigateur (selectedTerminalId, mémoire
    seule) ne peut pas fournir le terminal → sans réinjection, le modèle
    perd l'outil terminal après la reprise (diagnostic 16/10). Source de
    vérité : meta.terminalId du modèle dans /api/v1/models. Override : env
    OWUI_TERMINAL_ID. Retourne None si introuvable."""
    override = os.getenv("OWUI_TERMINAL_ID", "").strip()
    if override:
        return override
    try:
        data = api_get("/api/v1/models")
    except requests.RequestException as e:
        log.warning("terminal_id : GET /api/v1/models a échoué : %s", e)
        return None
    models = data.get("data") if isinstance(data, dict) else None
    for m in models or []:
        if not isinstance(m, dict) or m.get("id") != model:
            continue
        meta = (m.get("info") or {}).get("meta") or {}
        tid = meta.get("terminalId")
        if isinstance(tid, str) and tid.strip():
            return tid.strip()
        return None
    log.warning("terminal_id : modèle %s introuvable dans /api/v1/models.",
                model)
    return None


# ---------------------------------------------------------------------
# Reprise (IDs déterministes : idempotent si relancé)
# ---------------------------------------------------------------------

def build_resume_nodes(chat, message_id, msg, prompt=None, prefix="kolbo-resume"):
    """Nouveaux nœuds (user + assistant) branchés SUR le nœud en erreur,
    qui est conservé. IDs = uuid5 du parent : jamais de jumeau.

    prefix/prompt paramétrables : le watchdog (chat_watchdog.py) utilise
    "kolbo-wd" + son propre message ; le superviseur garde les valeurs par
    défaut ("kolbo-resume" + RESUME_PROMPT) — comportement inchangé.
    """
    prompt = RESUME_PROMPT if prompt is None else prompt
    model = msg.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Modèle assistant courant invalide.")

    chat_data = chat.get("chat", {})
    nodes = chat_data.get("history", {}).get("messages", {})
    parent = nodes.get(message_id)
    if not isinstance(parent, dict) or parent.get("role") != "assistant":
        raise ValueError("Nœud courant absent ou non assistant.")

    models = chat_data.get("models")
    if not isinstance(models, list) or not models:
        models = [model]

    # Chemin racine → nœud courant (erreur comprise, vides filtrés).
    path, seen, node_id = [], set(), message_id
    while node_id:
        if node_id in seen:
            raise ValueError("Cycle dans le graphe du chat.")
        seen.add(node_id)
        node = nodes.get(node_id)
        if not isinstance(node, dict):
            raise ValueError(f"Nœud absent : {node_id}")
        path.append(node)
        node_id = node.get("parentId")
    path.reverse()

    messages = []
    for node in path:
        role, content = node.get("role"), node.get("content")
        if role in ("user", "assistant", "system") and \
                content not in (None, "", []) and isinstance(content, (str, list)):
            messages.append({"role": role, "content": content})

    user_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{prefix}:{message_id}"))
    assistant_id = str(uuid.uuid5(uuid.NAMESPACE_URL,
                                  f"{prefix}-a:{message_id}"))
    ts = int(time.time())

    user_message = {
        "id": user_id,
        "role": "user",
        "content": prompt,
        "parentId": message_id,
        "childrenIds": [assistant_id],
        "timestamp": ts,
        "models": models,
    }
    assistant_message = {
        "id": assistant_id,
        "role": "assistant",
        "content": "",
        "parentId": user_id,
        "childrenIds": [],
        "model": model,
        "modelName": model,
        "modelIdx": msg.get("modelIdx", 0),
        "done": False,
        "timestamp": ts + 1,
    }
    return messages, user_message, assistant_message, model


def commit_nodes(chat_id, user_message, assistant_message):
    """POST partiel : le backend fusionne les nouveaux nœuds (pas de
    renvoi du parent — validé en v4). Idempotent (IDs déterministes)."""
    api_post(f"/api/v1/chats/{chat_id}", {
        "chat": {"history": {
            "currentId": assistant_message["id"],
            "messages": {
                user_message["id"]: user_message,
                assistant_message["id"]: assistant_message,
            },
        }},
    })


def verify_node(chat_id, assistant_id):
    chat2 = api_get(f"/api/v1/chats/{chat_id}")
    nodes = chat2.get("chat", {}).get("history", {}).get("messages", {})
    if not isinstance(nodes.get(assistant_id), dict):
        raise ValueError(
            f"Nœud assistant {assistant_id} absent après commit.")
    return nodes[assistant_id]


def fire_completion(chat_id, parent_id, user_message, assistant_message,
                    model, messages, prompt=None):
    """Tir NON BLOQUANT : OWUI prend la main et attend à notre place.
    On attend juste l'acceptation du POST (FIRE_WAIT_S max si le backend
    est down), puis on ferme le stream — la génération continue côté
    OWUI quoi qu'il arrive.

    Réinjecte terminal_id (meta.terminalId du modèle) : sans ça, le
    modèle repart sans l'outil terminal (le tir n'a pas de frontend).
    Ne JAMAIS ajouter de clé 'tools' au payload : elle court-circuiterait
    toute la résolution serveur (utils/middleware.py:3025)."""
    prompt = RESUME_PROMPT if prompt is None else prompt
    if not (messages and messages[-1].get("role") == "user"
            and messages[-1].get("content") == prompt):
        messages.append({"role": "user", "content": prompt})

    terminal_id = resolve_terminal_id(model)
    if terminal_id:
        log.info("Reprise : terminal_id=%s réinjecté "
                 "(meta.terminalId du modèle).", terminal_id)
    else:
        log.warning("Reprise : terminal_id introuvable pour %s — le "
                    "modèle part sans outil terminal.", model)

    payload = {
        "chat_id": chat_id,
        "id": assistant_message["id"],
        "parent_id": parent_id,
        "user_message": user_message,
        "model": model,
        "messages": messages,
        "stream": True,
        "session_id": str(uuid.uuid4()),
    }
    if terminal_id:
        payload["terminal_id"] = terminal_id
    r = session.post(OWUI_URL + "/api/chat/completions", json=payload,
                     stream=True,
                     timeout=(CONNECT_TIMEOUT, FIRE_WAIT_S))
    try:
        r.close()  # OWUI garde la main : on ne lit pas le stream.
    except Exception:
        pass


def send_resume(chat_id, chat, message_id, msg, prefix="kolbo-resume",
                prompt=None):
    """Commit des nœuds + tir non bloquant. Retourne l'ID assistant."""
    messages, user_message, assistant_message, model = \
        build_resume_nodes(chat, message_id, msg, prompt=prompt, prefix=prefix)
    commit_nodes(chat_id, user_message, assistant_message)
    verify_node(chat_id, assistant_message["id"])
    fire_completion(chat_id, message_id, user_message, assistant_message,
                    model, messages, prompt=prompt)
    return assistant_message["id"]


# ---------------------------------------------------------------------
# Fin de run
# ---------------------------------------------------------------------

def kill_group(pid, timeout=15):
    """Filet de sécurité : tue l'arbre de l'enfant s'il reste (crash du
    bench). L'enfant a son propre groupe de process (setsid) → pgid=pid."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            return  # plus personne
        deadline = time.time() + (timeout if sig is signal.SIGTERM else 5)
        while time.time() < deadline:
            try:
                os.killpg(pid, 0)
            except (ProcessLookupError, PermissionError):
                return
            time.sleep(0.5)
        log.warning("Groupe %s encore vivant après %s.", pid, sig)
    log.error("Groupe %s : résidus après SIGKILL — vérification manuelle.",
              pid)


def service_state():
    """État du service ollama-v100 — LECTURE SEULE, jamais d'action."""
    try:
        r = subprocess.run(["systemctl", "is-active", OLLAMA_SERVICE],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def check_and_resume():
    """Fin de run : vérification passive du service + reprise si besoin.
    Retourne (ok: bool, note: str)."""
    state = service_state()
    if state == "active":
        log.info("Service %s : active.", OLLAMA_SERVICE)
    else:
        # Pas de recovery : le run peut avoir laissé une config pourrie,
        # le toucher risquerait d'empirer. Intervention humaine.
        log.error("Service %s : %s — le superviseur ne touche RIEN "
                  "(risque de config pourrie). Intervention humaine "
                  "requise : vérifier le service, puis envoyer 'reprends' "
                  "au chat si besoin.", OLLAMA_SERVICE, state)

    time.sleep(POST_RUN_PAUSE_S)  # 10 s : le processus ollama repart.

    chat_id = select_chat()
    if not chat_id:
        log.warning("Aucun chat agent trouvé — reprise impossible.")
        return False, "aucun chat"
    chat = api_get(f"/api/v1/chats/{chat_id}")
    message_id, msg = extract_current(chat)
    if msg.get("role") != "assistant" or not has_error(msg):
        log.info("Chat %s : nœud courant sans erreur (role=%s done=%r) "
                 "→ rien à reprendre.",
                 chat_id[:8], msg.get("role"), msg.get("done"))
        return True, "rien à reprendre"

    err = msg.get("error")
    if isinstance(err, dict):
        err = json.dumps(err, ensure_ascii=False)[:300]
    log.warning("Nœud assistant courant %s a une erreur : %s — "
                "envoi de la reprise.", message_id[:8], str(err)[:300])
    assistant_id = send_resume(chat_id, chat, message_id, msg)
    log.info("Reprise envoyée (nœud assistant %s…) — OWUI prend la main.",
             assistant_id[:8])
    return True, f"reprise envoyée ({assistant_id[:8]})"


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------

def run_bench(bench_path, bench_args):
    """Lance le bench comme enfant (propre groupe), attend sa sortie.
    Retourne le code de sortie TEL QUEL (transparence)."""
    if not os.path.exists(bench_path):
        log.error("Bench introuvable : %s", bench_path)
        return 127
    cmd = [sys.executable, str(bench_path)] + list(bench_args)
    log.info("Lancement du bench : %s", " ".join(cmd))
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd, cwd=D, start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except OSError as e:
        log.error("Échec du lancement du bench : %s", e)
        return 126

    rc = proc.wait()  # pas de timeout : les timeouts sont dans le bench.
    dur = time.time() - t0
    tail = ""
    try:
        tail = "".join(proc.stdout.read().splitlines(True))[-4000:]
    except Exception:
        pass
    if tail.strip():
        log.info("Dernières lignes du bench :\n%s", tail.rstrip())
    log.info("Bench terminé : rc=%d en %.0fs.", rc, dur)

    kill_group(proc.pid)  # no-op si l'enfant est propre.
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Superviseur de run V100 avec reprise d'inférence.")
    ap.add_argument("mode", help="'run' (mode unique)")
    ap.add_argument("bench", help="script de bench à lancer (ex : bench_gen.py)")
    ap.add_argument("bench_args", nargs="*", help="args transmis au bench")
    ap.add_argument("--detach", action="store_true",
                    help="s'auto-détacher (survit à l'interruption de la "
                         "session agent) — à utiliser par l'agent lui-même")
    ap.add_argument("--_detached", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.mode != "run":
        log.error("Mode inconnu : %s (le seul mode est 'run').", args.mode)
        return 2

    if args.detach and not args._detached:
        cmd = [sys.executable, os.path.abspath(__file__)]
        cmd += [a for a in sys.argv[1:] if a != "--detach"] + ["--_detached"]
        with open(LOG_FILE, "a") as f:
            subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True)
        print(f"Superviseur détaché. Log : {LOG_FILE}")
        return 0

    if not load_token():
        log.error("Token absent : crée %s (0600) ou définis OWUI_API_KEY.",
                  TOKEN_FILE)
        return 2
    session.headers["Authorization"] = f"Bearer {OWUI_TOKEN}"

    # Verrou : 1 superviseur à la fois. Tolérance LOCK_TIMEOUT_S : une
    # courte compétition de démarrage (cron watchdog vs superviseur, ou
    # relance rapide) ne doit pas avorter le run — on attend sa
    # libération, pas plus.
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock = LOCK_FILE.open("w", encoding="utf-8")
    acquired = False
    deadline = time.time() + LOCK_TIMEOUT_S
    while True:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
            break
        except BlockingIOError:
            if time.time() >= deadline:
                log.error("Verrou %s tenu après %ds — un superviseur (ou "
                          "un watchdog anormalement long) est en cours ; "
                          "sortie.", LOCK_FILE, LOCK_TIMEOUT_S)
                return 1
            time.sleep(2)

    try:
        log.info("=== RUN supervisé : %s %s ===", args.bench,
                 " ".join(args.bench_args))
        rc = run_bench(args.bench, args.bench_args)

        try:
            ok, note = check_and_resume()
        except (requests.RequestException, ValueError, KeyError,
                TypeError) as e:
            ok, note = False, f"erreur reprise : {e}"
            log.error("Erreur pendant la phase de reprise : %s", e)

        LAST_RC_FILE.write_text(f"{rc} {note}\n")
        log.info("=== FIN du run : rc=%d, %s ===", rc, note)
        return rc
    finally:
        if acquired:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
