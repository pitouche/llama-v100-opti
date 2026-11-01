#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chat_watchdog.py — garde de veille des fins de génération OWUI (cron 5 min).

Missions (uniquement les échecs que le superviseur ne couvre PAS) :
  A. le nœud assistant courant du chat agent est FINI (done=True) avec une
     erreur de DÉPASSEMENT DE CONTEXTE (ex. "request (405240 tokens) exceeds
     the available context size (200192 tokens)" — un appel outil qui renvoie
     trop de données, ou un contexte qui a atteint sa limite) → COMPACTER
     puis relancer ;
  B. (17/10) le nœud assistant courant est FINI (done=True) avec une erreur
     TRANSITOIRE (runner de modèle arrêté à chaud "model runner has
     unexpectedly stopped…", "Ollama: Cannot connect…", …) → re-tir TEL
     QUEL, sans compactation, si :
       - le service ollama-v100 est `active` (lecture passive, le watchdog
         ne stoppe JAMAIS le service et ne le redémarre JAMAIS — sinon
         RIEN, intervention humaine) ;
       - le nœud n'est PAS une reprise morte (DISJONCTEUR : si le nœud en
         erreur est l'assistant d'une reprise déjà tirée — superviseur ou
         watchdog — alors RIEN, intervention humaine. 1 auto-reprise max
         par origine d'erreur : anti-boucle sur instabilité persistante,
         ex. .so corrompu qui re-crash à chaque re-tir).
      C'est le filet sur la zone résiduelle du superviseur : le tir de
      reprise du superviseur est non bloquant — s'il échoue PENDANT la
      génération (runner mort à chaud), la fin de run est déjà passée ;
      c'est le cas B qui le rattrape au tick suivant. (Chat empoisonné,
      ex. "Failed to tokenize prompt" — remisé : le re-tir probe puis le
      disjoncteur arrête la chaîne.)
  C. (27/10) PLAFOND SILENCIEUX : le nœud assistant courant est FINI
     (done=True) SANS AUCUNE erreur, mais le dernier sous-appel a claqué
     pile au plafond du contexte du runner : l'usage rapporté (usage du
     nœud = usage du DERNIER sous-appel : prompt_tokens +
     completion_tokens) est >= context_length(runner live) − 4 (marge
     MTP : un pas peut émettre plusieurs tokens, le total peut atterrir
     quelques tokens sous le plafond). Le stream se termine « proprement »
     : il n'y a PAS d'erreur stockée → les cas A/B ne voient rien. Le
     travail est tronqué au milieu de la tâche (cas réel 27/10 : chat
     fae260af, 127998/128000, arrêt en plein function_call du T3 v2).
     Action : re-tir DIRECT en SELF-CHECK (PAS de compaction forcée —
     depuis le 27/10) : le prompt demande au modèle de vérifier lui-même
     si la tâche était interrompue ; s'il dit que c'est terminé, il
     s'arrête (coût du faux positif : un mini tour) ; sinon il reprend.
     La compaction est déléguée à OWUI : chaque tir passe par
     `compact_messages_for_request` dans `process_chat_payload` (middleware)
     qui, si le seuil (chat.context_compaction.token_threshold) est
     dépassé, résume le contexte AVANT la génération. Validé 27/10 sur
     fae260af : tir direct à 16:00 → OWUI a compacté on-request (16:03,
     dropped=3) et le modèle a repris pile au point d'interruption ;
     un continue manuel 4 h plus tard a été compacté de même (20:33) et
     le run est allé au bout. Le `/compact` explicite a, lui, échoué en
     cascade (500 MTP « failed to flush deferred speculative prompt
     prefix » → timeout 180 s → `too_short` → faux « Compaction
     effective ») — il générait en plus un prompt de ~127k tokens sur le
     MÊME runner à 128k. Si la compaction on-request ne peut pas assez
     réduire (nœud géant sans checkpoint interne — le tir repart à ~127k
     et recolle au plafond), le nœud de re-tir est une reprise →
     DISJONCTEUR → humain (comportement identique à l'ancien, sans la
     cascade).

Le DISJONCTEUR est global (A, B et C) : si le nœud est une reprise morte
(son parent est un user « REPRISE AUTOMATIQUE » — superviseur OU
watchdog), RIEN — 1 auto-reprise max par origine. Ça borne aussi le
self-check : si le re-tir est lui-même long et recolle au plafond, le
nouveau nœud est une reprise → pas de 2ᵉ self-check.

Si le cas A se présente :
  1. COMPACTER le contexte du chat via `POST /api/v1/chats/<id>/compact`
     (OWUI 0.12.2 : le résumé de branche est écrit sur le nœud courant sous
     `contextSummary` ; à la prochaine génération le contexte est replacé
     par le résumé + les messages récents — c'est le mécanisme interne du
     "/COMPACT" de l'UI, appelé ici côté serveur). La compaction forcée
     reste ici parce que l'erreur de contexte du tir précédent prouve que
     le prompt tel quel dépasse le plafond : sans résumé préalable, le
     re-tir retomberait sur la même erreur. (Le cas C, lui, ne compacte
     PLUS : la compaction on-request d'OWUI au moment du tir suffit —
     voir docstring du cas C et README.)
  2. puis RELAUNCHER l'inférence avec exactement la même mécanique que le
     superviseur (nœuds déterministes `kolbo-wd:` / `kolbo-wd-a:`, commit
     partiel, tir non bloquant, réinjection terminal_id, JAMAIS de clé
     `tools`).

Aware du superviseur : même verrou que lui (`/tmp/kolbo-supervisor-bench.lock`,
flock EX). Le watchdog ne tente le verrou qu'une seule fois, NON bloquant :
s'il est tenu (superviseur en cours, de la fin du bench à la reprise), il
sort sans rien faire. C'est le seul point de collision possible — il est
fermé par construction. Idempotence : les IDs de nœud dépendent uniquement
du nœud en erreur → aucune reprise ne peut être doublée.

USAGE (cron, user guile, toutes les 5 min) :
  * * * * * cd /home/guile/llama-experiments/bench && /home/guile/venv.3.12.13/bin/python3.12 chat_watchdog.py >> /home/guile/llama-experiments/bench/chat_watchdog.cron.log 2>&1

LOG : bench/chat_watchdog.log (+ miroir cron.log pour le cron)
Env : OWUI_URL, OWUI_CHAT_ID (override chat), OWUI_TERMINAL_ID (override
terminal), WATCHDOG_MAX_RUN_S (safety, défaut 300).
"""
import fcntl
import json
import logging
import os
import signal
import sys
import time
import uuid
from pathlib import Path

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import supervisor_bench as sb  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(D, "chat_watchdog.log")

# Safety globale (surtout pour les timeouts de la compaction).
MAX_RUN_S = int(os.getenv("WATCHDOG_MAX_RUN_S", "300"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(process)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("chat-watchdog")

WD_PREFIX = "kolbo-wd"
WD_PROMPT = (
    "REPRISE AUTOMATIQUE (watchdog) : ta génération a échoué par dépassement "
    "de contexte (l'appel a renvoyé trop de données, ou le contexte a atteint "
    "sa limite). Le contexte a été compressé automatiquement avant cette "
    "reprise. Reprends EXACTEMENT l'action en cours là où elle s'est arrêtée "
    "— ne rejoue PAS les étapes déjà accomplies, ne redemande PAS "
    "d'information, et continue sans préambule sur l'interruption."
)
WD_TRANSIT_PROMPT = (
    "REPRISE AUTOMATIQUE (watchdog) : ta génération a été interrompue par "
    "une erreur transitoire du serveur (runner de modèle arrêté ou "
    "connexion perdue). Le service est de nouveau opérationnel. Reprends "
    "EXACTEMENT l'action en cours là où elle s'est arrêtée — ne rejoue PAS "
    "les étapes déjà accomplies, ne redemande PAS d'information, et "
    "continue sans préambule sur l'interruption."
)
WD_CEILING_PROMPT = (
    "REPRISE AUTOMATIQUE (watchdog) : ta précédente génération s'est "
    "terminée pile à la limite du contexte maximum — elle a pu être "
    "interrompue. Si le contexte a été trop long, il a été compressé "
    "automatiquement avant cette reprise. Avant toute action, vérifie : "
    "la tâche en cours était-elle réellement terminée ? Si elle a été "
    "interrompue, reprends EXACTEMENT là où tu t'es arrêté — sans rejouer "
    "les étapes déjà accomplies. Si elle était terminée, dis-le brièvement "
    "et n'entreprends rien d'autre."
)

# Marge du plafond silencieux (cas C, 27/10) : trigger quand
# usage(prompt+completion du dernier sous-appel) >= ctx_live − marge.
# Marge MTP (guile) : avec MTP un pas peut émettre 2+ tokens, le total
# rapporté peut atterrir quelques tokens SOUS le plafond (cas réel :
# 127998/128000). ≥ Cmax−4 le capte ; un strict == Cmax l'aurait manqué.
WD_CEILING_MARGIN = int(os.getenv("WD_CEILING_MARGIN", "4"))

# URL du serveur ollama (lecture passive /api/ps : runners chargés +
# leur context_length réel). Par défaut 127.0.0.1:11434 (ollama-v100 ;
# voir le unit system : le service ne définit pas de OLLAMA_HOST → port
# par défaut). Override : env WD_OLLAMA_URL.
WD_OLLAMA_URL = os.getenv("WD_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")

# Signature commune des nœuds user de reprise (superviseur ET watchdog) —
# c'est elle qui détecte une REPRISE MORTE pour le disjoncteur (voir
# is_dead_resume). Identique à la signature de vérification du README.
RESUME_SIGNATURE = "REPRISE AUTOMATIQUE"

# Signatures d'erreur de dépassement de contexte (périmètre du watchdog).
# Le reste (Ollama: Cannot connect, modèle absent, OOM, …) n'est PAS du
# watchdog : le superviseur le reprend après le run, ou guile intervient.
CTX_SIGNATURES = (
    "exceed_context_size_error",
    "exceeds the available context size",
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "prompt too long",
)


def _alarm(_signum, _frame):
    raise TimeoutError("watchdog : timeout global dépassé")


def error_text(msg):
    """Texte de l'erreur d'un nœud (dict {'content': …}, str, …)."""
    err = msg.get("error")
    if err in (None, False, ""):
        return ""
    if isinstance(err, dict):
        return json.dumps(err, ensure_ascii=False)
    return str(err)


def is_context_error(text):
    t = text.lower()
    return any(sig in t for sig in CTX_SIGNATURES)


def is_dead_resume(msg, nodes):
    """Le nœud courant est-il une REPRISE MORTE : son parent est-il un nœud
    user de reprise (contenu commençant par « REPRISE AUTOMATIQUE »,
    superviseur ou watchdog) ? C'est le DISJONCTEUR, GLOBAL (A, B et C) :
    une reprise dont la génération a échoué (ou recollé au plafond) n'est
    JAMAIS re-tirée — la chaîne d'auto-reprises est bornée à 1 par origine
    d'erreur (anti-boucle sur instabilité persistante, ex. .so corrompu qui
    re-crash à chaque re-tir)."""
    pid = msg.get("parentId")
    if not pid or not isinstance(nodes, dict):
        return False
    parent = nodes.get(str(pid))
    if not isinstance(parent, dict):
        return False
    content = parent.get("content")
    if isinstance(content, str):
        return content.startswith(RESUME_SIGNATURE)
    if isinstance(content, list):  # contenu en parts
        texts = [p.get("text", "") for p in content
                 if isinstance(p, dict)]
        return bool(texts) and texts[0].startswith(RESUME_SIGNATURE)
    return False


def acquire_watchdog_lock():
    """Tente UNE fois le verrou du superviseur, NON bloquant. Retourne le
    fd (gouverner jusqu'à la fin) ou None (superviseur en cours → inaction)."""
    lock = sb.LOCK_FILE.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return None
    return lock


def live_context_length(model):
    """context_length du runner live du modèle (plafond contextuel réel).
    Modèle du nœud → base_model_id (via /api/v1/models) → runner live
    (via /api/ps, LECTURE PASSIVE). Retourne None si introuvable (le cas
    plafond n'est PAS déclenché : on ne devine pas un plafond)."""
    base = None
    try:
        models = sb.api_get("/api/v1/models").get("data") or []
    except (requests.RequestException, AttributeError, ValueError) as e:
        log.warning("plafond silencieux : GET /api/v1/models a échoué : %s",
                    e)
        return None
    for m in models:
        if isinstance(m, dict) and m.get("id") == model:
            base = (m.get("info") or {}).get("base_model_id") or model
            break
    if not base:
        log.warning("plafond silencieux : modèle %s inconnu (pas de "
                    "base_model_id) — pas de déclenchement.", model)
        return None
    try:
        runners = sb.session.get(WD_OLLAMA_URL + "/api/ps",
                                 timeout=(sb.CONNECT_TIMEOUT, 15)).json()
    except (requests.RequestException, ValueError) as e:
        log.warning("plafond silencieux : GET %s/api/ps a échoué : %s",
                    WD_OLLAMA_URL, e)
        return None
    for r in runners.get("models") or []:
        if isinstance(r, dict) and r.get("name") == base:
            ctx = r.get("context_length")
            return ctx if isinstance(ctx, int) and ctx > 0 else None
    log.warning("plafond silencieux : runner %s non chargé dans /api/ps "
                "(pas de contexte live connu) — pas de déclenchement.",
                base)
    return None


def last_subcall_tokens(msg):
    """Tokens du DERNIER sous-appel du nœud (usage.prompt_tokens +
    usage.completion_tokens). L'usage du nœud est celui du dernier
    sous-appel — PAS le cumul (input_tokens/output_tokens, qui agrègent
    tous les sous-appels du tour et casseraient la règle). Retourne None
    si usage absent/incomplet."""
    usage = msg.get("usage")
    if not isinstance(usage, dict):
        return None
    p, c = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if not isinstance(p, int) or not isinstance(c, int):
        return None
    return p + c


def compact_chat(chat_id, timeout=180):
    """Compaction de la branche courante (mécanisme '/COMPACT' côté serveur).
    L'appel peut durer plusieurs minutes (le résumé est généré par le
    modèle) → timeout long, mais borné par le safety global."""
    r = sb.session.post(
        sb.OWUI_URL + f"/api/v1/chats/{chat_id}/compact",
        json={}, timeout=(sb.CONNECT_TIMEOUT, timeout))
    r.raise_for_status()
    try:
        return r.json()
    except ValueError:
        return {}


def has_summary_after(chat_id):
    """Le nœud courant porte-t-il un contextSummary (compaction effective) ?"""
    chat = sb.api_get(f"/api/v1/chats/{chat_id}")
    history = chat.get("chat", {}).get("history", {})
    nodes = history.get("messages", {})
    cur = history.get("currentId")
    node = nodes.get(str(cur)) or {}
    summary = node.get("contextSummary")
    return isinstance(summary, str) and bool(summary.strip()), str(cur)


def main():
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(MAX_RUN_S)

    lock = acquire_watchdog_lock()
    if lock is None:
        log.info("Verrou superviseur tenu — superviseur en cours, "
                 "rien à faire.")
        return 0
    try:
        return watch()
    finally:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass


def watch():
    if not sb.load_token():
        log.error("Token OWUI absent — sortie.")
        return 2
    sb.session.headers["Authorization"] = f"Bearer {sb.OWUI_TOKEN}"

    chat_id = sb.select_chat()
    if not chat_id:
        log.info("Aucun chat agent trouvé — rien à faire.")
        return 0

    chat = sb.api_get(f"/api/v1/chats/{chat_id}")
    try:
        message_id, msg = sb.extract_current(chat)
    except (ValueError, KeyError) as e:
        log.warning("Chat %s : %s — rien à faire.", chat_id[:8], e)
        return 0

    # Périmètre 1 : le nœud doit être FINI (une génération en cours,
    # done=False, n'est jamais touchée).
    if msg.get("done") is not True:
        log.info("Chat %s : nœud courant %s en cours (done=%r) — rien à "
                 "faire.", chat_id[:8], message_id[:8], msg.get("done"))
        return 0

    # Le nœud courant doit être un ASSISTANT (cible d'une reprise).
    if msg.get("role") != "assistant":
        log.info("Chat %s : nœud courant %s non assistant (role=%r) — rien "
                 "à faire (cas dégénéré, intervention humaine).",
                 chat_id[:8], message_id[:8], msg.get("role"))
        return 0

    nodes = ((chat.get("chat") or {}).get("history") or {}).get("messages")

    text = error_text(msg)

    # DISJONCTEUR (global, A, B et C) : si le nœud est une REPRISE MORTE
    # (son parent est un nœud user « REPRISE AUTOMATIQUE » — superviseur
    # OU watchdog), on ne re-tire JAMAIS : 1 auto-reprise max par origine
    # (anti-boucle, ex. .so corrompu qui re-crash à chaque re-tir) →
    # intervention humaine. Capte aussi un self-check (C) qui recollerait
    # au plafond.
    if is_dead_resume(msg, nodes):
        log.error("Chat %s : REPRISE MORTE sur %s (le re-tir n'a pas "
                  "abouti : erreur ou repli plafond) — disjoncteur : RIEN "
                  "à faire. Intervention humaine requise (1 auto-reprise "
                  "max par origine) : %.200s",
                  chat_id[:8], message_id[:8],
                  text or "(sans erreur — repli plafond, cf. usage)")
        return 0

    if not text:
        # Périmètre C (27/10) : PLAFOND SILENCIEUX. Nœud FINI SANS ERREUR
        # dont le dernier sous-appel a claqué pile au plafond du contexte
        # du runner (>= ctx live − marge MTP). Signal quantitatif, pas
        # textuel (il n'y a AUCUNE erreur stockée : le stream s'est
        # terminé « proprement » en fin de contexte).
        used = last_subcall_tokens(msg)
        model = msg.get("model")
        ctx = live_context_length(model) if isinstance(model, str) else None
        if ctx is None or used is None or used < ctx - WD_CEILING_MARGIN:
            log.info("Chat %s : nœud courant %s fini, sans erreur, hors "
                     "plafond (usage_dernier_appel=%s ctx_live=%s) — rien "
                     "à faire.",
                     chat_id[:8], message_id[:8], used, ctx)
            return 0
        log.warning("Chat %s : nœud %s FINI SANS ERREUR mais au PLAFOND du "
                    "contexte (dernier sous-appel %s tokens vs ctx live %s "
                    "− marge %s) — probable coupure au max contexte "
                    "(re-tir direct SELF-CHECK : la compaction est déléguée "
                    "à l'auto-compaction on-request d'OWUI au moment du "
                    "tir — 27/10, plus de /compact explicite ici).",
                    chat_id[:8], message_id[:8], used, ctx,
                    WD_CEILING_MARGIN)
        assistant_id = sb.send_resume(chat_id, chat, message_id, msg,
                                      prefix=WD_PREFIX,
                                      prompt=WD_CEILING_PROMPT)
        log.info("Reprise watchdog (plafond, self-check, nœud assistant "
                 "%s…) — OWUI prend la main.", assistant_id[:8])
        return 0
    elif is_context_error(text):
        # Périmètre A (inchangé) : erreur de CONTEXTE → compact + re-tir
        # (le bloc ci-dessous).
        log.warning("Chat %s : nœud %s FINI avec erreur de CONTEXTE : "
                    "%.200s",
                    chat_id[:8], message_id[:8], text)
    else:
        # Périmètre B (17/10) : erreur TRANSITOIRE (runner arrêté à chaud,
        # Cannot connect…) hors run supervisé → re-tir TEL QUEL, sans
        # compactation. Porte fermée avant le tir : service non `active`
        # → RIEN (le watchdog ne stoppe JAMAIS le service et ne le
        # redémarre JAMAIS — lecture passive).
        state = sb.service_state()
        if state != "active":
            log.warning("Chat %s : erreur transitoire sur %s mais service "
                        "%s = %s — RIEN à faire (le watchdog ne touche "
                        "jamais au service) : %.120s",
                        chat_id[:8], message_id[:8], sb.OLLAMA_SERVICE,
                        state, text)
            return 0
        log.warning("Chat %s : nœud %s FINI avec erreur transitoire "
                    "(service active, pas une reprise) : %.200s — re-tir "
                    "tel quel.",
                    chat_id[:8], message_id[:8], text)
        assistant_id = sb.send_resume(chat_id, chat, message_id, msg,
                                      prefix=WD_PREFIX,
                                      prompt=WD_TRANSIT_PROMPT)
        log.info("Reprise watchdog (transitoire) envoyée (nœud assistant "
                 "%s…) — OWUI prend la main.", assistant_id[:8])
        return 0

    # 1) Compaction forcée (mécanisme '/COMPACT' côté serveur) — cas A.
    #    Le cas C ne passe PAS ici (re-tir direct, docstring du cas C).
    try:
        res = compact_chat(chat_id)
        log.info("Compaction : %s", json.dumps(res, ensure_ascii=False)[:300])
    except Exception as e:
        log.error("Compaction en échec : %s — pas de relance (un tir sans "
                  "compaction échouerait de la même façon).", e)
        return 1

    # Le verdict vient de la RÉPONSE du /compact, pas du contexteSummary
    # stocké : un résumé écrit par une tentative ANTÉRIEURE (achevée côté
    # serveur après un timeout client) peut exister sans que CETTE
    # compaction ait réduit quoi que ce soit — cas réel 27/10 :
    # `compacted: false, reason: too_short` + contexteSummary hérité →
    # faux « Compaction effective » → tir inutile recollé au plafond.
    if res.get("compacted") is not True:
        log.error("Compaction inopérante (compacted=false, reason=%s) — "
                  "pas de relance (le re-tir retomberait sur la même "
                  "erreur de contexte).", res.get("reason"))
        return 1

    ok, cur_id = has_summary_after(chat_id)
    if not ok:
        log.error("Pas de contextSummary sur le nœud courant %s après "
                  "compaction — pas de relance.", cur_id[:8])
        return 1
    log.info("Compaction effective (contextSummary sur le nœud courant %s).",
             cur_id[:8])

    # 2) Relance — même mécanique que le superviseur (prefix kolbo-wd :
    #    les IDs n'entrent jamais en collision avec kolbo-resume).
    #    Le cas C a déjà tiré plus haut (self-check, sans compaction).
    assistant_id = sb.send_resume(chat_id, chat, message_id, msg,
                                  prefix=WD_PREFIX, prompt=WD_PROMPT)
    log.info("Reprise watchdog (nœud assistant %s…) — OWUI prend la main.",
             assistant_id[:8])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TimeoutError as e:
        log.error("%s", e)
        sys.exit(1)
    except Exception as e:
        log.exception("Erreur inattendue : %s", e)
        sys.exit(1)
