#!/usr/bin/env python3
"""Fixture de test pour chat_watchdog.py : crée un chat test avec un nœud
assistant FINI portant l'erreur demandée. Supprime le chat de test à la fin
(si --keep absent).

Scénarios (--error) :
  ctx (défaut)   : erreur de dépassement de contexte → le watchdog doit
                   COMPACTER puis re-tirer (périmètre A).
  transient      : erreur transitoire (runner arrêté) sur un nœud normal →
                   le watchdog doit re-tirer TEL QUEL si le service est
                   active (périmètre B).
  dead_resume    : nœud de REPRISE (parent = user « REPRISE AUTOMATIQUE »)
                   fini avec erreur transitoire → le watchdog ne doit RIEN
                   faire (disjoncteur).
  ceiling        : nœud FINI SANS ERREUR dont le dernier sous-appel a claqué
                   au plafond (usage ≈ ctx live du runner) → le watchdog doit
                   re-tirer DIRECT en SELF-CHECK, SANS compaction forcée
                   (compaction déléguée à OWUI on-request — 27/10 ;
                   périmètre C).
  ceiling_low    : idem mais usage bien sous le plafond → RIEN à faire.
  ceiling_dead   : nœud de REPRISE sans erreur au plafond → disjoncteur
                   (RIEN — borne du 2ᵉ self-check).

USAGE :
  python3 watchdog_test_fixture.py [--error ctx|transient|dead_resume] [--keep]
"""
import argparse
import json
import pathlib
import sys
import time
import uuid

import requests

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import supervisor_bench as sb  # noqa: E402

ERRORS = {
    "ctx": (
        {"error": {"code": 400,
                   "message": "request (245312 tokens) exceeds the available "
                              "context size (200192 tokens), try increasing it",
                   "type": "exceed_context_size_error",
                   "n_prompt_tokens": 245312, "n_ctx": 200192}},
        "ctx",
    ),
    "transient": (
        {"error": {"code": 500,
                   "message": "model runner has unexpectedly stopped, this "
                              "may be due to resource limitations or an "
                              "internal error, check ollama server logs for "
                              "details",
                   "type": "api_error"}},
        "transient",
    ),
}


def make_chat(sb):
    chat = sb.api_post("/api/v1/chats/new", {
        "name": "WD-TEST fixture (à supprimer)",
        "chat": {"history": {"currentId": None, "messages": {}}},
    }).json()
    return chat["id"]


def build_nodes(scenario):
    ts = int(time.time())
    model = "kolbo5"
    # Usage « plafond » : dernier sous-appel pile au num_ctx du runner
    # (valeurs réelles du 27/10 : 83308 + 44690 = 127998 vs 128000). Les
    # cumuls (input/output/total) sont volontairement énormes : la règle
    # DOIT lire prompt/completion_tokens, pas les cumuls.
    usage_ceiling = {"input_tokens": 4292112, "output_tokens": 380756,
                     "total_tokens": 4672868,
                     "prompt_tokens": 83308, "completion_tokens": 44690}
    usage_low = {"input_tokens": 12000, "output_tokens": 3500,
                 "total_tokens": 15500,
                 "prompt_tokens": 12000, "completion_tokens": 3500}
    err_transit = ERRORS["transient"][0]

    if scenario in ("dead_resume", "ceiling_dead"):
        # user normal → assistant OK (vide) → user REPRISE → assistant
        # de reprise FINI (erreur transitoire ou sans erreur au plafond).
        uid = str(uuid.uuid4())
        aid = str(uuid.uuid4())
        ruid = str(uuid.uuid4())
        raid = str(uuid.uuid4())
        user_node = {"id": uid, "role": "user",
                     "content": "fixture watchdog : tâche fictive en cours "
                                "(test disjoncteur)",
                     "parentId": None, "childrenIds": [aid],
                     "timestamp": ts, "models": [model]}
        asst_node = {"id": aid, "role": "assistant", "content": "",
                     "parentId": uid, "childrenIds": [ruid],
                     "model": model, "modelName": model, "modelIdx": 0,
                     "done": True, "timestamp": ts + 1}
        ruser_node = {"id": ruid, "role": "user",
                      "content": "REPRISE AUTOMATIQUE (watchdog) : fixture "
                                 "— parent de reprise morte (test disjoncteur)",
                      "parentId": aid, "childrenIds": [raid],
                      "timestamp": ts + 2, "models": [model]}
        rassist = {"id": raid, "role": "assistant", "content": "",
                   "parentId": ruid, "childrenIds": [],
                   "model": model, "modelName": model, "modelIdx": 0,
                   "done": True, "timestamp": ts + 3}
        if scenario == "ceiling_dead":
            rassist["usage"] = usage_ceiling
        else:
            rassist["error"] = {"content": json.dumps(err_transit,
                                                      ensure_ascii=False)}
        return {uid: user_node, aid: asst_node,
                ruid: ruser_node, raid: rassist}, raid

    # ctx / transient / ceiling / ceiling_low : user normal → assistant
    # FINI (+ erreur, + usage pour ceiling).
    uid = str(uuid.uuid4())
    aid = str(uuid.uuid4())
    user_node = {"id": uid, "role": "user",
                 "content": "fixture watchdog : tâche fictive en cours "
                            "(test %s)" % scenario,
                 "parentId": None, "childrenIds": [aid],
                 "timestamp": ts, "models": [model]}
    asst_node = {"id": aid, "role": "assistant", "content": "",
                 "parentId": uid, "childrenIds": [],
                 "model": model, "modelName": model, "modelIdx": 0,
                 "done": True, "timestamp": ts + 1}
    if scenario in ("ctx", "transient"):
        err, _ = ERRORS[scenario]
        asst_node["error"] = {"content": json.dumps(err,
                                                    ensure_ascii=False)}
    elif scenario == "ceiling":
        asst_node["usage"] = usage_ceiling
    elif scenario == "ceiling_low":
        asst_node["usage"] = usage_low
    return {uid: user_node, aid: asst_node}, aid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--error", choices=sorted(ERRORS) +
                    ["dead_resume", "ceiling", "ceiling_low", "ceiling_dead"],
                    default="ctx")
    ap.add_argument("--keep", action="store_true",
                    help="ne pas supprimer le chat de test")
    args = ap.parse_args()

    if not sb.load_token():
        print("token absent")
        return 2
    sb.session.headers["Authorization"] = f"Bearer {sb.OWUI_TOKEN}"

    cid = make_chat(sb)
    print("chat test créé :", cid)
    nodes, cur = build_nodes(args.error)
    sb.api_post(f"/api/v1/chats/{cid}", {
        "chat": {"history": {
            "currentId": cur,
            "messages": nodes,
        }}})
    print(f"scénario      : {args.error}")
    print("nœud courant  :", cur)
    print("CHAT_TEST_ID=" + cid)
    if not args.keep:
        try:
            sb.session.delete(sb.OWUI_URL + f"/api/v1/chats/{cid}", timeout=30)
            print("chat test supprimé.")
        except requests.RequestException as e:
            print("ATTENTION — suppression échouée, chat à supprimer à la "
                  "main : %s (%s)" % (cid, e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
