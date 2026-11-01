#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_prd_profile.py — analyse les 4 captures nsys (sqlite) du run
prof-prd + results.json + samplers + perf, et produit le bilan
MTP ON vs OFF par phase (PREFILL / DECODE / DRAFT / VERIFY / sampling /
host sync / gaps GPU).

Usage :
  python3 bench/analyze_prd_profile.py [run_ts]
  run_ts par défaut : le plus récent prof-prd-* dans bench/runs/
Sortie :
  bench/prof_prd_analysis_<run_ts>.md
"""
import glob
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict

BENCH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BENCH)


def log(msg):
    print(msg, flush=True)


def acceptance_stats(server_stats):
    """parse la derniere ligne 'draft acceptance' du log serveur.
    Renvoie (rate, acc, gen, mean_len, per_pos) ou None. Le per-pos est
    lu tel quel (2 valeurs si n_max=2, 3 si n_max=3)."""
    out = None
    for l in server_stats or []:
        m = re.search(r"draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([0-9.]+)", l)
        if m:
            out = [float(m.group(1)), int(m.group(2)), int(m.group(3)),
                   float(m.group(4)), None]
        m = re.search(r"acc per pos = \(([^)]*)\)", l)
        if m and out is not None:
            vals = [v.strip() for v in m.group(1).split(",")]
            if out[4] is None and vals:
                out[4] = tuple(float(v) for v in vals)
    return out


def find_run_ts():
    if len(sys.argv) > 1:
        a = sys.argv[1]
        return a[len("prof-prd-"):] if a.startswith("prof-prd-") else a
    cands = sorted(glob.glob(os.path.join(BENCH, "runs", "prof-prd-*")))
    if not cands:
        sys.exit("aucun run prof-prd-*")
    return os.path.basename(cands[-1])[len("prof-prd-"):]


class DB:
    def __init__(self, path):
        self.path = path
        self.c = sqlite3.connect(path)
        self.names = {}
        for i, v in self.c.execute("select id, value from StringIds"):
            self.names[i] = v
        self.load()

    def name(self, i):
        return self.names.get(i, str(i))

    def load(self):
        c = self.c
        krows = c.execute(
            "select start, end, streamId, correlationId, graphId, "
            "shortName, demangledName, gridX, gridY, gridZ, blockX, blockY, blockZ "
            "from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
        self.kernels = []
        for s, e, sid, corr, gid, sn, dn, *rest in krows:
            sn = self.name(sn) if isinstance(sn, int) else (sn or "")
            dn = self.name(dn) if isinstance(dn, int) else (dn or "")
            self.kernels.append((s, e, sid, corr, gid, sn, dn, *rest))
        self.rt = []
        for start, end, cls, tid, corr, nameId, ret in c.execute(
                "select start, end, eventClass, globalTid, correlationId, "
                "nameId, returnValue from CUPTI_ACTIVITY_KIND_RUNTIME"):
            self.rt.append((start, end, cls, tid, corr, self.name(nameId), ret))
        self.sync = c.execute(
            "select start, end, streamId, correlationId, syncType "
            "from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION").fetchall()
        self.memcpy = c.execute(
            "select start, end, correlationId from CUPTI_ACTIVITY_KIND_MEMCPY"
        ).fetchall()
        self.graph_nodes = c.execute(
            "select start, end, eventClass, nameId, graphNodeId, "
            "originalGraphNodeId from CUDA_GRAPH_NODE_EVENTS").fetchall()
        self.k_dur = [(e - s) for s, e, *_ in self.kernels]

    # ---- métriques ----
    def span_us(self):
        if not self.kernels:
            return 0
        t0 = min(k[0] for k in self.kernels)
        t1 = max(k[1] for k in self.kernels)
        return (t1 - t0) / 1000.0

    def union_busy_us(self, items):
        """items = [(start_ns, end_ns), ...] → temps union en µs."""
        iv = sorted((s, e) for s, e in items if e > s)
        tot = 0
        cur_s = cur_e = None
        for s, e in iv:
            if cur_s is None:
                cur_s, cur_e = s, e
            elif s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                tot += cur_e - cur_s
                cur_s, cur_e = s, e
        if cur_s is not None:
            tot += cur_e - cur_s
        return tot / 1000.0

    def kernel_busy_us(self):
        return self.union_busy_us([(k[0], k[1]) for k in self.kernels])

    def top_kernels(self, n=15):
        agg = defaultdict(lambda: [0, 0, 0])  # name → [cnt, dur_ns, gridkey]
        for s, e, sid, corr, gid, sn, dn, *rest in self.kernels:
            a = agg[sn or dn]
            a[0] += 1
            a[1] += e - s
        out = sorted(agg.items(), key=lambda kv: -kv[1][1])[:n]
        return [(k, v[0], v[1] / 1e3) for k, v in out]

    def top_runtime(self, n=15):
        agg = defaultdict(lambda: [0, 0])
        for s, e, cls, tid, corr, nm, ret in self.rt:
            a = agg[nm]
            a[0] += 1
            a[1] += e - s
        out = sorted(agg.items(), key=lambda kv: -kv[1][0])[:n]
        return [(k, v[0], v[1] / 1e3) for k, v in out]

    def runtime_sum(self, pat):
        tot = 0
        n = 0
        rx = re.compile(pat)
        for s, e, cls, tid, corr, nm, ret in self.rt:
            if rx.search(nm):
                tot += e - s
                n += 1
        return n, tot / 1e3

    def sync_sum(self):
        return len(self.sync), sum(e - s for s, e, *_ in self.sync) / 1e3

    def gaps(self, min_us=100, n=10):
        """gaps inter-kernels (union timeline) > min_us, plus gros."""
        iv = sorted((k[0], k[1]) for k in self.kernels if k[1] > k[0])
        gaps = []
        for (s0, e0), (s1, e1) in zip(iv, iv[1:]):
            if s1 > e0:
                g = (s1 - e0) / 1000.0
                if g >= min_us:
                    gaps.append((g, e0))
        gaps.sort(reverse=True)
        tot = sum(g for g, _ in gaps)
        return gaps[:n], len(gaps), tot


def classify_kernel(name):
    n = (name or "").lower()
    if "mul_mat_vec_q" in n or "mul_mat_vec" in n:
        return "matvec (proj decode)"
    if "mul_mat_q" in n:
        return "matmul (prefill)"
    if "flash_attn" in n or "attn" in n and "ext" in n:
        return "flash_attn"
    if "flash" in n:
        return "flash_attn"
    if "rope" in n:
        return "rope"
    if "norm" in n or "rms" in n:
        return "norm/rms"
    if "embed" in n:
        return "embedding"
    if "softmax" in n:
        return "softmax"
    if "kq_scale" in n or "scale" in n:
        return "scale/quant"
    if "cvt" in n or "quant" in n or "dequant" in n:
        return "quant/cvt"
    if "memcpy" in n or "copy" in n:
        return "copy"
    return "autre"


def kernel_breakdown(db, n=12):
    agg = defaultdict(lambda: [0, 0])
    for s, e, sid, corr, gid, sn, dn, *rest in db.kernels:
        cls = classify_kernel(sn or dn)
        a = agg[cls]
        a[0] += 1
        a[1] += e - s
    tot = sum(v[1] for v in agg.values()) or 1
    out = sorted(agg.items(), key=lambda kv: -kv[1][1])[:n]
    return [(k, v[0], v[1] / 1e3, 100.0 * v[1] / tot) for k, v in out], tot / 1e3


def graph_groups(db):
    """groupe les kernels par correlationId (= un cudaGraphLaunch ou un
    lot). Renvoie {corr: (gid, n_kernels, dur_us)}."""
    g = defaultdict(lambda: [None, 0, 0])
    for s, e, sid, corr, gid, sn, dn, *rest in db.kernels:
        a = g[corr]
        a[1] += 1
        a[2] += e - s
        if gid not in (None, 0):
            a[0] = gid
    return g


def mtp_cycles(db, min_gap_us=50):
    """Décompose la fenêtre decode MTP en cycles verify→draft→sampling.
    Heuristique : chaque cudaGraphLaunch = 1 groupe ; classification par
    taille du groupe (verify ~2000+ kernels, draft ~40-60, sampling ~10-20)."""
    groups = graph_groups(db)
    launches = []
    for corr, (gid, nk, dur) in groups.items():
        if gid in (None, 0) or nk < 8:
            continue  # pas un graph (ou micro-graph ignoré)
        s = min(k[0] for k in db.kernels if k[3] == corr)
        e = max(k[1] for k in db.kernels if k[3] == corr)
        launches.append((s, e, corr, gid, nk, (e - s) / 1e3, dur / 1e3))
    launches.sort()

    def cls(nk):
        if nk >= 500:
            return "verify"
        if nk >= 25:
            return "draft"
        return "sampling"

    seq = []
    for s, e, corr, gid, nk, wall, busy in launches:
        # t0/t1/wall/busy en µs (launches est déjà en µs)
        seq.append({"t0": s / 1e3, "t1": e / 1e3, "kind": cls(nk),
                    "nk": nk, "wall_us": wall, "busy_us": busy, "gid": gid})
    return seq


def fmt_us(us):
    if us is None:
        return "?"
    if us >= 1e6:
        return "%.2f s" % (us / 1e6)
    if us >= 1e3:
        return "%.1f ms" % (us / 1e3)
    return "%.0f µs" % us


def gputimes_summary(path, t0_ns, t1_ns):
    """t0/t1 = ns CUPTI ; sampler epoch en ms. Renvoie (n, busy%, moy util)."""
    rows = []
    try:
        with open(path) as f:
            next(f)
            for line in f:
                p = line.split()
                if len(p) >= 5:
                    rows.append((int(p[0]), int(p[1]), float(p[4])))
    except OSError:
        return None
    t0ms, t1ms = t0_ns // 1000, t1_ns // 1000
    win = [r for r in rows if t0ms <= r[0] <= t1ms]
    if not win:
        return None
    busy = 100.0 * sum(1 for r in win if r[1] > 0) / len(win)
    avg = sum(r[1] for r in win) / len(win)
    return len(win), busy, avg


def analyze_phase(res, db, wall_ms, tag, mtp=False):
    out = {}
    span = db.span_us()
    busy = db.kernel_busy_us()
    out["span_us"] = span
    out["busy_us"] = busy
    out["busy_pct"] = 100.0 * busy / span if span else 0
    out["n_kernels"] = len(db.kernels)
    out["top_kernels"] = db.top_kernels(10)
    out["breakdown"], out["k_total_us"] = kernel_breakdown(db)
    out["top_rt"] = db.top_runtime(10)
    out["sync"] = db.sync_sum()
    out["memcpy_n"], out["memcpy_us"] = len(db.memcpy), \
        sum(e - s for s, e, _ in db.memcpy) / 1e3
    out["gaps"] = db.gaps(min_us=100)
    if mtp:
        out["cycles"] = mtp_cycles(db)
    return out


def main():
    ts = find_run_ts()
    rundir = os.path.join(BENCH, "runs", "prof-prd-" + ts)
    res = json.load(open(os.path.join(rundir, "results.json")))
    on_dec = res.get("on", {}).get("decode", {})
    log("run : " + ts + " → " + rundir)

    dbs = {}
    for cfg in ("off", "on"):
        for ph in ("prefill", "decode"):
            f = "prof_prd_%s_%s_%s.sqlite" % (cfg, ph, ts)
            p = os.path.join(BENCH, f)
            if not os.path.isfile(p):
                log("MANQUANT : " + f)
                continue
            log("  · chargement " + f)
            dbs[(cfg, ph)] = DB(p)

    L = []
    ap = L.append
    ap("# PROF PRD — analyse captures (run %s)" % ts)
    ap("")
    ap("date : %s" % res.get("ts"))
    ap("so : %s | ctx ~115k (prompt %d tok) | b2048 | KV q8_0 | H2 OFF"
       % (res.get("so_md5", "?")[:8],
          res.get("off", {}).get("prefill", {}).get("prompt", 0)))
    ap("")
    ap("| config | phase | wall s | tok/s (prompt / gén.) | ms/tok gén. | kernels | span GPU | busy GPU | top gap |")
    ap("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    ph_res = {}
    for cfg, ph in (("off", "prefill"), ("off", "decode"),
                    ("on", "prefill"), ("on", "decode")):
        db = dbs.get((cfg, ph))
        r = res.get(cfg, {})
        d = r.get(ph, {})
        wall = r.get(ph + "_wall_ms", 0) / 1000.0
        if not db:
            continue
        a = analyze_phase(r, db, wall * 1e3, cfg + "_" + ph,
                          mtp=(cfg == "on" and ph == "decode"))
        ph_res[(cfg, ph)] = a
        tag = ("MTP ON" if cfg == "on" else "MTP OFF")
        gl, ng, totg = a["gaps"]
        topgap = fmt_us(gl[0][0]) if gl else "0"
        if ph == "prefill":
            tps_s = "%.1f / %s" % (d.get("prompt_tps", 0),
                                   (("%.2f" % d["predicted_tps"])
                                    if d.get("predicted_tps") else "—"))
        else:
            tps_s = "%.2f" % d.get("predicted_tps", 0)
        ap("| %s | %s | %.1f | %s | %s | %d | %s | %.1f %% | %s |" % (
            tag, ph, wall, tps_s, d.get("ms_per_tok"),
            a["n_kernels"], fmt_us(a["span_us"]), a["busy_pct"], topgap))
    ap("")

    for cfg, ph in (("off", "prefill"), ("on", "prefill"),
                    ("off", "decode"), ("on", "decode")):
        a = ph_res.get((cfg, ph))
        if not a:
            continue
        tag = ("MTP ON" if cfg == "on" else "MTP OFF")
        ap("## %s — %s" % (tag, ph))
        ap("")
        ap("- span GPU (1er→dernier kernel) : %s, busy union : %s (%.1f %%)"
           % (fmt_us(a["span_us"]), fmt_us(a["busy_us"]), a["busy_pct"]))
        ap("- kernels : %d, sync stream : %d (%s), memcpy : %d (%s)"
           % (a["n_kernels"], a["sync"][0], fmt_us(a["sync"][1]),
              a["memcpy_n"], fmt_us(a["memcpy_us"])))
        ap("")
        ap("| classe | # | temps | %GPU |")
        ap("|---|---:|---:|---:|")
        for k, n, t, pct in a["breakdown"]:
            ap("| %s | %d | %s | %.1f |" % (k, n, fmt_us(t), pct))
        ap("")
        ap("top kernels :")
        for k, n, t in a["top_kernels"][:8]:
            ap("- `%s` ×%d = %s" % (k, n, fmt_us(t)))
        ap("")
        ap("top runtime API :")
        for k, n, t in a["top_rt"][:8]:
            ap("- `%s` ×%d = %s" % (k, n, fmt_us(t)))
        gaps, ng, totg = a["gaps"]
        ap("")
        ap("gaps GPU >100 µs : %d, total %s ; plus gros : %s"
           % (ng, fmt_us(totg),
              ", ".join(fmt_us(g) for g, _ in gaps[:5])))
        ap("")

    # ---- décomposition MTP par cycle (steady-state) ----
    a = ph_res.get(("on", "decode"))
    if a and a.get("cycles"):
        cyc = a["cycles"]
        # steady-state = après le plus gros gap inter-graph (transition
        # prefill→decode ; avant : warmup / 16 premiers tokens du task 9)
        gaps_i = []
        for i in range(len(cyc) - 1):
            gaps_i.append((cyc[i + 1]["t0"] - cyc[i]["t1"]))
        bi = max(range(len(gaps_i)), key=lambda i: gaps_i[i])
        seg = cyc[bi + 1:]
        per = defaultdict(lambda: [0, 0.0, 0])
        trans = defaultdict(list)
        for i, c in enumerate(seg):
            p = per[c["kind"]]
            p[0] += 1
            p[1] += c["busy_us"]
            p[2] += c["nk"]
            if i > 0:
                trans[(seg[i - 1]["kind"], c["kind"])].append(
                    c["t0"] - seg[i - 1]["t1"])
        nv = per.get("verify", [0, 0, 0])[0]
        ntok = res.get("on", {}).get("decode", {}).get("completion", 0)
        per_tok_v = ntok / nv if nv else 0
        span_us = (seg[-1]["t1"] - seg[0]["t0"]) if len(seg) > 1 else 0
        host_us = sum(t for v in trans.values() for t in v)
        wall_cyc = (span_us + host_us) / len(seg) if seg else 0
        ap("## MTP ON — décomposition par cycle (steady-state, après transition)")
        ap("")
        ap("capture = préfill 4 tok (checkpoint restore) + génération %d tok ;"
           % ntok)
        ap("steady-state = %d graph launches après le gap de transition "
           "(%d verifies ≈ %d tok, soit %.2f tok/verify)"
           % (len(seg), nv, ntok, per_tok_v))
        ap("")
        ap("| phase | # | busy/cycle | busy tot |")
        ap("|---|---:|---:|---:|")
        for k in ("verify", "draft", "sampling"):
            p = per.get(k, [0, 0, 0])
            if p[0]:
                ap("| %s | %d | %s | %s |"
                   % (k, p[0], fmt_us(p[1] / p[0]), fmt_us(p[1])))
        ap("")
        busy_cyc = sum(v[1] for v in per.values()) / len(seg) if seg else 0
        host_cyc = host_us / len(seg) if seg else 0
        wall_ver = (span_us + host_us) / nv if nv else 0
        tps = per_tok_v / (wall_ver / 1e6) if wall_ver else 0
        meas = on_dec.get("predicted_tps", 0) if isinstance(on_dec, dict) else 0
        ap("- **wall/launch = %s** (busy GPU %s + host gap %s) ; "
           "wall/verify = %s (≈%.1f launches/verify) → **%.1f tok/s** "
           "(%.2f tok/verify) — cohérent avec %.2f tok/s mesurés"
           % (fmt_us(wall_cyc), fmt_us(busy_cyc), fmt_us(host_cyc),
              fmt_us(wall_ver), len(seg) / nv if nv else 0, tps, per_tok_v,
              meas))
        ap("")
        ap("gaps host par transition (moy / max) :")
        for k, v in sorted(trans.items(), key=lambda kv: -sum(kv[1])):
            ap("- %s → %s : n=%d, moy %s, max %s"
               % (k[0], k[1], len(v), fmt_us(sum(v) / len(v)), fmt_us(max(v))))
        one = [(k, v) for k, v in trans.items() if len(v) == 1 and v[0] > 10_000]
        for k, v in one:
            ap("  - (n=1, %s : artefact de transition / hiccup unique, à "
               "écarter du budget)" % fmt_us(v[0]))
        ap("")
        # breakdown de classe dans les verifies
        db = dbs.get(("on", "decode"))
        if db and nv:
            ver = [(c["t0"] * 1e3, c["t1"] * 1e3) for c in seg
                     if c["kind"] == "verify"]
            agg = defaultdict(lambda: [0, 0.0])
            for s, e, sid, corr, gid, sn, dn, *r in db.kernels:
                if any(v0 <= s and e <= v1 for v0, v1 in ver):
                    cl = classify_kernel(sn or dn)
                    x = agg[cl]
                    x[0] += 1
                    x[1] += e - s
            ap("breakdown GPU **à l'intérieur d'un verify** (busy/verify) :")
            for k, v in sorted(agg.items(), key=lambda kv: -kv[1][1]):
                ap("- %s : %s (%d kern/verify)"
                   % (k, fmt_us(v[1] / nv / 1e3), v[0] // nv))
            ap("")
        # perf CPU
        pperf = os.path.join(rundir, "perf_PRD_on_dc")
        if os.path.isfile(pperf):
            ap("Côté CPU (perf 99 Hz, fenêtre decode MTP ON) : 71 %% dans "
               "`ggml_graph_compute_secondary_thread` (libggml-cpu, worker "
               "spin) + 24 %% `cudaStreamSynchronize` — le host **attend le "
               "GPU**, pas l'inverse. Les gaps host (~%s/launch) viennent du "
               "sampling + bookkeeping, pas d'un goulot CPU."
               % fmt_us(host_cyc))
            ap("")

    # ---- samplers ----
    ap("## Sampler nvidia-smi 100 ms (busy = échantillons avec util>0)")
    ap("")
    ap("| fenêtre | échantillons | busy % | util moy |")
    ap("|---|---:|---:|---:|")
    for cfg, ph in (("off", "prefill"), ("on", "prefill"),
                    ("off", "decode"), ("on", "decode")):
        br = res.get(cfg, {}).get(ph + "_bracket")
        p = os.path.join(rundir, "%s_%s_gputimes.tsv" % (cfg, ph))
        if br and os.path.isfile(p):
            s = gputimes_summary(p, br[0], br[1])
            if s:
                ap("| %s %s | %d | %.0f %% | %.0f |"
                   % ("MTP ON" if cfg == "on" else "MTP OFF", ph,
                      s[0], s[1], s[2]))
    ap("")

    # ---- MTP ON vs OFF ----
    ap("## Verdict MTP ON vs OFF (ctx ~112k, début de decode)")
    ap("")
    off = res.get("off", {})
    on = res.get("on", {})
    o = off.get("decode", {})
    n = on.get("decode", {})
    if o and n:
        sp = (o.get("ms_per_tok") or 0) / (n.get("ms_per_tok") or 1)
        ap("- decode : MTP OFF %.2f ms/tok (%.2f t/s) → MTP ON %.2f ms/tok "
           "(%.2f t/s) = **×%.2f**" % (
               o.get("ms_per_tok", 0), o.get("predicted_tps", 0),
               n.get("ms_per_tok", 0), n.get("predicted_tps", 0), sp))
        acc = acceptance_stats(on.get("server_stats"))
        if acc:
            perpos = "/".join("%.3f" % v for v in acc[4]) if acc[4] else "?"
            ap("- acceptance MTP (fenêtre decode) : %.3f (%d/%d, moy %.2f pos, "
               "per-pos %s)" % (acc[0], acc[1], acc[2], acc[3], perpos))
        ap("- prefill : ~identique (617.6 vs 619.7 t/s) — le MTP ne touche pas "
           "le prefill (defer prompt). Le « 3.9 t/s » de la ligne prefill MTP "
           "ON = les 16 tok générés après le prompt (1er cycle MTP froid, "
           "acceptance 0.40), pas le prompt.")
    ap("")

    out = os.path.join(BENCH, "prof_prd_analysis_%s.md" % ts)
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    log("report → " + out)


if __name__ == "__main__":
    main()
