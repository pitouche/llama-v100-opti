#!/usr/bin/env python3
"""
Analyseur "orchestrateur CUDA" pour traces CUPTI 12.9 du shim libcupath (format v2).

Questions : qui orchestre les launches GPU, à quelle cadence, quel coût CPU/API,
d'où viennent les gaps, et le decode steady est-il en CUDA graphs (cudaGraphLaunch)
ou en milliers de kernels unitaires ?

Format binaire (56 octets/record, little-endian) :
    0 u64 seq | 8 u64 gpu_start | 16 u64 gpu_end | 24 u64 dur_ns
    32 u64 corr | 40 u32 stream | 44 u32 op (0=KER 1=CPY 2=RT 3=RESET 4=MARK)
    48 u64 aux (RT: cbid)
Domaine CUPTI = horloge en ns basée epoch (≈ CLOCK_REALTIME) sur cette machine.
Le bracket (TS_START TS_END) est en US réels -> *1000 pour entrer en ns CUPTI.
Le record RESET (op=3) a start=end=0 : toujours exclu des bornes temporelles.

Usage :
    python3 nso_orchestrator_cupath_analyze.py <prefix> [bracket_file]
"""
import sys, os, re, json
import numpy as np

DT = np.dtype([
    ("seq","<u8"),("gpu_start","<u8"),("gpu_end","<u8"),("dur_ns","<u8"),
    ("corr","<u8"),("stream","<u4"),("op","<u4"),("aux","<u8")])
REC = 56
CBID_LAUNCH = {211, 214, 430, 431}   # cudaLaunchKernel{,_ptsz,ExC,ExC_ptsz}
CBID_GRAPH  = {311}                  # cudaGraphLaunch

def load_binary(path):
    sz = os.path.getsize(path)
    n = sz // REC
    raw = np.fromfile(path, dtype=np.uint8, count=n*REC).reshape(n, REC)
    arr = np.ascontiguousarray(raw).view(DT)
    contig = bool(np.all(np.diff(arr["seq"]) == 1)) if n > 1 else True
    return arr, n, contig

def pct(a, p):
    if a.size == 0:
        return 0.0, 0.0, 0.0
    a = np.sort(a.astype(np.int64))
    f = lambda q: a[min(a.size-1, int(q*(a.size-1)))]
    return f(0.50), f(0.95), f(0.99)

def load_names(path):
    ker_name = {}
    rx = re.compile(rb"^(KER|CPY|RT)\s+seq=(\d+)\b")
    with open(path, "rb") as f:
        for line in f:
            m = rx.match(line)
            if not m:
                continue
            op = m.group(1).decode()
            seq = int(m.group(2))
            i = line.rfind(b"name=")
            name = line[i+5:].rstrip(b"\n").decode("latin1") if i >= 0 else "?"
            if op == "KER":
                ker_name[seq] = name
    return ker_name

def norm(name):
    return re.sub(r"<.*>", "", name).strip()

def main():
    prefix = sys.argv[1]
    bracket_f = sys.argv[2] if len(sys.argv) > 2 else None
    cbi = {int(k): v for k, v in json.load(open("/tmp/cbid_map.json")).items()}

    print("="*78)
    print("TRACE :", os.path.basename(prefix))
    print("="*78)
    arr, n, contig = load_binary(prefix)
    ops = arr["op"]
    data = arr[np.isin(ops, [0,1,2])]   # RESET exclu
    kers = arr[ops==0]; cprs = arr[ops==1]; rts = arr[ops==2]
    print(f"records = {n} (seq continu={contig})  KER={len(kers)} CPY={len(cprs)} RT={len(rts)}")
    gmin = int(data["gpu_start"].min()); gmax = int(data["gpu_end"].max())
    print(f"span CUPTI (KER/CPY/RT) = {(gmax-gmin)/1e9:.3f} s   [{gmin} .. {gmax} ns]")

    # ---------------- fenêtre steady ----------------
    if bracket_f:
        bs, be = [int(x) for x in open(bracket_f).read().split()]
        ws, we = bs*1000, min(be*1000, gmax)   # us->ns ; borné par fin de trace
    else:
        we = gmax; ws = gmax - 2*10**9
    win_us = (we-ws)/1000.0
    print(f"\nFENETRE STEADY : {win_us/1e6:.3f} s   (bracket={bool(bracket_f)})")

    wker = kers[(kers["gpu_start"] < we) & (kers["gpu_end"] > ws)]
    wcpy = cprs[(cprs["gpu_start"] < we) & (cprs["gpu_end"] > ws)]
    wrts = rts[(rts["gpu_start"] < we) & (rts["gpu_end"] > ws)]
    print(f"KER={len(wker)} CPY={len(wcpy)} RT={len(wrts)}")
    if win_us == 0:
        return

    # ---------------- 1. mode de lancement ----------------
    wnames = np.empty(len(wrts), dtype=object)
    for i, c in enumerate(wrts["aux"]):
        wnames[i] = cbi.get(int(c), f"cbid_{c}")
    wl = sum(int(c) in CBID_LAUNCH for c in wrts["aux"])
    wg = sum(int(c) in CBID_GRAPH for c in wrts["aux"])
    print(f"\n>>> LANCEMENTS (fenêtre) : kernel-unitaires = {wl}   cudaGraphLaunch = {wg}")
    print(f"    cadence : {len(wker)/win_us*1e6:.0f} KER/s   {len(wrts)/win_us*1e6:.0f} RT/s   "
          f"{wl/win_us*1e6:.0f} launch/s")

    # ---------------- 2. top API dans la fenêtre ----------------
    print(f"\nAPI RUNTIME (fenêtre) :")
    print(f"{'API':34s}{'count':>8s}{'/s':>8s}{'tot_us':>10s}{'%wall':>7s}{'p50_us':>8s}{'p99_us':>9s}")
    wd = wrts["dur_ns"].astype(np.int64)
    host_busy_us = wd.sum()/1000.0
    uniq, cnt = np.unique(wnames, return_counts=True)
    for i in np.argsort(-cnt)[:15]:
        u = uniq[i]
        sel = wnames == u
        d = wd[sel]
        p50,_,p99 = pct(d,50)
        tt = d.sum()/1000.0
        print(f"{u:34s}{sel.sum():>8d}{sel.sum()/win_us*1e6:>8.0f}{tt:>10.0f}"
              f"{100*tt/win_us if win_us else 0:>6.1f}%{p50:>8.1f}{p99:>9.1f}")
    print(f"{'HOST-API busy (somme des durées)':34s}{'':>8s}{'':>8s}{host_busy_us:>10.0f}"
          f"{100*host_busy_us/win_us if win_us else 0:>6.1f}%  (sur {win_us/1000.0:.0f} ms de wall)")

    # ---------------- 3. GPU busy / gaps ----------------
    iv = []
    if len(wker): iv.append(np.column_stack((wker["gpu_start"], wker["gpu_end"])))
    if len(wcpy): iv.append(np.column_stack((wcpy["gpu_start"], wcpy["gpu_end"])))
    iv = np.concatenate(iv).astype(np.int64)
    # clamp sur la fenêtre (sinon les records qui débordent biaisent busy)
    iv[:,0] = np.maximum(iv[:,0], ws); iv[:,1] = np.minimum(iv[:,1], we)
    iv = iv[iv[:,1] > iv[:,0]]
    iv = np.sort(iv, axis=0)
    busy = 0; cs, ce = iv[0,0], iv[0,1]
    gaps = []   # (gap_ns, gap_start)
    for si, ei in zip(iv[1:,0], iv[1:,1]):
        if si > ce:
            busy += ce - cs
            if si - ce > 0:
                gaps.append((si-ce, ce))
            cs, ce = si, ei
        else:
            if ei > ce: ce = ei
    busy += ce - cs
    wall_ns = we - ws
    cov = 100*busy/wall_ns
    print(f"\nGPU BUSY (union KER+CPY) = {busy/1e6:.1f} ms / {wall_ns/1e6:.1f} ms  => {cov:.1f}%  ; idle = {100-cov:.1f}%")

    gaps = np.array(gaps, dtype=np.int64)
    if gaps.size:
        g = gaps[:,0]
        p50,p95,p99 = pct(g,50)
        print(f"gaps (tous) : n={gaps.size} total={g.sum()/1e6:.1f} ms  p50={p50/1000:.0f} µs p95={p95/1000:.0f} µs max={g.max()/1000:.0f} µs")
        big = g[g > 100_000]
        if big.size:
            print(f"gaps >100µs : n={big.size} total={big.sum()/1e6:.1f} ms  p50={pct(big,50)[0]/1000:.0f} µs max={big.max()/1000:.0f} µs")
        # histogramme
        bins = [(0,10_000),(10_000,50_000),(50_000,100_000),(100_000,500_000),(500_000,1_000_000),(1_000_000,10**18)]
        print("histogramme gaps :")
        for lo,hi in bins:
            m = (g>=lo)&(g<hi)
            lab = f">={lo/1000:.0f} µs < {hi/1000:.0f} µs" if hi < 10**18 else f">= {lo/1000:.0f} µs"
            print(f"  {lab:22s} n={m.sum():6d}  total={g[m].sum()/1e6:9.1f} ms")

    # ---------------- 4. qui fait quoi pendant les gros gaps ----------------
    ker_name = load_names(prefix + ".names")
    wker_kname = np.array([norm(ker_name.get(int(s), "?")) for s in wker["seq"]])
    print(f"\nTOP 12 GAPS — attribution fractionnée (RT ∩ gap) + dernier kernel avant gap :")
    if gaps.size:
        order = np.argsort(-gaps[:,0])[:12]
        for oi in order:
            gs, ge = int(gaps[oi,1]), int(gaps[oi,1]) + int(gaps[oi,0])
            ov = (np.minimum(wrts["gpu_end"].astype(np.int64), ge) -
                  np.maximum(wrts["gpu_start"].astype(np.int64), gs)).clip(0)
            sel = ov > 0
            if sel.any():
                cnts = {}
                for c, o in zip(wrts["aux"][sel], ov[sel]):
                    nm = cbi.get(int(c), f"cbid_{c}")
                    cnts[nm] = cnts.get(nm, 0) + int(o)
                top = sorted(cnts.items(), key=lambda kv:-kv[1])[:4]
                desc = ", ".join(f"{k}={v/1000:.0f}µs" for k,v in top)
            else:
                desc = "AUCUN RT (host non-CUDA : python/sampling/bookkeeping)"
            before = np.where(wker_kname != "")[0]
            # dernier kernel ayant fini avant le gap
            idx = np.where(wker["gpu_end"] <= gs)[0]
            kn2 = (wker_kname[idx[-1]][:30] if idx.size else "-")
            print(f"  gap {gaps[oi,0]/1000:7.0f} µs @ {gs/1e9-gmin/1e9:8.3f}s | avant: {kn2:30s}| {desc}")

    # ---------------- 5. segmentation ----------------
    for TH in (100_000, 500_000, 1_000_000):
        sep = gaps[gaps[:,0] > TH]
        if sep.size >= 2:
            ncyc = sep.size + 1
            cuts = sorted(int(s) for s in sep[:,1])
            bounds = [ws] + cuts + [we]
            durs = []; nks = []; nrs = []
            for a,b in zip(bounds[:-1], bounds[1:]):
                durs.append(b-a)
                nks.append(int(((wker["gpu_start"]>=a)&(wker["gpu_end"]<=b)).sum()))
                nrs.append(int(((wrts["gpu_start"]>=a)&(wrts["gpu_end"]<=b)).sum()))
            durs = np.array(durs); nks = np.array(nks); nrs = np.array(nrs)
            print(f"\nSÉPARATION gap>{TH//1000} µs : {ncyc} segments / {wall_ns/1e9:.3f} s "
                  f"=> {ncyc/(wall_ns/1e9):.1f} seg/s ; wall/seg p50={pct(durs,50)[0]/1e6:.2f} ms "
                  f"KER/seg p50={pct(nks,50)[0]} RT/seg p50={pct(nrs,50)[0]}")

    # ---------------- 6. mix kernels ----------------
    print("\nMIX KERNELS (fenêtre) — par durée GPU :")
    mixc = {}; mixd = {}; durn = {}
    for j in range(len(wker)):
        nm = wker_kname[j]
        mixc[nm] = mixc.get(nm, 0) + 1
        mixd[nm] = mixd.get(nm, 0) + int(wker["dur_ns"][j])
        durn.setdefault(nm, []).append(int(wker["dur_ns"][j]))
    totk = sum(mixc.values()); totd = sum(mixd.values())
    print(f"{'kernel':38s}{'count':>8s}{'%cnt':>7s}{'tot_us':>10s}{'%time':>7s}{'p50_us':>8s}")
    for nm in sorted(mixd, key=lambda k:-mixd[k])[:14]:
        p50 = pct(np.array(durn[nm]),50)[0]/1000 if nm in durn else 0
        print(f"{nm[:38]:38s}{mixc[nm]:>8d}{100*mixc[nm]/totk:>6.1f}%{mixd[nm]/1000:>10.0f}{100*mixd[nm]/totd:>6.1f}%{p50:>8.1f}")

if __name__ == "__main__":
    main()
