#!/usr/bin/env python3
"""Segmentation gputimes → identifie fenêtres prefill (gpu≥50) et decode (gpu<50)
   par leg, en exploitant les transitions VRAM 0↔20k+."""
import statistics as st

rows=[]
with open("mtp_prof_stepA_gputimes.tsv") as f:
    next(f)
    for line in f:
        p=line.split()
        if len(p)==5: rows.append((int(p[0]),int(p[1]),int(p[2]),int(p[3]),float(p[4])))
T0=rows[0][0]; dt=st.median([rows[i+1][0]-rows[i][0] for i in range(len(rows)-1)])

# segments actifs (mem>20000), avec étiquette de phase (busy gpu>=50 vs decode gpu<50)
segs=[]; cur=None
for i in range(len(rows)):
    t,g,m,mem,pw = rows[i]
    active = mem>20000
    busy = g>=50
    key=(active,busy)
    if cur is None or cur["key"]!=key:
        if cur is not None: segs.append(cur)
        cur={"key":key,"t0":t,"t1":t,"g":g,"mem":mem}
    else:
        cur["t1"]=t
        cur["mem"]=max(cur["mem"],mem)
if cur: segs.append(cur)

print(f"dt_med={dt:.1f}ms  n={len(rows)}  total={(rows[-1][0]-T0)/1000:.1f}s")
print("\n== Segments (mem>20k), s après T0 ==")
print(f"{'type':8s} {'t0':>7s} {'t1':>7s} {'dur':>6s} {'mem':>6s}  note")
for s in segs:
    if not s["key"][0]:
        continue  # pas de VRAM active
    t0=(s["t0"]-T0)/1000; t1=(s["t1"]-T0)/1000
    typ = "PREFILL" if s["key"][1] else "decode "
    note = ""
    print(f"{typ:8s} {t0:7.1f} {t1:7.1f} {t1-t0:6.1f} {s['mem']:6d}")
