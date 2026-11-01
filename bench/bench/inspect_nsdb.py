#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inspect_nsdb.py — inspecte le contenu d'un .nsys-rep (export sqlite en mémoire).
Usage : python3 inspect_nsdb.py <fichier.nsys-rep>
"""
import sqlite3
import subprocess
import sys
import os

NSYS = "/opt/nvidia/nsight-systems/2025.5.2/target-linux-x64/nsys"


def main():
    rep = sys.argv[1]
    tmp = "/tmp/_nsdb_%d" % os.getpid()
    r = subprocess.run([NSYS, "export", "--type", "sqlite", "--force-overwrite",
                        "true", "-o", tmp, rep], capture_output=True, text=True)
    if r.returncode != 0:
        print("EXPORT KO :")
        print((r.stderr or r.stdout)[-2000:])
        sys.exit(1)
    # l'export produit <tmp> (sans extension) ou <tmp>.sqlite
    dbfile = tmp if os.path.exists(tmp) else tmp + ".sqlite"
    c = sqlite3.connect(dbfile)
    tabs = [r[0] for r in c.execute(
        "select name from sqlite_master where type='table' order by name")]
    print("TABLES (%d) :" % len(tabs))
    for t in tabs:
        try:
            n = c.execute('select count(*) from "%s"' % t).fetchone()[0]
        except Exception:
            n = "?"
        print("  %-44s %6s" % (t, n))
    print()
    print("=== DIAGNOSTIC_EVENT (CUPTI/CUDA related) ===")
    try:
        for r in c.execute("select * from DIAGNOSTIC_EVENT"):
            s = str(r)
            if "CUPTI" in s or "CUDA" in s or "cuda" in s:
                print("  ", s[:240])
    except Exception as e:
        print("  (aucun DIAGNOSTIC_EVENT : %s)" % e)
    print()
    # noyau CUDA : cherche les tables CUPTI_ACTIVITY_KERNEL / CUDA API
    for pat in ("KERNEL", "RUNTIME", "DRIVER", "MEMCPY", "GRAPH"):
        cand = [t for t in tabs if pat in t.upper() and "ENUM" not in t]
        if cand:
            print("=== tables '%s' ===" % pat)
            for t in cand:
                try:
                    n = c.execute('select count(*) from "%s"' % t).fetchone()[0]
                    print("  %s : %s lignes" % (t, n))
                    cols = [d[1] for d in c.execute('PRAGMA table_info("%s")' % t)]
                    if n and n < 50:
                        for row in c.execute('select * from "%s" limit 8' % t):
                            print("     ", str(row)[:220])
                except Exception as e:
                    print("  %s err %s" % (t, e))
    for f in (tmp, tmp + ".sqlite"):
        try:
            if os.path.exists(f):
                os.remove(f)
        except OSError:
            pass
    sys.exit(0)


if __name__ == "__main__":
    main()
