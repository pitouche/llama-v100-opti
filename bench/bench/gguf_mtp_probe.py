#!/usr/bin/env python3
"""Probe GGUF header/metadata only (no tensor data) to count MTP modules.

Usage: python3 gguf_mtp_probe.py <path-to-gguf>
Reads: GGUF header, KV metadata, tensor info table. Prints keys matching
'mtp' (case-insensitive) and tensor names matching 'mtp' with max index.
Bounded: only the metadata section is read (few MB typically).
"""
import struct, sys

GGUF_TYPES = {
    0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2),
    4: ("I", 4), 5: ("i", 4), 6: ("f", 4), 7: ("?", 1),
    8: ("str", None), 9: ("arr", None), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8),
}


def rd_string(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")


def rd_value(f, t):
    name, size = GGUF_TYPES[t]
    if t == 8:
        return rd_string(f)
    if t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        cnt = struct.unpack("<Q", f.read(8))[0]
        e2, e3 = GGUF_TYPES[et]
        if e3 is None:
            vals = [rd_string(f) for _ in range(cnt)]
            return ("array-str", et, cnt, vals[:5])
        data = f.read(cnt * e3)
        vals = struct.unpack(f"<{cnt}{e2[0]}", data) if cnt <= 50 else None
        return ("array", et, cnt, vals)
    if size is not None:
        return struct.unpack(f"<{name[0]}", f.read(size))[0]
    raise ValueError(f"unexpected type {t}")


def main():
    path = sys.argv[1]
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            print("NOT A GGUF", magic)
            return
        ver, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        print(f"version={ver} n_tensors={n_tensors} n_kv={n_kv}")
        kv = {}
        for _ in range(n_kv):
            k = rd_string(f)
            t = struct.unpack("<I", f.read(4))[0]
            v = rd_value(f, t)
            kv[k] = v
        mtp_kv = {k: v for k, v in kv.items() if "mtp" in k.lower()}
        print(f"\n== {len(mtp_kv)} MTP-related KV keys ==")
        for k in sorted(mtp_kv):
            v = mtp_kv[k]
            if isinstance(v, tuple) and v[0] == "array":
                print(f"  {k}: array type={v[1]} n={v[2]} sample={v[3]}")
            else:
                print(f"  {k}: {v!r}")
        # also architecture + expert / block count hints
        for k in sorted(kv):
            kl = k.lower()
            if any(s in kl for s in ("arch", "block_count", "n_layer", "head_count", "experts")):
                print(f"  [ctx] {k}: {kv[k]!r}")
        # tensor table
        print("\n== tensors ==")
        n_mtp = 0
        max_idx = -1
        idx_re_cache = []
        for _ in range(n_tensors):
            name = rd_string(f)
            ndim = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack(f"<{ndim}Q", f.read(8 * ndim))
            ttype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            if "mtp" in name.lower():
                n_mtp += 1
                if n_mtp <= 40:
                    print(f"  {name} dims={dims} type={ttype}")
                # extract trailing .mtp.<i> or .<i>.mtp or module index after 'mtp'
                base = name.lower()
                pos = base.rfind("mtp")
                tail = base[pos + 3:]
                segs = [s for s in tail.split(".") if s.isdigit()]
                if segs:
                    max_idx = max(max_idx, int(segs[0]))
        print(f"\ntotal mtp tensors: {n_mtp}")
        print(f"max mtp module index found in tensor names: {max_idx}")
        # guess module count: if indices 0..max contiguous -> modules = max_idx+1
        if max_idx >= 0:
            print(f"=> likely {max_idx + 1} MTP module(s) (indices 0..{max_idx})")


if __name__ == "__main__":
    main()
