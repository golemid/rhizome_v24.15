#!/usr/bin/env python3
r"""
resolve_map_offsets.py -- RHIZOME v24.02 (DRAFT) DIAG1 companion tool.

v2 (field fix, 2026-10-23): DIAG1's first cut failed on real MSVC maps
("map has no 'Publics by Value' entries") because it expected the marker
at the start of a line. Real MSVC maps write the publics listing under a
column header line (" Address   Publics by Value   Lib:Object") and append
a "Static symbols" listing. v2 parses both; anything else ends the scan.

Translates Windows crash "Fault offset" values (RVAs, e.g. from Event Log
Application Error records for rhizome_app.exe) into function symbols, using
the linker map emitted by the DIAG1 build (CMakeLists.txt /MAP option).

METHOD
  - Reads PE section table from build\Release\rhizome_app.exe to get each
    section's base RVA.
  - Reads build\rhizome_app.map "Publics by Value" entries
    (format: " 0001:000cab81  <symbol>  <tail>"), where 0001 is the 1-based
    PE section number and 000cab81 the offset within that section.
  - Symbol RVA = section_rva[num-1] + section_offset.
  - For each queried offset: reports the nearest symbol at or below it
    (i.e. the function whose body contains the fault), the byte distance
    into that function, and the contributing object file (last token of the
    map line's tail, when present).

USAGE (from native\\, standard build tree; run AFTER a DIAG1 rebuild):
  python tools\\resolve_map_offsets.py 0xcab81 0xbe6f2 [more hex offsets...]
  (no arguments -> resolves the two offsets recorded by Windows on
   2026-10-22 against the fix2-state binary)

Exit codes: 0 = resolved (at least one offset mapped to a symbol);
1 = nothing resolved (missing map/exe printed to stdout); 2 = usage error.
"""

import os
import re
import struct
import sys

DEFAULT_OFFSETS = ["0xcab81", "0xbe6f2"]  # 2026-10-22 Event Log records (fix2 state)

MAP_SEARCH = [
    "build/rhizome_app.map",
    "build/Release/rhizome_app.map",
    "rhizome_app.map",
]
EXE_SEARCH = [
    "build/Release/rhizome_app.exe",
    "build/rhizome_app.exe",
    "rhizome_app.exe",
]

MAP_LINE_RE = re.compile(
    r"^\s*([0-9A-Fa-f]{4}):([0-9A-Fa-f]{8})\s+(\S+)(?:\s+(.*))?$"
)


def find_first(paths, what):
    for p in paths:
        if os.path.isfile(p):
            return p
    print(f"  ERROR: {what} not found. Searched:")
    for p in paths:
        print(f"    {p}")
    return None


def parse_pe_sections(exe_path):
    """Return [(name, rva, vsize)] in PE table order."""
    with open(exe_path, "rb") as f:
        head = f.read(0x1000)
    if len(head) < 0x40 or head[:2] != b"MZ":
        raise ValueError("not an MZ executable")
    e_lfanew = struct.unpack_from("<I", head, 0x3C)[0]
    if head[e_lfanew:e_lfanew + 4] != b"PE\0\0":
        raise ValueError("bad PE signature")
    num_sections = struct.unpack_from("<H", head, e_lfanew + 6)[0]
    opt_size = struct.unpack_from("<H", head, e_lfanew + 20)[0]
    sec_off = e_lfanew + 24 + opt_size
    out = []
    for i in range(num_sections):
        base = sec_off + i * 40
        if base + 40 > len(head):
            with open(exe_path, "rb") as f:
                f.seek(base)
                raw = f.read(40)
        else:
            raw = head[base:base + 40]
        name = raw[:8].rstrip(b"\0").decode("ascii", "replace")
        vsize, rva = struct.unpack_from("<II", raw, 8)
        out.append((name, rva, vsize))
    return out


def parse_map_publics(map_path):
    """Return [(section_num, sec_off, symbol, tail)] from the map's symbol
    listings ("Publics by Value" and the "Static symbols" listing that
    follows it in MSVC maps). RVAs are resolved later against the exe.

    Real MSVC maps put the publics marker inside a column header line
    (" Address   Publics by Value   Lib:Object"), not at the start of a
    line, so the marker is matched anywhere in a line. Any other
    non-entry text (blank lines pass; an "Exports" section, section
    epilogue, or similar) ends the scan."""
    entries = []
    in_symbols = False
    with open(map_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not in_symbols:
                if "Publics by Value" in line:
                    in_symbols = True
                continue
            m = MAP_LINE_RE.match(line)
            if m:
                sec = int(m.group(1), 16)
                off = int(m.group(2), 16)
                entries.append((sec, off, m.group(3), m.group(4) or ""))
            elif "Static symbols" in line:
                continue  # second listing header; keep scanning symbols
            elif line.strip():
                break  # a different map section begins; done
    return entries


def containing_section(sections, rva):
    for name, base, vsize in sections:
        if base <= rva < base + max(vsize, 1):
            return name
    return "?"


def main(argv):
    raw_offsets = argv[1:] or DEFAULT_OFFSETS
    offsets = []
    for r in raw_offsets:
        try:
            offsets.append(int(r, 16) if r.lower().startswith("0x") else int(r, 16))
        except ValueError:
            print(f"ERROR: '{r}' is not a hex offset")
            return 2

    print("RHIZOME v24.02 DIAG1 -- crash offset resolver")
    print("=" * 60)
    map_path = find_first(MAP_SEARCH, "linker map (build the DIAG1 change first)")
    exe_path = find_first(EXE_SEARCH, "rhizome_app.exe")
    if not map_path or not exe_path:
        return 1
    print(f"Map: {map_path}")
    print(f"Exe: {exe_path}")
    print()

    sections = parse_pe_sections(exe_path)
    entries = parse_map_publics(map_path)
    if not entries:
        print("ERROR: map has no 'Publics by Value' entries")
        return 1

    symbols = []  # (rva, name, obj)
    skipped = 0
    for sec, off, name, tail in entries:
        if not (1 <= sec <= len(sections)):
            skipped += 1
            continue
        rva = sections[sec - 1][1] + off
        obj = tail.split()[-1] if tail.split() else ""
        symbols.append((rva, name, obj))
    symbols.sort(key=lambda s: s[0])
    print(f"Loaded {len(symbols)} symbols from {os.path.basename(map_path)}"
          + (f" ({skipped} non-section entries skipped)" if skipped else ""))
    print()

    resolved_any = False
    for want in offsets:
        print(f"Offset 0x{want:X}  (section: {containing_section(sections, want)})")
        best = None
        for s in symbols:
            if s[0] <= want:
                best = s
            else:
                break
        if best is None:
            print("  -> NO symbol at or below this offset (before first symbol)")
        else:
            resolved_any = True
            dist = want - best[0]
            print(f"  -> {best[1]}")
            print(f"     RVA 0x{best[0]:X}, +0x{dist:X} bytes into the function"
                  + (f"  [{best[2]}]" if best[2] else ""))
        print()

    print("=" * 60)
    print("Paste the block above (Map/Exe paths + per-offset results) to your assistant.")
    return 0 if resolved_any else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
