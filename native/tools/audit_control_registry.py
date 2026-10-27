#!/usr/bin/env python3
"""Rhizome v24.01 build-time audit: CSV <-> control registry consistency.

AGENTS.md contract Sec 1.3 requires every control CTRL001..CTRL251 from
param_controls_complete_list.csv to exist, function identically, and map to
the AppState single source of truth (domain rows) or the live ImGuiStyle
(style-editor rows). This script enforces that contract at
BUILD time (wired into CMake as the `rhizome_control_audit` target, a build
dependency of `rhizome_app`) and can also be run standalone:

    python native/tools/audit_control_registry.py \
        --csv  param_controls_complete_list.csv \
        --registry native/src/ui/widgets/control_registry.cpp \
        --logger  native/src/core/unified_logger.cpp \
        --stamp  <optional stamp path>

Audited surfaces (catalog identity only -- `draw_label` / `format` are
v23.28-presentation fields and are deliberately NOT compared):
  A. CSV parse (mixed 10/11-field rows; toggle rows carry the label QUOTED in
     the Control_Type position with TRUE/FALSE in Min/Max) -> 251 records.
  B. k_all_controls[251] in control_registry.cpp:
       id sequence, Panel_Section, catalog Label, Category_ID,
       Logging_Triggered==YES, expected ControlKind from CSV function/type,
       unit (with the v23.28 logger normalization "Click only" -> ""),
       slider float/int ranges exact, resolver discipline (exactly one
       RHZ_BIND_* on value rows, zero on action rows, CTRL047-049 exempt as
       documented per-selected-seed toggles).
  C. k_all_sections[26] section views tile the registry contiguously, in CSV
     section order, with CSV section sizes. (v24.01 Phase 0: 22 sections
     after the Specimen Appearance re-home; v24.01 Phase 1: the lighting
     section grew 7 -> 17 rows; v24.01 Phase 2: three style-editor sections
     appended — Sizes 23 / Colors 63 == ImGuiCol_COUNT at the 8d census
     (ImGui 1.92.8+, re-pinned from 58 by scripts/v2401_phase2_repin_colors.py)
     / UI Rendering 7.)
  D. g_all_controls[251] JSON-Lines monitor catalog in unified_logger.cpp:
     id/section/label/category/unit byte-equal to the registry rows.
  E. Source_File (v24.01 Phase 0.5 + Phase 2): every CSV row names the real
     panel home (panel_growth 001-052 + 242-251 / panel_styles 053-106 + 149-241 /
     panel_scene 107-108 + 112-148 / panel_verification 109-111) and the
     logger catalog's file field equals the CSV Source_File.

Exit code 0 + stamp file on success; non-zero with a per-row diff report on
any mismatch (fails the CMake build).

Pure stdlib. Python 3.8+ (Windows vcpkg dev boxes and CI containers alike).
"""
import argparse
import csv
import re
import sys

EXPECTED_TOTAL = 251
EXPECTED_SECTIONS = 26
CLICK_ONLY = "Click only"  # CSV "unit" for action rows -> logger normalization ""
INDEXED_TOGGLES = ("CTRL047", "CTRL048", "CTRL049")  # per-selected-seed, panel-local
VALUE_KINDS = {"SliderFloat", "SliderInt", "Toggle", "ColorEdit4", "InputText"}

# v24.01 Phase 1 + Phase 2: verified Source_File homes (id -> expected panel file)
def expected_source(ctrl_id):
    n = int(ctrl_id[4:])
    if 1 <= n <= 52:
        return "panel_growth.cpp"
    if 53 <= n <= 106:
        return "panel_styles.cpp"
    if 107 <= n <= 108:
        return "panel_scene.cpp"
    if 109 <= n <= 111:
        return "panel_verification.cpp"
    if 149 <= n <= 241:
        return "panel_styles.cpp"  # v24.01 Phase 2: style-editor sections
    if 242 <= n <= 251:
        return "panel_growth.cpp"  # v24.12 Track F R1: developmental section
    return "panel_scene.cpp"

# CSV (Control_Function, Control_Type) -> registry ControlKind
KIND_FROM_CSV = {
    ("render_compact_slider_float", None): "SliderFloat",
    ("render_compact_slider_int", None): "SliderInt",
    ("render_toggle_switch", None): "Toggle",
    ("render_pill_button", "Button"): "PillButton",
    ("ImGui::Button", "Button"): "Button",
    ("begin_pill_combo", "Combo"): "Combo",
    ("render_pill_combo", "Combo"): "Combo",
    ("ImGui::InputText", "Text Input"): "InputText",
    ("ImGui::Selectable", "Selectable"): "Selectable",
    ("ImGui::ColorEdit4", "Color Picker"): "ColorEdit4",
}

STR = r'"((?:[^"\\]|\\.)*)"'  # C++ string literal body


# ---------------------------------------------------------------- CSV parsing
def parse_csv(path):
    """Return list of dicts using the v23.28 mixed-width row conventions."""
    records = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        expect = ["Control_ID", "Source_File", "Panel_Section", "Control_Function",
                  "Control_Type", "Label", "Min_Value", "Max_Value", "Unit",
                  "Category_ID", "Logging_Triggered"]
        if [h.strip() for h in header] != expect:
            raise SystemExit(f"audit: unexpected CSV header: {header}")
        for raw in reader:
            if not raw or not raw[0].strip():
                continue
            n = len(raw)
            rec = {"n": n, "raw": raw, "id": raw[0].strip(), "source": raw[1].strip()}
            if n == 10:
                # id, file, section, function, LABEL, min, max, unit, cat, log
                (cid, src, sect, func, label, mn, mx, unit, cat, log) = raw
                rec.update(section=sect.strip(), function=func.strip(),
                           ctype=None, label=label.strip(),
                           f_min=mn.strip(), f_max=mx.strip(),
                           unit=unit.strip(), category=cat.strip(),
                           logs=log.strip())
            elif n == 11:
                (cid, src, sect, func, ctype, label, mn, mx, unit, cat, log) = raw
                if ctype.strip() in ("Button", "Combo", "Text Input",
                                     "Selectable", "Color Picker"):
                    rec.update(section=sect.strip(), function=func.strip(),
                               ctype=ctype.strip(), label=label.strip(),
                               f_min=mn.strip(), f_max=mx.strip(),
                               unit=unit.strip(), category=cat.strip(),
                               logs=log.strip())
                else:
                    # Toggle row: label occupies the Control_Type position,
                    # Label column is empty, Min/Max hold TRUE/FALSE.
                    rec.update(section=sect.strip(), function=func.strip(),
                               ctype=None, label=ctype.strip(),
                               f_min="", f_max="",
                               unit=unit.strip(), category=cat.strip(),
                               logs=log.strip())
            else:
                raise SystemExit(
                    f"audit: {rec['id']}: unexpected CSV field count {n}: {raw}")
            records.append(rec)
    return records


# -------------------------------------------------------- registry extraction
ROW_RE = re.compile(
    r'\{"(CTRL\d+)",\s*' + STR + r",\s*" + STR + r",\s*" + STR + r",\s*\n\s*"
    r"ControlKind::(\w+),\s*(true|false),\s*\n\s*"
    + STR + r",\s*" + STR + r",\s*\n\s*"
    r"([-+0-9.eE]+)f,\s*([-+0-9.eE]+)f,\s*(-?\d+),\s*(-?\d+),\s*\n\s*"
    + STR + r",\s*\n\s*"
    r"(.+?)\}?$",  # resolver tail: to end of the row segment (DOTALL below)
    re.DOTALL)


def parse_registry(path):
    src = open(path, encoding="utf-8").read()
    anchor = "k_all_controls[251]"
    if anchor not in src:
        raise SystemExit("audit: k_all_controls[251] not found in registry")
    body = src[src.index(anchor):src.index("};", src.index(anchor))]
    # Split into per-row segments at each row opener. This is robust to the
    # final row lacking a trailing comma before the array terminator.
    starts = [m.start() for m in re.finditer(r'\{"CTRL\d+"', body)]
    if not starts:
        raise SystemExit("audit: no control rows found in k_all_controls")
    segments = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(body)
        segments.append(body[s:e].rstrip().rstrip(","))
    rows = []
    for seg in segments:
        m = ROW_RE.match(seg)
        if not m:
            raise SystemExit("audit: unparsable registry row: " + seg[:120])
        (cid, sect, label, cat, kind, logs, draw, fmt,
         fmin, fmax, imin, imax, unit, tail) = m.groups()
        tail = tail.rstrip().rstrip("}").rstrip().rstrip(",")
        resolvers = len(re.findall(r"RHZ_BIND_\w+", tail))
        stray_ptrs = len(re.findall(r"\bnullptr\b", tail))
        rows.append({"id": cid, "section": sect, "label": label,
                     "category": cat, "kind": kind, "logs": logs == "true",
                     "draw_label": draw, "format": fmt,
                     "f_min": float(fmin), "f_max": float(fmax),
                     "i_min": int(imin), "i_max": int(imax),
                     "unit": unit, "resolvers": resolvers,
                     "nullptrs": stray_ptrs})
    return rows, body


def parse_section_views(path):
    src = open(path, encoding="utf-8").read()
    anchor = "k_all_sections[26]"
    if anchor not in src:
        raise SystemExit("audit: k_all_sections[26] not found in registry")
    body = src[src.index(anchor):src.index("};", src.index(anchor))]
    views = re.findall(r'\{' + STR + r",\s*k_all_controls\s*\+\s*(\d+),\s*(\d+)\}",
                       body)
    return [(name, int(off), int(cnt)) for name, off, cnt in views]


def parse_logger_catalog(path):
    src = open(path, encoding="utf-8").read()
    m = re.search(r"g_all_controls\[251\]\s*=\s*\{", src)
    if not m:
        raise SystemExit("audit: g_all_controls[251] not found in logger")
    body = src[m.end():src.index("};", m.end())]
    row_re = re.compile(r'\{"(CTRL\d+)",\s*' + STR + r",\s*" + STR + r",\s*"
                        + STR + r",\s*" + STR + r",\s*" + STR + r"\}")
    out = {}
    for r in row_re.finditer(body):
        cid, srcf, sect, label, cat, unit = r.groups()
        out[cid] = {"source": srcf, "section": sect, "label": label,
                    "category": cat, "unit": unit}
    return out


# ------------------------------------------------------------------- checking
def norm_unit(csv_unit):
    return "" if csv_unit == CLICK_ONLY else csv_unit


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--logger", default=None)
    ap.add_argument("--stamp", default=None)
    args = ap.parse_args()

    errors = []

    csv_rows = parse_csv(args.csv)
    reg_rows, _ = parse_registry(args.registry)
    views = parse_section_views(args.registry)
    logger_rows = parse_logger_catalog(args.logger) if args.logger else {}

    # ---- A. CSV shape ------------------------------------------------------
    if len(csv_rows) != EXPECTED_TOTAL:
        errors.append(f"CSV: expected {EXPECTED_TOTAL} rows, parsed {len(csv_rows)}")
    for i, r in enumerate(csv_rows):
        if r["id"] != f"CTRL{i+1:03d}":
            errors.append(f"CSV: row {i+1} id '{r['id']}' breaks the contiguous "
                          f"CTRL001..CTRL{EXPECTED_TOTAL:03d} sequence")

    # ---- B. registry vs CSV, row by row -------------------------------------
    if len(reg_rows) != EXPECTED_TOTAL:
        errors.append(f"registry: expected {EXPECTED_TOTAL} rows, parsed {len(reg_rows)}")
    by_id = {r["id"]: r for r in reg_rows}
    if len(by_id) != len(reg_rows):
        errors.append("registry: duplicate CTRL ids present")

    sections_csv = []
    for r in csv_rows:
        if r["section"] not in sections_csv:
            sections_csv.append(r["section"])

    for r in csv_rows:
        cid = r["id"]
        d = by_id.get(cid)
        if d is None:
            errors.append(f"{cid}: missing from k_all_controls")
            continue
        # catalog identity
        for field, csvv, regv in (("section", r["section"], d["section"]),
                                  ("label", r["label"], d["label"]),
                                  ("category", r["category"], d["category"])):
            if csvv != regv:
                errors.append(f"{cid}: {field} mismatch: CSV '{csvv}' vs registry '{regv}'")
        # logging flag
        if (r["logs"].upper() == "YES") != d["logs"]:
            errors.append(f"{cid}: Logging_Triggered CSV '{r['logs']}' "
                          f"vs registry logs={d['logs']}")
        # unit with the v23.28 logger normalization
        if norm_unit(r["unit"]) != d["unit"]:
            errors.append(f"{cid}: unit mismatch: CSV '{r['unit']}' "
                          f"(normalized '{norm_unit(r['unit'])}') vs registry '{d['unit']}'")
        # kind
        expected_kind = KIND_FROM_CSV.get((r["function"], r["ctype"]))
        if expected_kind is None:
            errors.append(f"{cid}: unrecognized CSV function/type pair "
                          f"({r['function']!r}, {r['ctype']!r})")
        elif expected_kind != d["kind"]:
            errors.append(f"{cid}: kind mismatch: CSV implies {expected_kind}, "
                          f"registry has {d['kind']}")
        # ranges: exact for sliders, per v23.29 Phase 4 transcription contract
        if d["kind"] == "SliderFloat":
            for name, csvv, regv in (("Min_Value", r["f_min"], d["f_min"]),
                                     ("Max_Value", r["f_max"], d["f_max"])):
                try:
                    if abs(float(csvv) - regv) > 1e-6:
                        errors.append(f"{cid}: float {name} mismatch: "
                                      f"CSV {csvv} vs registry {regv}")
                except ValueError:
                    errors.append(f"{cid}: CSV float {name} '{csvv}' unparseable")
        elif d["kind"] == "SliderInt":
            for name, csvv, regv in (("Min_Value", r["f_min"], d["i_min"]),
                                     ("Max_Value", r["f_max"], d["i_max"])):
                try:
                    if int(float(csvv)) != regv:
                        errors.append(f"{cid}: int {name} mismatch: "
                                      f"CSV {csvv} vs registry {regv}")
                except ValueError:
                    errors.append(f"{cid}: CSV int {name} '{csvv}' unparseable")
        # resolver discipline
        if d["kind"] in VALUE_KINDS:
            want = 0 if cid in INDEXED_TOGGLES else 1
            if d["resolvers"] != want:
                errors.append(f"{cid}: value control must carry {want} resolver(s), "
                              f"found {d['resolvers']}")
        else:
            if d["resolvers"] != 0:
                errors.append(f"{cid}: action control must carry 0 resolvers, "
                              f"found {d['resolvers']}")
        if d["nullptrs"] + d["resolvers"] != 5:
            errors.append(f"{cid}: resolver tail must have 5 slots, found "
                          f"{d['resolvers']} binding(s) + {d['nullptrs']} nullptr")

    # ---- C. section views ---------------------------------------------------
    if len(views) != EXPECTED_SECTIONS:
        errors.append(f"registry: k_all_sections must list {EXPECTED_SECTIONS} views, "
                      f"found {len(views)}")
    csv_counts = {s: sum(1 for r in csv_rows if r["section"] == s)
                  for s in sections_csv}
    off = 0
    for (name, voff, vcnt), csvname in zip(views, sections_csv):
        if name != csvname:
            errors.append(f"section view '{name}' out of CSV order (expected '{csvname}')")
        if voff != off:
            errors.append(f"section view '{name}': offset {voff} != expected {off}")
        if vcnt != csv_counts.get(name):
            errors.append(f"section view '{name}': count {vcnt} != CSV "
                          f"{csv_counts.get(name)}")
        off += vcnt
    if off != EXPECTED_TOTAL:
        errors.append(f"section views cover {off} rows, expected {EXPECTED_TOTAL}")

    # ---- D. logger monitor catalog -------------------------------------------
    if args.logger:
        if len(logger_rows) != EXPECTED_TOTAL:
            errors.append(f"logger: g_all_controls must list {EXPECTED_TOTAL} rows, "
                          f"found {len(logger_rows)}")
        for r in csv_rows:
            cid = r["id"]
            lg = logger_rows.get(cid)
            if lg is None:
                errors.append(f"{cid}: missing from g_all_controls monitor catalog")
                continue
            d = by_id.get(cid)
            for field, regv, lgv in (("section", d["section"] if d else None, lg["section"]),
                                     ("label", d["label"] if d else None, lg["label"]),
                                     ("category", d["category"] if d else None, lg["category"]),
                                     ("unit", d["unit"] if d else None, lg["unit"])):
                if regv is not None and regv != lgv:
                    errors.append(f"{cid}: logger {field} '{lgv}' != registry '{regv}'")
            if norm_unit(r["unit"]) != lg["unit"]:
                errors.append(f"{cid}: logger unit '{lg['unit']}' != normalized "
                              f"CSV unit '{norm_unit(r['unit'])}'")

    # ---- E. Source_File homes (v24.01 Phase 0.5) ---------------------------
    for r in csv_rows:
        cid = r["id"]
        want_src = expected_source(cid)
        if r.get("source") != want_src:
            errors.append(f"{cid}: Source_File '{r.get('source')}' != expected "
                          f"'{want_src}' (v24.01 panel home)")
        if args.logger:
            lg = logger_rows.get(cid)
            if lg is not None and lg["source"] != want_src:
                errors.append(f"{cid}: logger file '{lg['source']}' != expected "
                              f"'{want_src}'")

    # ---- report --------------------------------------------------------------
    if errors:
        print("RHIZOME CONTROL AUDIT: FAILED "
              f"({len(errors)} error{'s' if len(errors) != 1 else ''})")
        for e in errors:
            print("  - " + e)
        print("The 251-control contract (AGENTS.md Sec 1.3) is violated: "
              "param_controls_complete_list.csv and the UI control registry "
              "are out of sync. Fix the registry (or the CSV) and rebuild.")
        return 1

    kinds = {}
    for d in reg_rows:
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    print(f"RHIZOME CONTROL AUDIT: PASS — {EXPECTED_TOTAL}/{EXPECTED_TOTAL} controls, "
          f"{EXPECTED_SECTIONS} sections, CSV == registry"
          + (" == monitor catalog" if args.logger else "")
          + ", Source_File homes verified")
    print("  kinds: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    if args.stamp:
        with open(args.stamp, "w", encoding="utf-8") as f:
            f.write("pass\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
