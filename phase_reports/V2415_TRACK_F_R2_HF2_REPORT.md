# V24.15 TRACK F R2-HF2 RELEASE REPORT — Growth-Panel ImGui ID-Conflict Fix (Hotfix)

Release: v24.15 (Track F R2-HF2), 2026-10-24
Supersedes: v24.14 (`rhizome_v24.14_compile_fix.zip` — it compiled and ran, but
its Growth panel carried an ImGui ID conflict between two "Branch Angle"
sliders; this hotfix repairs exactly that and retires the zip).
Sandbox tree: `rhizome_current/` (v24.15). Records: `AGENTS.md` Sec 19,
`TRACK_F_R2_NOTE.md` Sec 10.

## 1. Trigger

The user's first interactive pass on the compiled v24.14 reported: "in
'growth' -> 'algorithm' -> 'branch angle' has conflicting ID with 'growth' ->
'vessel & caliber' -> 'branch angle'". The second slider actually lives in the
LATERAL ROOTS section — the section directly beneath VESSEL & CALIBER (that
section itself carries no Branch Angle row); with all section headers
expanded, the two identically-labeled rows sit a few sections apart in the
same Growth window, and dragging either one moved both.

## 2. Root cause

The compact widget family (`ui/widgets/widget_factory.cpp`:
`render_compact_slider_float`, `render_compact_slider_int`,
`render_toggle_switch`) opens with `ImGui::PushID(label)` and derives every
item and edit ID from the draw label alone:

```cpp
ImGui::PushID(label);
ImGuiID slider_id = ImGui::GetID("##c_slider_f");   // edit-box identity
...
ImGui::InvisibleButton("##c_track_f", ...);          // drag track identity
```

`render_control` (the registry's generic renderer) passed the registry
`draw_label` straight through — so two registry controls sharing a draw label
inside one ImGui window collapsed onto the same ImGuiID. The Growth window
carries exactly that: three "Branch Angle" rows —

| Control | Section | Rendered | Binding |
| --- | --- | --- | --- |
| CTRL016 "Murray Branch Angle" | Algorithm header, mode-2 branch | mode 2 only | `murray_branch_angle` |
| CTRL040 "Branch Angle" | Lateral Roots | every mode | `lateral_branch_angle` |
| CTRL246 "Branch Angle" | Algorithm header, mode-4 branch | mode 4 only | `dev_branch_angle` |

Mode 2 co-renders CTRL016+CTRL040 (a v23.28-heritage collision that existed
since the widget family was introduced); mode 4 (v24.12) added the pair the
user hit while testing Root Typ: CTRL246+CTRL040. With one shared active ID:

- `IsItemActive()` returns true for BOTH track submissions in the same frame,
  so the drag branch `if (is_active && IsMouseDown)` executes for both —
  each writes its own bound value from the same mouse-position fraction
  (`dev_branch_angle` and `lateral_branch_angle` move together, each mapped
  to its own range: 30-85 deg vs 15-85 deg).
- The value-edit identity `g_active_slider_edit_id == slider_id` matches both
  rows, so clicking either value column opens the edit InputText on both
  (and the two `##c_in_f` InputTexts themselves share an ID).

A whole-registry census (all 251 entries parsed) found exactly one other
duplicate draw label — "Spores" (CTRL017, Growth window / CTRL123, Scene
window) — which never collided because ImGui IDs are window-scoped and the
two live in different windows.

## 3. Why the fix sits in render_control (and not in the labels)

Renaming draw labels (e.g. "Murray Branch Angle" / "Lateral Branch Angle")
would change visible UI text — including v23.28-heritage strings in modes 0-2
— and invites drift between `label` (the pinned catalog name in the CSV /
inventory / audit) and `draw_label`. Scoping by the registry's own unique key
(`def.id`, "CTRLnnn" — enforced unique by `control_registry_integrity_check`)
fixes the entire 251-control registry at the single choke point every panel
already routes through, with zero visual and zero catalog impact.

## 4. The fix

`ui/widgets/control_registry.cpp`: the `render_control` switch body moved
verbatim to a file-local `static bool render_control_value(...)`, and the
public `render_control` becomes a thin ImGui-ID-scoped wrapper:

```cpp
bool render_control(const ControlDefinition& def, AppState& state,
                    const float* f_min_override, const float* f_max_override) {
    ImGui::PushID(def.id);
    const bool changed = render_control_value(def, state, f_min_override, f_max_override);
    ImGui::PopID();
    return changed;
}
```

Every control's item and edit IDs become unique per window regardless of
draw-label collisions, for all five value kinds (SliderFloat / SliderInt /
Toggle / ColorEdit4 / InputText) and every panel. Properties preserved:

- `PushID` draws nothing — zero visual change for every control in every
  panel (the label is drawn by the widgets themselves, verbatim as before).
- `IsItemHovered()` semantics after `render_control` returns are unchanged
  (`PopID` does not touch `LastItemData`), so the panels' tooltip call sites
  behave identically.
- The edit-id (`g_active_slider_edit_id`) is computed inside the same
  `PushID(def.id)` scope every frame, so it stays stable across frames and
  now identifies exactly one control per window.
- The direct panel-local `render_toggle_switch(CTRL047/048/049.draw_label,
  ...)` calls (per-seed toggles) keep their unique labels and needed no
  change; the pill buttons / combos already use `##`-suffixed literal IDs.
- The shared widgets in `widget_factory.cpp` are untouched.
- Modes 0-3 engine behavior: byte-identical (UI-layer-only change; the
  growth engine, kernels, defaults, and bake path are not touched).
- The heritage mode-2 collision (CTRL016 vs CTRL040) heals with the same
  change.

## 5. Verification (sandbox, text-level)

- New `scripts/p20_ui_id_checks.py`: fix shape (impl renamed + wrapper order
  PushID -> delegate -> PopID, switch in the impl only), duplicate-draw_label
  census == exactly the two known groups, the Branch Angle trio rows pinned
  byte-exact (guards against future string-rename "fixes"), repro-context
  render sites (CTRL246 in the mode-4 branch, CTRL016 in the mode-2 branch,
  CTRL040 in the Lateral Roots block — all via `render_control`), widget
  sentinels (5 `PushID(label)` sites, `g_active_slider_edit_id` file-local),
  CSV pinned (252 lines; the three Branch Angle rows) + live 251/26 catalog
  audit PASS, delimiter balance + the P19 bug-class sweep on the edited
  source, doc stamps, and the carry-forward R1+R2 suites. Result: ALL PASS.
- Pre-edit identity vs `manifests/v24.14_track_f_r2_hf.sha256`: strict PASS
  (0/0/0) before any edit.
- Phase-end delta gate vs the same manifest, declared touch-set of 6 files
  (`ui/widgets/control_registry.cpp` + the five stamped docs): EXACT MATCH.
- Snapshot `manifests/v24.15_track_f_r2_hf2.sha256` generated; strict identity
  vs the new manifest: 0/0/0.
- Packaging `rhizome_v24.15_ui_id_fix.zip`: census 20 root notes + 21
  manifests + 21 phase reports + 81 governed app files; packaged-tree
  identity 81/81 vs the v24.15 manifest; Track F R2/R1 product markers and
  the v24.08-v24.11 ride-along markers verified inside the zip; byte-stable
  across consecutive runs.

## 6. Compatibility

- Modes 0-3: engine byte-identical; UI interaction identical except the
  duplicate-ID bug disappearing (in mode 2, the Murray Branch Angle and
  Lateral Roots Branch Angle sliders are now independent — previously they
  cross-dragged since v23.28).
- Mode 4: identical behavior; the Algorithm-header Branch Angle and Lateral
  Roots Branch Angle sliders are now independent.
- Catalog: unchanged, 251 controls / 26 sections; no registry data, CSV, or
  monitor churn (the live audit re-run PASSes untouched).
- Presets, algorithm files, watchlogs, structure-dump format: unchanged.
- Watchdog v1.2: unchanged, still current.
- imgui.ini: window layout keys unaffected (widget state was never persisted
  by ID).

## 7. Open items

- The v24.13/v24.14 acceptance pass (crown-anchored root ball on the real
  mug, [Print Audit] + [Structure Dump] lines, live skeleton animation, the
  quick-second-Start repro riding the v24.11 guards, export + slice) still
  has not run — the compiled v24.14 UI defect blocked meaningful tuning. It
  resumes unchanged on v24.15, now with both Branch Angle sliders
  independently draggable.
- Process note: the duplicate-draw_label census is now a standing machine
  check (`p20_ui_id_checks.py` 2a) — any future registry addition that
  reuses a draw label inside a window will fail the census before shipping
  (and is harmless by construction anyway while the `PushID(def.id)`
  wrapper stands).
- `rhizome_v24.14_compile_fix.zip` retired from `download/` per the
  single-landmark principle: v24.15 carries every ride-along (v24.08-v24.14)
  inside.
