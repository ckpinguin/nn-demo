# Implementation Plan: Project Health Improvements

**Branch**: `main` (the setup script reports `001-project-health-improvements`; no such git branch exists) |
**Date**: 2026-11-22 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-project-health-improvements/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command; its definition describes the execution workflow.

## Summary

This plan adds a safety net around the two courses and the standalone demo, plus three smaller fixes.

- **Check (P1)**: a standard-library `unittest` suite, run with `python -m unittest` and no window. It shows every page
  of both courses, records text that does not fit instead of only printing it, and runs `nn.py` to compare its data
  with `tinynet.py`. It also checks the hand-written slopes against slopes found by nudging. Course 2's code labels
  are tied to a piece of line text instead of a line number, so they follow the code or fail loudly.
- **Export (P2)**: page files carry the course name (`nn_tutor_step_01.png`). An export removes and lists its own
  leftover pages from an earlier, longer export. The committed previews are renamed, and course 2's previews are
  added.
- **Slow pages (P3)**: pages that compute something slow declare it when registered. In a window they show a
  "Preparing…" notice first and build the page a moment later.
- **README (P3)**: verified Python versions, how to rebuild a broken environment, how to run the check, and where the
  previews are.

The reasoning for each decision is in [research.md](research.md).

## Technical Context

**Language/Version**: Python 3.14 and 3.15. All three programs were verified on 3.15.2 on 2026-11-22.

**Primary Dependencies**: numpy 2.5.3, matplotlib 3.11.2, scikit-learn 1.9.1 (pinned in `requirements.txt`). No new
dependencies.

**Storage**: Files only. PNG page exports go into a folder the user chooses; committed previews live in `pages/`.

**Testing**: Standard-library `unittest`, run as `python -m unittest` from the repository root with the Agg backend
(no display).

**Target Platform**: Desktop with a matplotlib window backend (developed on macOS). The check runs on any machine,
with or without a display.

**Project Type**: Desktop teaching application (two matplotlib window courses) plus a standalone script.

**Performance Goals**:

- The check finishes in under 2 minutes; about 1 minute is expected, since rendering 69 pages takes about 30 s and
  `nn.py` about 3 s.
- Some feedback appears within 0.5 s of navigating to any page.
- Startup is at most 1 s slower than today.

**Constraints**:

- Pages look exactly as they do today: course 1's re-export stays byte-identical to the committed previews on the
  same machine and library versions.
- `tinynet.py` stays unchanged, because course 2 shows its source to learners.
- `nn.py` stays a standalone top-to-bottom script.
- All window code stays single-threaded.

**Scale/Scope**:

- About 4,100 lines in 6 modules.
- 69 pages: 38 in course 1, 31 in course 2.
- 5 slow pages: course 1 steps 22 and 32–35.
- 3 labelled code listings: course 2 steps 24, 25 and 28.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

`.specify/memory/constitution.md` is still the unfilled template: it has no ratified principles, so there are no
gates to evaluate. **Result: PASS (no gates).** The post-design re-check gives the same result. Running
`/speckit-constitution` would give later features real gates, for example "no new dependencies" or "pages must stay
byte-identical".

## Project Structure

### Documentation (this feature)

```text
specs/001-project-health-improvements/
├── plan.md              # This file
├── research.md          # Phase 0: decisions and measurements
├── data-model.md        # Phase 1: entities, rules, preparation states
├── quickstart.md        # Phase 1: validation guide
├── contracts/
│   ├── cli.md           # command-line behaviour: courses, export, check
│   └── tutorkit-api.md  # the course-building interface the two courses use
├── checklists/
│   └── requirements.md  # spec quality checklist
└── tasks.md             # Phase 2 (/speckit-tasks, not created here)
```

### Source Code (repository root)

```text
nn-demo/
├── nn.py                   # unchanged; the check runs it to compare its data
├── tinynet.py              # unchanged (course 2 shows its source to learners)
├── office.py               # unchanged
├── tutorkit.py             # Course(name=...), Course.export(), course.step(title, prepare=...),
│                           # Tutor(interactive=...), Tutor.layout_problems, the "Preparing…" notice
├── nn_tutor.py             # Course(..., name="nn_tutor"); prepare= on steps 22, 32, 33, 34, 35
├── backprop_tutor.py       # Course(..., name="backprop_tutor"); code_box marks keyed by line text
├── tests/
│   ├── __init__.py         # selects the Agg backend before anything imports pyplot
│   ├── test_pages.py       # every page of both courses: no error, text fits (FR-002, FR-003)
│   ├── test_data.py        # nn.py and tinynet.py use the same afternoons (FR-004)
│   ├── test_slopes.py      # backprop vs nudging; deep_check (FR-005)
│   ├── test_code_labels.py # labels resolve to exactly one line (FR-006)
│   ├── test_export.py      # file names, no overwriting, leftovers (FR-009, FR-011)
│   └── test_prepare.py     # notice and deferred build in a window, none in exports (FR-013, FR-018)
├── pages/
│   ├── nn_tutor_step_01.png … nn_tutor_step_38.png              # renamed from step_NN.png
│   └── backprop_tutor_step_01.png … backprop_tutor_step_31.png  # new
├── README.md               # versions, troubleshooting, check, export names, previews
├── .gitignore              # `/__pycache__` → `__pycache__/` so tests/__pycache__ is ignored too
└── requirements.txt        # unchanged
```

**Structure Decision**: The flat root layout stays as it is, because learners open these files directly. The only new
folder is `tests/`, a package so that plain `python -m unittest` finds it from the root. Discovery was verified on
Python 3.15.2: it imports root modules and skips `.venv`. Course content files change only where a requirement needs
it: the course name, `prepare=` on five steps, and the label anchors on three code listings.

## Complexity Tracking

No constitution gates exist, so there are no violations to justify.
