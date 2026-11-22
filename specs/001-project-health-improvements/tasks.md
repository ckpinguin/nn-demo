---

description: "Task list for Project Health Improvements"
---

# Tasks: Project Health Improvements

**Input**: Design documents from `/specs/001-project-health-improvements/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/ (cli.md, tutorkit-api.md), quickstart.md

**Tests**: Requested. User Story 1 *is* the automated check (FR-001 to FR-008), and plan.md lists the test modules
for User Stories 2 and 3. Where a test covers new behaviour, write it first and see it fail. Tests that guard
existing behaviour (`test_data.py`, `test_slopes.py`) pass at once; the quickstart fault injection proves they bite.

**Organization**: Tasks are grouped by user story. All four stories touch either `tutorkit.py` or `README.md`, so
they are independently *testable* but should be *implemented* one after another, in priority order.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task belongs to (US1–US4)
- Paths are relative to the repository root. The project is flat: modules live in the root, tests in `tests/`.

---

## Phase 1: Setup

**Purpose**: A working environment, plus a reference export taken before any code changes.

- [X] T001 Make sure a working Python 3.14 or 3.15 environment with `requirements.txt` installed exists in `.venv/`. If `.venv/bin/python` is a broken link (it is today), rebuild it as in quickstart.md §0. Confirm with `python nn.py` (close the window).
- [X] T002 Before touching any code, export both courses into a reference folder outside the repository, as in quickstart.md §0: `python nn_tutor.py --save "$REF/nn"` and `python backprop_tutor.py --save "$REF/bp"`, with `REF=$(mktemp -d)`. Write the `REF` path down; T014, T022 and T029 compare against it. The files use the old names `step_NN.png`.
- [X] T003 [P] In `.gitignore`, replace `/__pycache__` with `__pycache__/` so that `tests/__pycache__/` is ignored too (research.md R1, FR-008).

---

## Phase 2: Foundational

**Purpose**: The `tests/` package that every story's tests live in.

**⚠️ CRITICAL**: No test task can run until this phase is complete.

- [X] T004 Create `tests/__init__.py`. It calls `matplotlib.use("Agg")` before anything imports pyplot (research.md R2), with a one-line docstring saying why. Verify that `python -m unittest` runs, finds 0 tests and opens no window.

**Checkpoint**: `python -m unittest` works from the repository root. User story work can begin.

---

## Phase 3: User Story 1 – One check that catches broken pages and silent drift (Priority: P1) 🎯 MVP

**Goal**: `python -m unittest` shows every page of both courses, catches text overflow, compares the data of
`nn.py` and `tinynet.py`, checks backprop against nudging, and keeps course 2's code labels on the right lines. It
needs no display and runs in under 2 minutes.

**Independent Test**: `python -m unittest -v` passes on the unchanged project (3 runs, `git status` clean). Each
fault in quickstart.md §2 (a, b, c, d, e2) makes it fail with a message naming what broke; fault e1 passes.

### Tests for User Story 1

- [X] T005 [P] [US1] Write `tests/test_data.py` (research.md R4; data-model.md Check result, row `tests/test_data.py`):
  - Run `nn.py` with `runpy.run_path`, with stdout redirected through `contextlib.redirect_stdout(io.StringIO())` and its `UserWarning` about a non-interactive canvas suppressed. Call `plt.close("all")` afterwards.
  - Assert `numpy.array_equal` for `X_train`, `X_test`, `y_train` and `y_test` against `tinynet`, with a message naming the array.
  - Assert `numpy.allclose` between the two `rescale(X_train)`.
  - Expected to pass at once.
- [X] T006 [P] [US1] Write `tests/test_slopes.py` (research.md R5; data-model.md Check result, row `tests/test_slopes.py`):
  - For seeds 0, 1 and 2, compare `Net.random(1, 2, seed)` on `office.X1, office.Y` and `Net.random(2, 4, seed)` on `office.X2, office.Y`: `net.backprop(X, Y).knobs()` against `net.nudge_slopes(X, Y)`.
  - Use `numpy.testing.assert_allclose(rtol=1e-5, atol=1e-8)`, with an `err_msg` naming inputs, hidden neurons and seed.
  - Assert `backprop_tutor.deep_check() < 1e-6`.
  - Expected to pass at once.
- [X] T007 [P] [US1] Write `tests/test_pages.py` (research.md R2, R3; data-model.md Step.title, Course.steps, Layout problem.scope; contracts/cli.md "Check"):
  - For each module (`nn_tutor`, `backprop_tutor`), create one `tutorkit.Tutor(module.course)`.
  - For every index `i`, inside `self.subTest(course=<module name>, step=i + 1, title=<step title>)`, call `tutor.show(i)` and then `tutor.fig.canvas.draw()`, and assert `tutor.layout_problems == []`.
  - An exception in one step must not stop the following steps; `subTest` handles this.
  - Close the figure in `tearDown`/`addCleanup`.
  - Expected to fail until T009, because `layout_problems` does not exist yet.
- [X] T008 [P] [US1] Write `tests/test_code_labels.py` (research.md R6; data-model.md Code label.line_text, Code label.label; contracts/tutorkit-api.md `code_box`):
  - Assert that `backprop_tutor.mark_lines(code_of(fn), marks)` resolves without error and returns one entry per label, for each pair (`Net.forward`, `FORWARD_MARKS`), (`Net.backprop`, `BACKPROP_MARKS`) and (`deep_slopes`, `DEEP_MARKS`).
  - Assert that an anchor matching no line, and an anchor matching two lines, each raise `ValueError` whose message contains the label and the text.
  - Do **not** pin line numbers, because quickstart fault e1 must pass.
  - Expected to fail until T010 and T011.

### Implementation for User Story 1

- [X] T009 [US1] In `tutorkit.py`, add `Tutor.layout_problems` (data-model.md Layout problem.message, Layout problem.scope):
  - Initialise it to `[]` in `Tutor.__init__` and reset it in `Tutor.show()` before the page is built.
  - In `Tutor.write()`, append each of the two existing notes ("too narrow", "overflows") with the same text it prints to stderr, and keep printing them.
  - T007 now passes.
- [X] T010 [US1] In `backprop_tutor.py`, add `mark_lines(code, marks) -> {line number: label}` next to `code_box` (data-model.md Code label.line_text, Code label.label; contracts/tutorkit-api.md):
  - Each key must occur in exactly one line. Otherwise raise `ValueError(f"code label {label!r}: {count} lines contain {text!r}")`.
  - Change `code_box` to resolve `marks` through it, so the drawing code is otherwise unchanged. Update the `code_box` docstring.
- [X] T011 [US1] In `backprop_tutor.py`, define the module constants `FORWARD_MARKS`, `BACKPROP_MARKS` and `DEEP_MARKS` with the anchors from research.md R6. Use them in the `code_box` calls of `page_code_forward` (step 24), `page_code_backward` (step 25) and `page_deep_code` (step 28) instead of the line-number dicts. Depends on T010. T008 now passes.
- [X] T012 [US1] In `README.md`, add a "Check" section (FR-017; contracts/cli.md "Check"):
  - The commands `python -m unittest` and `python -m unittest -v`.
  - That it opens no window and takes about a minute.
  - What it covers: all pages of both courses, text fit, the data of `nn.py` and `tinynet.py`, backprop against nudging, and code labels.
  - Add a `tests/` row to the Files table.
- [X] T013 [US1] Validate the check in `tests/` (quickstart.md §1): `time python -m unittest -v` passes three times in under 2 minutes, with no window and with `git status --short` showing only intended changes (SC-001, SC-003, FR-007, FR-008).
- [X] T014 [US1] Validate the code labels in `backprop_tutor.py` and the five faults:
  - Run `python backprop_tutor.py --save "$OUT"` into a fresh temporary folder. Confirm with `cmp` that steps 24, 25 and 28 are byte-identical to `$REF/bp/step_24.png`, `step_25.png` and `step_28.png` from T002 (FR-018).
  - Run quickstart.md §2 faults a, b, c, d, e1 and e2 one at a time in `backprop_tutor.py`, `nn_tutor.py`, `nn.py` and `tinynet.py`, reverting each with `git checkout -- <file>` (SC-002).

**Checkpoint**: The MVP. The maintainer has a one-command safety net for all later stories.

---

## Phase 4: User Story 2 – Exporting both courses without one overwriting the other (Priority: P2)

**Goal**: Exported files carry the course name, so both courses can share `pages/`. Re-exports remove their own
leftovers, and the repository has previews of both courses.

**Independent Test**: quickstart.md §3 gives 69 files (38 + 31). A planted `nn_tutor_step_99.png` is removed and
listed, and `notes.txt` is kept. quickstart.md §4 shows no `DIFF` lines.

### Tests for User Story 2

- [X] T015 [P] [US2] Write `tests/test_export.py` (research.md R7; data-model.md Course.name, Exported page.file_name, Exported page.location, Exported page.leftover; contracts/cli.md `--save DIR`):
  - Build two tiny courses, `name="demo_course"` with 3 steps and `name="other_course"` with 2 steps. Each has one part, an empty glossary and slider dict, and pages that only call `t.write("…")`.
  - In a `tempfile.TemporaryDirectory`, export both, along with an unrelated `notes.txt`. Assert that exactly `demo_course_step_01..03.png` and `other_course_step_01..02.png` exist, that `export()` returns the written paths in step order, and that `notes.txt` is unchanged.
  - Plant `demo_course_step_07.png` and re-export `demo_course`. Assert that the file is gone, that captured stdout contains `removed leftover demo_course_step_07.png`, and that `other_course` files are byte-identical to before.
  - Assert that `Course(..., name="Bad-Name")` raises `ValueError`.
  - Expected to fail until T016.

### Implementation for User Story 2

- [X] T016 [US2] In `tutorkit.py`, implement the export (contracts/tutorkit-api.md `Course`, `Course.export`, `Course.run`; contracts/cli.md):
  - Add a keyword-only required `name` to `Course.__init__`, validated against the pattern in data-model.md Course.name (raise `ValueError`).
  - Add `Course.export(out_dir)`: create the folder, render every step with a `Tutor`, and save `<name>_step_<NN>.png`. Then delete this course's leftovers, printing `removed leftover <file name>` for each, and return the written paths (data-model.md Exported page.file_name, Exported page.location, Exported page.leftover).
  - Make `Course.run()` with `--save DIR` call `export(DIR)` and print `saved <count> steps of <name> to <DIR>`.
  - T015 now passes.
- [X] T017 [P] [US2] In `nn_tutor.py`, pass `name="nn_tutor"` to `Course(...)` (data-model.md Course.name). Depends on T016.
- [X] T018 [P] [US2] In `backprop_tutor.py`, pass `name="backprop_tutor"` to `Course(...)` (data-model.md Course.name). Depends on T016.
- [X] T019 [US2] Rename the committed previews in `pages/` (data-model.md Exported page.committed_previews; research.md R8):
  - Rename the 38 files with `git mv pages/step_NN.png pages/nn_tutor_step_NN.png`.
  - Run `python backprop_tutor.py --save pages` to add the 31 course 2 pages.
  - Run `python nn_tutor.py --save pages`, then `git status`. It must show only the 38 renames (no content change) and 31 new files.
- [X] T020 [US2] In `README.md`, update the export documentation (FR-010, FR-012; contracts/cli.md `--save DIR`):
  - Explain that `--save pages` writes `<name>_step_NN.png`, that both courses can share the folder, and that leftover pages of a course are removed.
  - Add a `pages/` row to the Files table.
  - Show one preview per course, e.g. `pages/nn_tutor_step_29.png` and `pages/backprop_tutor_step_25.png`.
- [X] T021 [US2] Validate quickstart.md §3 with `nn_tutor.py` and `backprop_tutor.py`: 69 files, leftover removed and listed, `notes.txt` kept (SC-004, FR-009 to FR-011).
- [X] T022 [US2] Validate quickstart.md §4: no `DIFF` against the committed previews in `pages/` or the reference export from T002 (FR-018). Run `python -m unittest` again; it must pass.

**Checkpoint**: User Stories 1 and 2 both work. The check still passes.

---

## Phase 5: User Story 3 – No unexplained pauses when stepping through course 1 (Priority: P3)

**Goal**: The first visit to steps 22 and 32–35 shows a "Preparing this page…" notice at once in a window, then the
page. Exports, tests and revisits never show the notice.

**Independent Test**: quickstart.md §5 (by eye in a window) and §4 (exports unchanged). `tests/test_prepare.py`
passes.

### Tests for User Story 3

- [X] T023 [P] [US3] Write `tests/test_prepare.py` (research.md R10; data-model.md Step.prepare, Page preparation state; contracts/tutorkit-api.md `Tutor`, `Tutor.build`, `Tutor.prepared`). Build a tiny course (`name="prep_course"`) with step 1 without `prepare` and step 2 with a counting `prepare`. Pages record into a list when they run.
  - **Not interactive** (`Tutor(course)`): `show(1)` calls `prepare` once, before the page; no notice text exists; `1 in tutor.prepared`. A second `show(1)` does not call `prepare` again.
  - **Interactive** (`Tutor(course, interactive=True)`): `show(1)` leaves the page not yet run, `prepare` not called, and a visible text containing "Preparing" among the figure's texts. Then `tutor.build(1)` removes the notice, calls `prepare` once and runs the page. A later `show(1)` builds directly, with no notice.
  - **Cancel**: interactive `show(1)` followed by `show(0)` leaves `prepare` uncalled and `1 not in tutor.prepared`.
  - Expected to fail until T024 and T025. Depends on T016, because the tiny course needs `name=`.

### Implementation for User Story 3

- [X] T024 [US3] In `tutorkit.py`, store steps as a small named tuple `Step(part, title, page, prepare)` and add `prepare=None` to `Course.step(title, prepare=None)` (data-model.md Step.part, Step.title, Step.page, Step.prepare). Update every unpacking of steps in `Course.no`, `Tutor.__init__` and `Tutor.show`; the data-model.md "Step (page)" note lists them. Existing `step(title)` calls keep working.
- [X] T025 [US3] In `tutorkit.py`, implement the deferred first visit (data-model.md Page preparation state; contracts/tutorkit-api.md `Tutor`, `Tutor.show`, `Tutor.build`, `Tutor.prepared`). Depends on T024. T023 now passes.
  - Add `Tutor(course, interactive=False)` and `Tutor.prepared = set()`.
  - Add a single-shot build timer of about 50 ms, created like `self.switch`.
  - In `show()`: stop a pending build timer. For a step with `prepare` that is not in `prepared`: if `interactive`, add an owned "Preparing this page…" notice in the plot area, call `draw_idle()` and start the timer, which calls `build(index)`. Otherwise call `prepare()` directly and then build.
  - Add `Tutor.build(index)`: remove the notice, call `prepare()`, add the index to `prepared` and build the page.
  - Make `Course.run()` in window mode create `Tutor(self, interactive=True)`.
  - Keep the comment style plain, like the existing `go()` docstring.
- [X] T026 [US3] In `nn_tutor.py`, add `prepare=` to five step registrations (data-model.md Step.prepare):
  - `@step("The learning rate", prepare=eta_runs)` (step 22)
  - `@step("How many hidden neurons?", prepare=hidden_runs)` (step 32)
  - `@step("Different starts, different results", prepare=start_runs)` (step 33)
  - `@step("Too flexible: overfitting", prepare=overfit_run)` (step 34)
  - `@step("The honest test", prepare=sklearn_net)` (step 35)
- [ ] T027 [US3] Validate quickstart.md §5 in a window with `nn_tutor.py`: the notice appears at once on the first visit to steps 22 and 32–35, revisits show no notice, quick repeated → presses do not crash, and startup is not visibly slower (SC-005, SC-006, FR-013, FR-014). Record the observed behaviour in the task's completion note.
  - **Open, by eye**: during `/speckit-implement`, no window could be opened. Measured instead with `Tutor(interactive=True)` and one redraw per page (research.md R9): notice after 0.02 s on steps 22 and 32–35, page 0.5–2.0 s later, revisits at most 0.16 s with no notice, startup to step 1 1.46 s. Still to check in a real window: the notice is painted before the computation starts.
- [X] T028 [US3] Validate exports again: repeat quickstart.md §4 with a fresh export of `nn_tutor.py` and `backprop_tutor.py` (no `DIFF`, so no notice leaks into exports, FR-018) and run `python -m unittest` (it must pass).

**Checkpoint**: User Stories 1–3 work. Exports are still byte-identical.

---

## Phase 6: User Story 4 – Setup instructions that hold up after a system update (Priority: P3)

**Goal**: The README names the verified Python versions and explains how to recover from an environment broken by a
Python upgrade.

**Independent Test**: quickstart.md §6. Someone who follows only the README gets everything running, and can rebuild
a broken `.venv`.

### Implementation for User Story 4

- [X] T029 [US4] In `README.md`, update the Setup section: replace "Tested with Python 3.14 and the versions pinned in `requirements.txt`" with "Verified with Python 3.14 and 3.15.2 and the versions pinned in `requirements.txt`" (FR-015; research.md R11).
- [X] T030 [US4] In `README.md`, add a short "Troubleshooting" section after Setup (FR-016; research.md R11):
  - **Symptom**: after a Python upgrade, for example through Homebrew, commands fail with "no such file or directory" or "bad interpreter" for the environment's `python`.
  - **Cause**: `.venv` points to a Python that was removed.
  - **Fix**: `rm -rf .venv`, then repeat the three setup commands.
- [X] T031 [US4] Validate quickstart.md §6: follow only `README.md` in a fresh clone (`git clone . "$TMPDIR/nn-demo-check"`) with Python 3.15. Install, run `nn.py`, both courses and `python -m unittest`, then find the troubleshooting note and the previews (SC-007).

**Checkpoint**: All four user stories are complete.

---

## Phase 7: Polish & Cross-Cutting Concerns

- [X] T032 Read the full diff of `tutorkit.py`, `nn_tutor.py` and `backprop_tutor.py` as a learner would: short plain-English comments in the project's style, no unused code or leftover debug output. Make sure `tinynet.py`, `office.py` and `nn.py` are unchanged (`git diff --stat`).
- [X] T033 Run the whole of quickstart.md, §1 to §6, once more on the final state. Run `python -m unittest -v` and keep its summary line, with the run time, for the commit message.
- [X] T034 Set **Status** in `specs/001-project-health-improvements/spec.md` from "Draft" to "Implemented", and note any deviation from plan.md found during implementation in `specs/001-project-health-improvements/research.md`.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies. T002 **must** finish before any code change.
- **Foundational (Phase 2)**: T004 depends on T001 and blocks all test tasks.
- **User Stories (Phases 3–6)**: All depend on Phase 2. Implement them in priority order, US1 → US2 → US3 → US4,
  because they share `tutorkit.py` (US1, US2, US3) and `README.md` (US1, US2, US4).
- **Polish (Phase 7)**: Depends on all stories.

### User Story Dependencies

- **US1 (P1)**: Depends only on Phase 2. It is the MVP and the safety net for everything after it.
- **US2 (P2)**: Testable without US1 (`test_export.py` uses tiny courses). In practice US1 comes first, so the page
  check guards the `Course` changes.
- **US3 (P3)**: `test_prepare.py` needs `Course(name=...)` from T016 (US2). Otherwise it is independent.
- **US4 (P3)**: README text only. It can be done at any time after T001, sequenced with other README edits (T012,
  T020).

### Within Each Story

- Test task first; confirm it fails where new behaviour is tested (T007, T008, T015, T023).
- `tutorkit.py` changes before the course modules that use them (T016 → T017, T018; T024 → T025 → T026).
- Validation tasks last in each phase.

---

## Parallel Opportunities

- **Setup**: T003 can run next to T001 and T002.
- **US1**: T005, T006, T007 and T008 are four independent new test files. T009 (`tutorkit.py`) can run in parallel
  with T010 → T011 (`backprop_tutor.py`).
- **US2**: T017 and T018 touch different course files after T016. T015 can be written while T019's renames are
  prepared.
- **US3**: T023 can be written while T024 is in progress.
- **US4**: T029 and T030 can be done any time but should not overlap with other README edits.

### Parallel Example: User Story 1

```bash
# Four test files at once (all [P], different files):
Task: "Write tests/test_data.py (T005)"
Task: "Write tests/test_slopes.py (T006)"
Task: "Write tests/test_pages.py (T007)"
Task: "Write tests/test_code_labels.py (T008)"

# Then two implementation tracks in different files:
Task: "Tutor.layout_problems in tutorkit.py (T009)"
Task: "mark_lines + anchored marks in backprop_tutor.py (T010 → T011)"
```

### Parallel Example: User Story 2

```bash
# After T016 (tutorkit.py):
Task: "name='nn_tutor' in nn_tutor.py (T017)"
Task: "name='backprop_tutor' in backprop_tutor.py (T018)"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Phase 1 (T001–T003), including the reference export T002.
2. Phase 2 (T004).
3. Phase 3 (T005–T014).
4. **Stop and validate**: `python -m unittest -v` passes, and the faults from quickstart §2 are caught. Commit.

### Incremental Delivery

1. US1 gives the check (MVP). Commit.
2. US2 gives safe exports and previews of both courses; the check must still pass. Commit, including the `pages/`
   renames and additions.
3. US3 gives the preparation notice; exports must stay byte-identical. Commit.
4. US4 gives the README setup notes. Commit.
5. Polish (T032–T034).

---

## Notes

- `tinynet.py` must stay unchanged, because course 2 shows its source to learners. Only quickstart fault injection
  touches it, and each fault is reverted at once.
- `nn.py` stays a standalone script; the check only runs it.
- Byte-identity checks (T014, T022, T028) assume the same machine and library versions as the reference export
  (T002).
- Commit only when the maintainer asks, and never commit `.venv/` or anything from the scratch folders.
