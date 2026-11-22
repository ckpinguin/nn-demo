# Research: Project Health Improvements

All measurements were taken on 2026-11-22 on the maintainer's laptop with Python 3.15.2, numpy 2.5.3, matplotlib
3.11.2 and scikit-learn 1.9.1. The Technical Context in [plan.md](plan.md) has no open NEEDS CLARIFICATION items;
the decisions below settle every design question the spec leaves open.

## R1. How the check is run

- **Decision**: Use a standard-library `unittest` suite in a `tests/` package, run with `python -m unittest` (add
  `-v` for one line per test) from the repository root.
- **Rationale**: It needs no new dependency, and `requirements.txt` stays limited to what learners need. `subTest`
  reports each page separately, with course, step and title, and a failure on one page does not hide failures on
  later pages. A `tests/` package keeps the root folder, which learners browse, uncluttered. A scratch layout
  confirmed that discovery finds `tests/test_*.py`, imports root modules and skips `.venv`, which has no
  `__init__.py`.
- **Side effect handled**: importing test modules creates `tests/__pycache__/`, and `.gitignore` currently ignores
  only the root `/__pycache__`. It is changed to `__pycache__/`, which matches every level, so that a check run
  leaves `git status` unchanged (FR-008).
- **Alternatives considered**:
  - pytest: nicer output and parametrisation, but it adds a dependency that only the maintainer needs.
  - A home-made `check.py`: it would reinvent test discovery and reporting.
  - Test files in the root folder: they would clutter the file list the README walks learners through.

## R2. Showing every page without a display

- **Decision**: `tests/__init__.py` selects the Agg backend before anything imports pyplot. `test_pages.py` builds
  one `Tutor` per course and calls `show(i)` and then `fig.canvas.draw()` for every step, inside
  `subTest(course=..., step=..., title=...)`. It closes the figure afterwards.
- **Rationale**: `--save` already renders every page with Agg, so Agg is known to work for all 69 pages: the scan
  exported them all with no errors and no layout notes. Today `tutorkit.py` switches to Agg only when `--save` is in
  `sys.argv`, so the tests must select it themselves.
- **Alternatives considered**:
  - Running `python nn_tutor.py --save tmp` as a subprocess: it reports only the first error, with no step name.
  - Interacting with the widgets (sliders, buttons) on every page: useful later, but beyond FR-002, which only asks
    that each page opens.

## R3. Detecting text that does not fit

- **Decision**: `Tutor.write()` keeps printing its two notes ("too narrow", "overflows") to stderr and also appends
  them to `Tutor.layout_problems`, which `show()` resets. The page test asserts that the list is empty after each
  page.
- **Rationale**: This is a two-line change to existing logic that already uses the right thresholds (12 px
  sideways, 0.02 of the panel height at the bottom). Today no page trips them, so the check produces no false alarms
  (spec edge case).
- **Alternatives considered**: Capturing and parsing stderr is brittle, and it couples the test to wording.

## R4. Checking that `nn.py` and `tinynet.py` use the same afternoons

- **Decision**: `test_data.py` runs `nn.py` with `runpy.run_path`, with stdout redirected and Agg's "cannot show"
  warning suppressed. It then compares `X_train`, `X_test`, `y_train` and `y_test` from the resulting namespace with
  `tinynet`'s arrays using `numpy.array_equal`. It also compares `rescale` on `X_train` with `numpy.allclose`.
- **Rationale**: `nn.py` must stay a standalone script that can be read from top to bottom (spec assumption), so it
  cannot import its data from `tinynet.py`. Running it whole takes about 2–3 s. That adds a free check that the demo
  still runs, which also covers "checks … the standalone demo" in FR-001.
- **Alternatives considered**:
  - Moving `nn.py`'s data into a shared module: this breaks its standalone readability.
  - Executing only the data section of `nn.py`, cut out at a comment marker: brittle.
  - Comparing source text: it misses semantic changes and flags harmless formatting.

## R5. Checking the slopes

- **Decision**: `test_slopes.py` compares `Net.backprop(X, y).knobs()` with `Net.nudge_slopes(X, y)` using
  `numpy.testing.assert_allclose(rtol=1e-5, atol=1e-8)`. It does this for three seeds each of a network with 1 input
  and 2 hidden neurons on `office.X1` and one with 2 inputs and 4 hidden neurons on `office.X2`. It also asserts that
  `backprop_tutor.deep_check()` (deep sigmoid network versus nudging) is below `1e-6`.
- **Rationale**: For all six networks the largest absolute difference was 1.2e-11, and `deep_check()` returned
  1.6e-11. The smallest slope was 2.8e-4, far above the absolute tolerance. A single wrong term gives errors many
  orders of magnitude larger: a missing `H * (1 - H)`, `sum` instead of `mean`, or a missing transpose. So the tolerance is safe against noise and still catches every single-term mistake
  (FR-005, SC-002). The quickstart mutation checks this.
- **Alternatives considered**: Comparing with scikit-learn's fitted weights does not test the backward pass; it only
  tests where training ended up.

## R6. Code labels that follow the code (course 2, steps 24, 25, 28)

- **Decision**: `code_box(..., marks=...)` takes `{line text: label}` instead of `{line number: label}`. Each key is
  a piece of text that must occur in **exactly one** line of the listing, and the label goes on that line. If a key
  matches no line or several lines, `code_box` raises `ValueError` and names the label and the text. The page then
  fails in the page test, which names the course and step (FR-006). The anchors that reproduce today's positions:
  - Step 24, `Net.forward`: `"H = sigmoid("` → hidden layer, `"p = sigmoid("` → output.
  - Step 25, `Net.backprop`: `"self.forward(X)"` → forward, `"e = p - y"` → step 1, `"D = np.outer"` → steps 2 + 3,
    `"return Net("` → step 4.
  - Step 28, `deep_slopes`: `"H = [X]"` → forward, `"D = (p - y)"` → start, `"for k in reversed"` → backward,
    `"dWs.insert"` → knobs, `"D = (D @ Ws[k])"` → blame back.
- **Rationale**: Inserting or removing lines moves the labels with the code. Rewriting a labelled line fails loudly
  instead of quietly labelling the wrong line. The anchors resolve to the same lines as today, so the pages stay
  byte-identical (FR-018).
- **Alternatives considered**:
  - Comment tags such as `# [step 1]` inside `tinynet.py`: they would appear in the code learners read.
  - A test that pins today's line numbers: it would still need manual updates and would not move the labels.

## R7. Export file names and leftover pages

- **Decision**: Each course gets an explicit `name` (`"nn_tutor"`, `"backprop_tutor"`), and exports are written as
  `<name>_step_<NN>.png` into the chosen folder. A new `Course.export(out_dir)` does the export, and `run()` calls
  it. After writing, it deletes files in that folder that match `^<name>_step_(\d+)\.png$` and have a step number
  above the course's current step count. It prints each deleted file name and returns the written paths.
- **Rationale**: Both courses can share `pages/`, so the README command `--save pages` stays valid for both, and no
  file of one course can overwrite the other's (FR-009, FR-010). Only the course's own, provably stale files are
  removed, so the folder holds exactly the current pages (FR-011). An explicit name keeps working whether the course
  is started as a script, as a module or from a test. A separate `export()` lets `test_export.py` run on a tiny
  three-page course in well under a second.
- **Alternatives considered**:
  - One subfolder per course (`pages/nn_tutor/step_01.png`): it changes what `--save DIR` means and lengthens
    README paths.
  - Only warning about leftovers: it leaves the cleanup to the user.
  - Emptying the folder before export: destructive for any other files there.
  - Deriving the name from `sys.argv[0]`: wrong under `python -m` and in tests.

## R8. Committed preview pages

- **Decision**: Rename the 38 committed files with `git mv pages/step_NN.png pages/nn_tutor_step_NN.png`, so their
  history is kept and their content is unchanged. Add course 2's 31 pages (about 3.7 MB) from a fresh export. The
  README gets a `pages/` row in its file table and shows one preview image per course.
- **Rationale**: This meets FR-012 and the spec's assumption that 3.7 MB of images is acceptable. A fresh course 1
  export after the code changes must match the renamed files byte for byte, which also verifies FR-018.
- **Alternatives considered**: Generating previews only on demand: the published repository would show no course 2
  preview (spec User Story 2, scenario 3).

## R9. Measuring page latency without a window

- **Finding**: With Agg, `draw_idle()` redraws immediately, whereas window backends batch redraws. Every slider a
  page creates therefore triggers a full extra redraw headless, so headless timings overstate slider-heavy pages. For
  example, step 16 measures 0.54 s headless but 0.18 s or less with batched redraws. A fair estimate patches
  `FigureCanvasBase.draw_idle` to do nothing and does one `draw()` per page.
- **Measured first visits, batched redraws**:

  | Step | Title | Cached computation | Compute | First visit | Revisit |
  | --- | --- | --- | --- | --- | --- |
  | 34 | Too flexible: overfitting | `overfit_run` | 1.80 s | 1.94 s | 0.12 s |
  | 33 | Different starts, different results | `start_runs` | 0.87 s | 1.08 s | 0.15 s |
  | 32 | How many hidden neurons? | `hidden_runs` | 0.64 s | 0.76 s | 0.11 s |
  | 35 | The honest test | `sklearn_net` | 0.46 s | 0.61 s | 0.13 s |
  | 22 | The learning rate | `eta_runs` | 0.36 s | 0.46 s | 0.11 s |

  Every other page in both courses takes 0.24 s or less. Course 2's cached functions take under 0.01 s each. The spec
  has been corrected with these numbers: an earlier draft said "three pages" and "under 0.4 s".

## R10. Slow pages: notice first, then build

- **Decision**: `course.step(title, prepare=None)` accepts a zero-argument function that does the page's slow
  computation. It must be cached and must not touch matplotlib. Steps 22, 32, 33, 34 and 35 pass `eta_runs`,
  `hidden_runs`, `start_runs`, `overfit_run` and `sklearn_net`. `Tutor(course, interactive=False)` gets
  `interactive=True` only from `Course.run()` in window mode.
  - **In a window**, on the first visit to such a step, `show()` does the usual cleanup and sets the title, counter
    and progress bar. It then adds a "Preparing this page…" notice in the plot area and calls `draw_idle()`. A
    single-shot timer of about 50 ms then removes the notice, calls `prepare()`, marks the step prepared and builds
    the page. If the learner navigates away before the timer fires, `show()` stops that timer.
  - **Otherwise** (export, tests and revisits), `prepare()` runs directly before the page is built, so no notice is
    ever drawn. Exports therefore stay byte-identical.
- **Rationale**: The notice appears after about one light redraw (well under 0.5 s), which meets FR-013 and SC-005
  for all five pages. Startup does not change, because step 1 has no `prepare` (FR-014); opening with `--step 34`
  shows the notice sooner than today's blank wait. The timer goes through the event loop, so no re-entrant event
  processing occurs. This matters because `tutorkit` already defers `show()` through a timer for exactly that reason
  (`Tutor.go`). Clicks queued while the computation runs are handled afterwards in order, which meets the spec edge
  case for repeated Next presses.
- **Alternatives considered**:
  - Precomputing everything at startup: it would add about 4.1 s to startup and violate FR-014.
  - A background warm-up thread: this training is mostly Python-level loops that hold the GIL, so it would compete
    with the window for about 4 s right after start. `functools.cache` does not prevent duplicate concurrent work,
    and threads add complexity to code learners read. It could be added later, on top of the notice.
  - Faster training through vectorisation: `tinynet.train` is shown to learners and must stay simple.
  - A results cache on disk: extra files that can go stale.

## R11. Setup instructions

- **Decision**: The README states "Verified with Python 3.14 and 3.15.2 and the versions pinned in
  `requirements.txt`". It gains a "Troubleshooting" note covering the symptom (commands such as `.venv/bin/python`
  fail with "no such file or directory" or "bad interpreter" after a Python upgrade, for example through Homebrew),
  the cause (the environment points to a Python that no longer exists) and the fix (delete `.venv`, then repeat the
  three setup commands). It also gains a "Check" section (`python -m unittest`, about a minute, no window),
  `<name>_step_NN.png` in the export section, and `tests/` and `pages/` rows in the file table.
- **Rationale**: This is exactly what happened to the project's own `.venv` during the scan: its interpreter link
  points to a removed Homebrew Python 3.14.7. A fresh environment on 3.15.2 installed the pinned versions and ran all
  three programs.
- **Alternatives considered**: Pinning a Python version with a tool such as `.python-version` or pyenv: it adds a
  tool for beginners to learn, and the README note covers the common case.

## Deviations found during implementation (2026-11-22)

- **README setup command**: the first setup line is now `python3 -m venv .venv` instead of `python -m venv .venv`.
  Homebrew's Python on the maintainer's Mac provides only `python3`, so the old line failed with "command not found"
  before any environment existed. After `source .venv/bin/activate`, `python` works as before.
- **Troubleshooting wording**: it now lists the messages actually observed with a dangling interpreter link: "bad
  interpreter", "no such file or directory", "command not found: python", and `No module named 'matplotlib'` when
  `python3` falls back to the system Python.
- **`.gitignore`**: besides `__pycache__/` (R1), `.DS_Store` was added during the ignore-file verification step of
  `/speckit-implement`.
- **README code blocks** are now marked `sh`, which markdownlint asks for.
- **Check run time**: about 23 s, well under the expected minute (R1, contracts/cli.md).
- **The maintainer's `.venv`** was rebuilt in place (`python3 -m venv --clear .venv`) for T001, on Python 3.15.2.
- **Open**: tasks.md T027, checking by eye that the "Preparing" notice is painted in a real window before the
  computation starts. Without a window, `Tutor(interactive=True)` shows the notice after 0.02 s (R9 method).

