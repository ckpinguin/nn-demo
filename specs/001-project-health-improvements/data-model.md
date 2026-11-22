# Data Model: Project Health Improvements

This feature adds no database or stored records. Its "data" consists of the in-memory structures of `tutorkit.py`,
the files an export writes, and the results of the check. Fields marked **new** are added by this feature; all others
already exist. Function signatures are in [contracts/tutorkit-api.md](contracts/tutorkit-api.md).

## Course

An ordered set of pages grouped into parts. Created once per course module.

| Field | Type | Rules |
| --- | --- | --- |
| `title` | str | Window title, e.g. "Neural networks, step by step". |
| `name` **new** | str | Required. Must match `^[a-z][a-z0-9_]*$` and be unique among courses. Equals the module's file stem: `"nn_tutor"`, `"backprop_tutor"`. Used as the export file prefix. |
| `parts` | list[str] | Part names in order. Course 1 has 6, course 2 has 5. |
| `steps` | list[Step] | In registration order; the 1-based position is the step number. Course 1 has 38, course 2 has 31. |
| `glossary`, `slider_specs`, `doc` | unchanged | — |

## Step (page)

| Field | Type | Rules |
| --- | --- | --- |
| `part` | int | Index into `Course.parts`. |
| `title` | str | Shown as "N.  title". Identifies the page in check failures, together with the course name and step number. |
| `page` | callable(Tutor) | Builds the page. |
| `prepare` **new** | callable() or None | Optional. Must take no arguments, be cached (a second call is fast) and not touch matplotlib. Set on course 1 steps 22 (`eta_runs`), 32 (`hidden_runs`), 33 (`start_runs`), 34 (`overfit_run`) and 35 (`sklearn_net`). |

The steps are stored today as `(part, title, page)` tuples. Adding `prepare` changes them to 4-tuples or a small
named tuple. Every place that unpacks steps (`Course.no`, `Tutor.__init__`, `Tutor.show`) must be updated together.

## Page preparation state (per Tutor, per step with `prepare`)

```text
unprepared ──show() in a window──▶ preparing ──timer calls build(index)──▶ prepared
     ▲                                  │
     └── show() of another step before ─┘   (timer stopped, prepare() not started)

unprepared ──show() not in a window──▶ prepared  (prepare() runs directly, no notice)
```

| State | What the learner sees | Rules |
| --- | --- | --- |
| unprepared | — | Initial state. |
| preparing | Title, counter, progress bar and a "Preparing this page…" notice. | Entered only when `Tutor.interactive` is true. The notice is owned by the page and removed before the page is built. At most one build timer is pending at a time. |
| prepared | The page itself. | `prepare()` has returned. Later `show()` calls build the page directly, without a notice. |

Steps without `prepare` are always built directly, exactly as today.

## Layout problem **new**

A note that a page's text does not fit, recorded by `Tutor.write()` in `Tutor.layout_problems` (a list of str).

| Field | Type | Rules |
| --- | --- | --- |
| `message` | str | One of today's two stderr notes, unchanged in wording and thresholds: "too narrow" (a text block sticks out more than 12 px sideways) or "overflows" (the text runs into the glossary notes at the bottom). It is still printed to stderr as well. |
| `scope` | — | Collected in `Tutor.layout_problems`, which is reset at the start of every `show()`. After a page is built, the list holds only that page's problems. |

## Code label (course 2 code listings)

The `marks` argument of `backprop_tutor.code_box`, changed from `{line number: label}` to `{line text: label}`.

| Field | Type | Rules |
| --- | --- | --- |
| `line_text` (dict key) | str | Must occur in exactly one line of the listing. Zero or several matches raise `ValueError` naming the label and the text. |
| `label` (dict value) | str | Drawn in the gutter beside the matched line. |

The current anchors for steps 24, 25 and 28 are listed in [research.md](research.md) (R6).

## Exported page **new naming**

| Field | Rules |
| --- | --- |
| `file_name` | `<Course.name>_step_<NN>.png`, with NN the 1-based step number in at least two digits (`f"{n:02d}"`). |
| `location` | The folder given to `--save` (created if missing). Both courses may share one folder. |
| `leftover` | A file in that folder matching `^<Course.name>_step_(\d+)\.png$` whose number exceeds the course's step count. An export deletes leftovers after writing and prints each deleted name. Files of other courses and unrelated files are never touched. |
| `committed_previews` | `pages/nn_tutor_step_01.png` … `_38.png` (renamed from `pages/step_NN.png`) and `pages/backprop_tutor_step_01.png` … `_31.png` (new). |

## Check result

The outcome of `python -m unittest`: exit code 0 (all passed) or 1 (failures or errors).

| Test module | Covers | A failure names |
| --- | --- | --- |
| `tests/test_pages.py` | FR-002, FR-003 | course name, step number, step title, and the exception or layout problem |
| `tests/test_data.py` | FR-004 | which array differs (training or test data, inputs or votes) |
| `tests/test_slopes.py` | FR-005 | the network (inputs, hidden neurons, seed) and the largest difference |
| `tests/test_code_labels.py` | FR-006 | the label and its line text, for anchors matching 0 or more than 1 line |
| `tests/test_export.py` | FR-009, FR-011 | the missing, overwritten or leftover file |
| `tests/test_prepare.py` | FR-013, FR-018 | whether the notice appeared when it should not, or the reverse |
