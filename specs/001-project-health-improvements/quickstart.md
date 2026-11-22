# Quickstart: Validating Project Health Improvements

This guide proves that the feature works end to end. Run everything from the repository root. Commands and outputs
are specified in [contracts/cli.md](contracts/cli.md); entity rules are in [data-model.md](data-model.md).

## 0. Prerequisites

A working environment on Python 3.14 or 3.15. The project's current `.venv` is broken: it points to a removed
Homebrew Python 3.14.7. Rebuild it, which also exercises scenario 6:

```sh
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Before changing any code, export both courses once into a reference folder. Scenarios 3 and 4 compare against it.

```sh
REF=$(mktemp -d)
python nn_tutor.py --save "$REF/nn" && python backprop_tutor.py --save "$REF/bp"
```

Until the feature is in place, both reference folders contain the old names `step_NN.png`.

## 1. The check passes (SC-001, SC-003, FR-007, FR-008)

```sh
time python -m unittest -v
git status --short
```

**Expected**: `OK`, exit code 0, no window opened, and a run time under 2 minutes (about 1 minute expected).
`git status --short` shows nothing new. Run it three times; all three runs must pass.

On a machine without a display (for example over SSH, or with `DISPLAY` unset on Linux) the result must be the
same.

## 2. Five faults are caught (SC-002)

Introduce one fault at a time, run `python -m unittest`, check the failure message, then undo it with
`git checkout -- <file>`.

| # | Fault | Where | Expected failure |
| --- | --- | --- | --- |
| a | Add `raise RuntimeError("boom")` as the first line of `page_start` | `backprop_tutor.py` | names `backprop_tutor`, step 1, "Where we are, and where we go" |
| b | Add a paragraph of about 400 words to the `t.write(...)` of step 1 | `nn_tutor.py` | names `nn_tutor`, step 1, and says the text overflows |
| c | Change `default_rng(7)` to `default_rng(8)` | `nn.py` | says the training data of `nn.py` and `tinynet.py` differ |
| d | In `Net.backprop`, change `D.mean(axis=0)` to `D.sum(axis=0)` | `tinynet.py` | names the network and a slope difference far above the tolerance |
| e1 | In `Net.backprop`, insert the comment line `# blame` above `e = p - y` | `tinynet.py` | **passes**; a fresh export of course 2 step 25 shows each label still beside its line |
| e2 | In `Net.backprop`, change `e = p - y` to `e = (p - y)` | `tinynet.py` | names `backprop_tutor`, step 25, and the label "step 1" whose line text was not found |

## 3. Exports do not collide, and leftovers go (SC-004, FR-009 to FR-011)

```sh
OUT=$(mktemp -d)
python nn_tutor.py --save "$OUT" && python backprop_tutor.py --save "$OUT"
ls "$OUT" | wc -l                          # 69
ls "$OUT" | grep -c '^nn_tutor_step_'      # 38
ls "$OUT" | grep -c '^backprop_tutor_step_'  # 31
touch "$OUT/nn_tutor_step_99.png" "$OUT/notes.txt"
python nn_tutor.py --save "$OUT"           # prints: removed leftover nn_tutor_step_99.png
ls "$OUT" | wc -l                          # 70: the 69 pages plus notes.txt
```

## 4. Pages look exactly as before (FR-018)

Course 1's new export must match the committed, renamed previews byte for byte. Both courses must match the
reference export from step 0. This assumes the same machine and library versions.

```sh
for f in pages/nn_tutor_step_*.png; do cmp -s "$f" "$OUT/$(basename "$f")" || echo "DIFF $f"; done
for n in $(seq -w 1 38); do cmp -s "$REF/nn/step_$n.png" "$OUT/nn_tutor_step_$n.png" || echo "DIFF nn $n"; done
for n in $(seq -w 1 31); do cmp -s "$REF/bp/step_$n.png" "$OUT/backprop_tutor_step_$n.png" || echo "DIFF bp $n"; done
```

**Expected**: no `DIFF` lines. The second and third loops cover course 2's re-anchored labels (steps 24, 25, 28) and
show that no "Preparing" notice leaks into exports.

## 5. Slow pages give immediate feedback (SC-005, SC-006, FR-013, FR-014)

This is checked by eye in a window.

1. `python nn_tutor.py --step 34`: the title and a "Preparing this page…" notice appear at once, and the page
   follows about 2 s later.
2. Restart the course and jump to steps 22, 32, 33 and 35 with the progress bar. Each shows the notice at once, then
   the page.
3. Revisit any of them: the page appears directly, with no notice.
4. On a fresh start, go to step 33 and press → five times quickly while the notice is visible. The course does not
   crash and ends on a later page, in order.
5. `python nn_tutor.py` and `python backprop_tutor.py`: the first page appears as quickly as before the change. Compare
   with `git stash` if unsure; the allowed difference is under 1 s.

## 6. The README alone is enough (SC-007, FR-015 to FR-017)

Have someone, or yourself, follow only the README in a fresh clone:

- They install, run `nn.py`, both courses and `python -m unittest`. The README names Python 3.14 and 3.15.2 as
  verified.
- They find the troubleshooting note for an environment broken by a Python upgrade and rebuild it. The broken
  `.venv` from step 0 is a real example of the symptom.
- They find both courses' preview pages in `pages/` through the README.
