# Contract: Course-building interface (`tutorkit.py`, `backprop_tutor.code_box`)

This is the interface that `nn_tutor.py`, `backprop_tutor.py` and the tests use. Only additions and changes are
listed; everything else in `tutorkit.py` stays as it is. Entity rules are in [data-model.md](../data-model.md).

## `Course`

```python
Course(title, glossary, sliders, doc, *, name)
```

- `name` (new, required, keyword-only): the course's file stem, `"nn_tutor"` or `"backprop_tutor"`.

```python
@course.step(title, prepare=None)
```

- `prepare` (new, optional): a zero-argument, cached function that does the page's slow computation, without
  matplotlib. Existing calls `step(title)` keep working.

```python
Course.export(out_dir) -> list[pathlib.Path]
```

- New. Creates `out_dir` if needed and renders every step with a non-interactive `Tutor`.
- Writes `<name>_step_<NN>.png` files, then deletes this course's leftover pages and prints one line per deleted
  file.
- Returns the written paths in step order. Prints nothing else; `run()` prints the summary line.

```python
Course.run()
```

- With `--save DIR`: calls `export(DIR)` and prints `saved <count> steps of <name> to <DIR>`.
- Without it: `Tutor(self, interactive=True)`, `show(step - 1)`, `plt.show()`.

## `Tutor`

```python
Tutor(course, interactive=False)
```

- `interactive` (new): true only in a window. When true, a first visit to a step with `prepare` shows the notice and
  builds the page through a single-shot timer. When false, `prepare()` runs directly before the page is built.

`Tutor.layout_problems: list[str]` (new)

- Reset by every `show()`. `write()` appends one message per text block that sticks out sideways and one if the text
  overflows, using the same wording it prints to stderr.

`Tutor.show(index)` (changed)

- Stops a pending build timer from an earlier `show()`.
- Runs `prepare` as described in [data-model.md](../data-model.md) (page preparation state).
- Otherwise behaves exactly as today.

`Tutor.build(index)` (new)

- The second stage of a deferred first visit, called by the single-shot timer.
- Removes the notice, calls the step's `prepare()`, adds `index` to `prepared` and builds the page.
- Tests call it directly, because Agg timers never fire.

`Tutor.prepared: set[int]` (new)

- The step indices whose `prepare()` has finished in this `Tutor`.

## `backprop_tutor.code_box`

```python
code_box(t, left, top, width, code, title=None, marks=None, fontsize=11.5)
```

- `marks` (changed): `{line text: label}`. Each key must occur in exactly one line of `code`; otherwise the function
  raises `ValueError(f"code label {label!r}: {count} lines contain {text!r}")`.
- The label is drawn exactly where the old line-number form drew it, so pages stay byte-identical.

To let `tests/test_code_labels.py` test anchors without drawing, the matching should live in a small helper, e.g.
`mark_lines(code, marks) -> {line number: label}`, which `code_box` uses.
