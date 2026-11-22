# Contract: Command-line behaviour

These commands are run from the repository root, with the project's environment active. Anything not mentioned here
keeps today's behaviour.

## Courses

```text
python nn_tutor.py       [--step N] [--save DIR]
python backprop_tutor.py [--step N] [--save DIR]
```

| Option | Behaviour | Change |
| --- | --- | --- |
| (none) | Opens the course window at step 1. | Startup is no more than 1 s slower than today (FR-014). |
| `--step N` | Opens at step N. Out-of-range values are clamped to the first or last step. | If step N has a slow computation, the "Preparing this page…" notice appears first (FR-013). |
| `--save DIR` | Renders every step to PNG without opening a window, creating DIR if needed. | See below. |

`--save DIR` details:

- **Files written**: `DIR/<name>_step_<NN>.png` for every step, where `<name>` is `nn_tutor` or `backprop_tutor` and
  `NN` is the step number in at least two digits. Example: `pages/backprop_tutor_step_07.png`.
- **Files never touched**: files of the other course, and any file that does not match `<name>_step_<digits>.png`.
- **Leftovers**: files `DIR/<name>_step_<NN>.png` with NN above the current step count are deleted after the export,
  and each is listed on stdout.
- **Stdout, last line**: `saved <count> steps of <name> to <DIR>`. Before it comes one line per deleted leftover:
  `removed leftover <file name>`.
- **Exit code**: 0 on success. Any exception propagates, giving a non-zero exit code and a traceback, as today.
- **Pages**: identical to a window at the default size, and never showing a "Preparing" notice (FR-018).

Exporting both courses into the same `DIR` yields 38 + 31 = 69 files (SC-004).

## Standalone demo

```text
python nn.py
```

Unchanged. The check runs it as a script, with output suppressed, to compare its data.

## Check

```text
python -m unittest          # summary only
python -m unittest -v       # one line per test
```

| Property | Contract |
| --- | --- |
| Display | Not needed. The tests select the Agg backend and never open a window (FR-007). |
| Duration | Under 2 minutes on the maintainer's laptop; about 1 minute expected (FR-007, SC-001). |
| Side effects | None in the repository. Exports go to a temporary folder that is deleted afterwards, and `git status` is unchanged after a run (FR-008). |
| Exit code | 0 if everything passes, 1 if anything fails (standard `unittest`). |
| Failure output | A page failure names the course name, step number and step title, e.g. `(course='backprop_tutor', step=25, title='The backward pass in code')`. Other failures name what differs (see [data-model.md](../data-model.md), "Check result"). |
| Scope | All 69 pages; the `nn.py` vs `tinynet.py` data; backprop vs nudging; code-label anchors; export naming and leftovers; preparation behaviour. |
