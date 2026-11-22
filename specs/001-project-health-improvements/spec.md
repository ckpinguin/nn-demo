# Feature Specification: Project Health Improvements

**Feature Branch**: `main` (no feature branch created; spec directory `specs/001-project-health-improvements`)

**Created**: 2026-11-22

**Status**: Implemented (2026-11-22); one check by eye in a window is still open, tasks.md T027

**Input**: User description: "Scan the current project and try to find improvements"

## User Scenarios & Testing *(mandatory)*

The project has two step-by-step courses for beginners: course 1, "Neural networks" (38 pages in 6 parts), and
course 2, "Backpropagation" (31 pages in 5 parts). It also has a standalone demo script and some shared code for the
data and the network. A scan on 2026-11-22 found that the courses look and work well. All 69 pages render without
errors or text overflow, and exporting course 1 again gives images identical to the committed ones. The improvements
below cover what was missing around the courses: a way to know they still work after an edit, safe exports, smooth
navigation and up-to-date setup instructions.

### User Story 1 - One check that catches broken pages and silent drift (Priority: P1)

The maintainer edits a course page, the shared network code or the standalone demo, then runs a single check before
committing. Within two minutes it says whether:

- every page of both courses still opens and its text fits the panel,
- the slopes the courses teach still agree with slopes measured by nudging,
- the standalone demo and the courses still use the same afternoons,
- the labelled code listings in course 2 still line up with the code they label.

When something is wrong, the check names the course and the step.

**Why this priority**: The project has about 4,100 lines and 69 interactive pages but no automated checks. Today the
only way to find a broken page is to click through all 69 pages by hand. Three things can also go wrong without any
warning:

- The standalone demo and the shared course code each generate the 200 afternoons themselves. This is on purpose, so
  the demo can be read on its own, but nothing confirms the two copies still match.
- Course 2 shows the real forward and backward pass with labels such as "step 1" and "steps 2 + 3" attached to fixed
  line positions. Inserting one line into the shared code moves the code under every label below it.
- Text-panel overflow is reported only as a console note, which is easy to miss.

The other stories are safer to change once this check exists.

**Independent Test**: Run the check on the unchanged project and confirm it passes. Then introduce each fault from
the scenarios below, one at a time, and confirm the check fails with a message that says what broke and where.

**Acceptance Scenarios**:

1. **Given** the unchanged project, **When** the maintainer runs the check, **Then** it reports success for all 38
   pages of course 1, all 31 pages of course 2 and every consistency check, without opening a window.
2. **Given** a page that raises an error when shown, **When** the check runs, **Then** it fails and names the course,
   step number and step title.
3. **Given** a paragraph too long for the text panel on some step, **When** the check runs, **Then** it fails and
   names that step, instead of only printing a console note.
4. **Given** the standalone demo's data generation was changed (for example a different random seed) and the shared
   course data was not, **When** the check runs, **Then** it fails and says that the two sets of afternoons differ.
5. **Given** a mistake in the backward pass of the shared network code, **When** the check runs, **Then** it fails
   because the computed slopes disagree with the slopes measured by nudging every knob.
6. **Given** a line inserted into the forward or backward pass of the shared network code, **When** the check runs,
   **Then** either the course 2 labels still sit on the code they describe, or the check fails and names the affected
   step.

---

### User Story 2 - Exporting both courses without one overwriting the other (Priority: P2)

A teacher or the maintainer follows the README to export both courses as images, for handouts, slides or previews
on the repository page. They get two complete, separate sets of pages, each clearly belonging to its course.

**Why this priority**: The README says to export both courses into a folder called `pages`, and both courses name
their files `step_01`, `step_02` and so on. Exporting course 2 after course 1 overwrites course 1's steps 1–31 and
leaves its steps 32–38 behind. The result is a folder that mixes two courses with nothing to tell them apart. The
committed `pages` folder holds only course 1, so anyone browsing the repository sees no preview of course 2.

**Independent Test**: Follow the README export instructions for both courses into an empty location, then count the
pages and check that each one can be traced to its course.

**Acceptance Scenarios**:

1. **Given** an empty location, **When** both courses are exported as the README describes, **Then** 38 pages of
   course 1 and 31 pages of course 2 exist and none was overwritten.
2. **Given** any exported page, **When** a person looks at its file name or folder, **Then** they can tell which
   course and which step it shows.
3. **Given** the published repository, **When** someone browses it without running anything, **Then** preview pages
   of both courses are available and match what the courses currently show.

---

### User Story 3 - No unexplained pauses when stepping through course 1 (Priority: P3)

A learner clicks Next through course 1. Every page appears promptly. If a page needs extra time to prepare, the
learner sees right away that something is happening.

**Why this priority**: Five pages of course 1 train networks or fit a model the first time they are opened. In this
scan their first visits took about 1.9 s ("Too flexible: overfitting", step 34), 1.1 s ("Different starts, different
results", step 33), 0.8 s ("How many hidden neurons?", step 32), 0.6 s ("The honest test", step 35) and 0.5 s ("The
learning rate", step 22). These times were measured without a window, counting one redraw per page as a window does.
Meanwhile the window gives no sign of activity. Later visits to these pages take at most 0.15 s, and every other page
in both courses takes at most 0.25 s. The courses are for beginners, and a frozen window looks like a crash to them.
This is P3 because each pause happens once per session and lasts at most about two seconds.

**Independent Test**: Start course 1, open step 22 and steps 32–35 for the first time, and time how long each page
(or a notice that it is being prepared) takes to appear.

**Acceptance Scenarios**:

1. **Given** a freshly started course 1, **When** the learner opens step 22 or any of steps 32 to 35 for the first
   time, **Then** the page, or a visible notice that it is being prepared, appears within half a second.
2. **Given** the learner has already visited one of those pages, **When** they return to it, **Then** it appears as
   quickly as any other page.
3. **Given** either course is launched, **When** the learner waits for the first page, **Then** it appears no more than
   1 second later than it does today.

---

### User Story 4 - Setup instructions that hold up after a system update (Priority: P3)

A beginner, or the maintainer months later, follows the README setup and gets all three programs running. If the
Python runtime was upgraded after the environment was created, the README tells them how to recognise the problem
and rebuild the environment.

**Why this priority**: The README says "Tested with Python 3.14". During this scan, the project's own environment no
longer started: the Python installation it pointed to had been removed by a system package upgrade. The pinned
library versions install cleanly on the current Python 3.15.2, and all three programs run correctly on it. The README
mentions neither 3.15 nor this failure, which commonly catches the beginners the courses are written for.

**Independent Test**: Follow the README on a machine with a stated runtime version. Separately, use an environment
whose runtime has been removed, and confirm the README leads to a working setup in both cases.

**Acceptance Scenarios**:

1. **Given** the README, **When** a reader looks for the supported runtime versions, **Then** it names the versions
   on which installation and all three programs were last verified, including 3.15.2.
2. **Given** an environment whose runtime was removed by a system update, **When** the reader follows the README's
   troubleshooting note, **Then** they can rebuild a working environment without other help.

---

### Edge Cases

- The check runs on a machine with no display, such as a remote server or an automated job. It must work without
  opening windows.
- Numbers can differ slightly between machines and library versions. Slope comparisons allow a small tolerance, and
  the check does not require exported images to be byte-identical.
- No page that fits today may be flagged as overflowing.
- A course gets shorter (for example from 38 to 36 steps) and is exported again into the same location. Leftover
  pages from the longer version must not pass for current ones: they are either removed or clearly flagged.
- The learner presses Next several times while a slow page is being prepared. The course must not crash, and the
  learner must land on the page they navigated to.

## Requirements *(mandatory)*

### Functional Requirements

#### Check (User Story 1)

- **FR-001**: The project MUST provide one command that checks both courses and the standalone demo and reports pass
  or fail.
- **FR-002**: The check MUST show every page of both courses and fail if any page raises an error, naming the course,
  step number and step title.
- **FR-003**: The check MUST fail when the text of any page does not fit its text panel, naming the step.
- **FR-004**: The check MUST fail if the afternoons used by the standalone demo differ from those used by the courses,
  for both the training and the test split.
- **FR-005**: The check MUST compare the slopes from the backward pass with slopes measured by nudging each knob, for
  at least one network with one input and one with two inputs. It MUST fail when they differ by more than a
  tolerance that still catches any single wrong term in the backward pass.
- **FR-006**: The labels on course 2's code listings MUST stay on the lines they describe when the shared network
  code changes. Otherwise the check MUST fail and name the affected step.
- **FR-007**: The check MUST run without a display and finish in under 2 minutes on a typical laptop.
- **FR-008**: The check MUST NOT change any file in the project. Anything it produces goes to a temporary location.

#### Export (User Story 2)

- **FR-009**: Every exported page MUST identify its course by file name or folder, so that exporting one course never
  overwrites pages of the other.
- **FR-010**: The README's export instructions MUST produce a separate, complete set of pages for each course.
- **FR-011**: After a course is re-exported, the location MUST hold exactly that course's current pages, or the
  export MUST list any leftover pages from an earlier, longer export.
- **FR-012**: The repository MUST include preview pages for both courses that match their current content, and the
  README MUST say where to find them.

#### Navigation (User Story 3)

- **FR-013**: In course 1, every page MUST appear within 0.5 s of the learner navigating to it, or show a visible
  notice within that time that it is being prepared.
- **FR-014**: Meeting FR-013 MUST NOT delay the first page of either course by more than 1 second compared to today.

#### Setup instructions (User Story 4)

- **FR-015**: The README MUST name the runtime versions on which setup and all three programs were last verified.
- **FR-016**: The README MUST explain how to recognise an environment whose runtime was removed or upgraded, and how
  to rebuild it.
- **FR-017**: The README MUST explain how to run the check from FR-001.

#### Unchanged behaviour

- **FR-018**: Every page MUST look and behave as it does today. The only allowed exceptions are the notice from FR-013
  and any change needed to meet FR-006.

### Key Entities

- **Course**: an ordered sequence of pages grouped into parts. Course 1 has 38 pages in 6 parts, course 2 has 31
  pages in 5 parts.
- **Page (step)**: one screen with plots, text and controls, identified by its course, step number and title.
- **Exported page set**: one image per page of a course, each traceable to its course and step.
- **Check report**: the result of the check, either pass or a list of failures. Each failure says what broke and, where
  it applies, which course and step.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The maintainer can confirm that both courses and the demo are intact with one command in under 2
  minutes, instead of clicking through 69 pages by hand.
- **SC-002**: All five fault types from User Story 1 are caught: a page error, a text overflow, changed demo data, a
  wrong backward-pass slope and a shifted code line. Each failure message names the course and step where one
  applies.
- **SC-003**: On the unchanged project, the check passes with zero failures on 3 consecutive runs.
- **SC-004**: Exporting both courses as the README describes gives 69 page images (38 + 31), none of them overwritten.
- **SC-005**: In course 1, the longest wait between navigating to a page and seeing either the page or a notice that
  it is being prepared is 0.5 s or less. Today it is about 1.9 s, at step 34.
- **SC-006**: Neither course takes more than 1 second longer to start than it does today.
- **SC-007**: A reader who follows only the README gets all three programs running on a stated runtime version. After
  a runtime upgrade, they can also rebuild a working environment without outside help.

## Assumptions

- The courses are for beginners, and the check is for the maintainer. No user accounts, network services or hosting
  are involved.
- The standalone demo keeps its own copy of the data generation so that it can be read alone. The check guards the
  duplication instead of removing it.
- Course content, wording, visuals and step order are out of scope.
- "Typical laptop" means a recent consumer laptop comparable to the one used for this scan, where exporting both
  courses took about 27 seconds in total.
- Committing course 2's preview pages (about 3.7 MB of images, next to course 1's 6.2 MB) is acceptable.
- The check is run locally. Running it automatically on the code-hosting service on every push is out of scope for
  this feature.
- Repairing the maintainer's own local environment is not part of this feature. The README guidance in FR-016 covers
  it.
- Licensing and packaging the project for distribution are out of scope.
