"""
The window behind the step-by-step courses nn_tutor.py and backprop_tutor.py.

Every step is a page: plots on the left, text on the right, Back / Next underneath and a
progress bar that shows the parts of the course. A course creates a Course, registers its
pages with @course.step("title") and calls course.run(). Import this module before pyplot.
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib

if "--save" in sys.argv:
    matplotlib.use("Agg")                # no window needed for exporting

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.text import Text
from matplotlib.widgets import AxesWidget, Button, RadioButtons, Slider

FS_TEXT, FS_NOTE = 11.5, 9.5
C_KNOB = "#7b3fa0"                       # sliders: the reader's own knobs


# --- Small helpers for plots ---
def m(value, digits=0):
    """A number formatted for use inside $...$, thousands separated by thin spaces."""
    if abs(value) < 0.5 * 10 ** -digits:
        value = 0.0                      # avoid printing "-0"
    return f"{value:,.{digits}f}".replace(",", r"\,")


def style(ax, title=None):
    if title:
        ax.set_title(title, loc="left", fontsize=11.5, color="#333")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def readout(ax, x=0.02, y=0.97, **kw):
    """An empty text box in the corner of a plot, to be filled by an update function."""
    style_ = dict(transform=ax.transAxes, va="top", fontsize=11, zorder=10,
                  bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bbb", alpha=0.95))
    return ax.text(x, y, "", **(style_ | kw))


def rings(ax, color="#d64545", s=300):
    return ax.scatter(np.empty(0), np.empty(0), s=s, facecolor="none", edgecolor=color, linewidth=2, zorder=5)


def set_points(collection, xs, ys):
    collection.set_offsets(np.column_stack([xs, ys]) if len(xs) else np.empty((0, 2)))


def node(ax, x, y, text="", color="#444", size=0.42, fc="white", fontsize=15, **kw):
    """A circle for a diagram, in data coordinates of an axes with equal aspect."""
    ax.add_patch(Circle((x, y), size, facecolor=fc, edgecolor=color, linewidth=2, zorder=3))
    return ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=4, **kw)


def box(ax, x, y, w, h, text="", color="#444", fc="white", fontsize=13, **kw):
    """A rounded rectangle centred at x, y for a diagram."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                facecolor=fc, edgecolor=color, linewidth=2, zorder=3))
    return ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=4, **kw)


def arrow(ax, start, end, color="#666", shrink=0, width=1.5, **kw):
    return ax.annotate("", end, start, arrowprops=dict(arrowstyle="-|>", color=color, linewidth=width, shrinkA=shrink,
                                                       shrinkB=shrink, mutation_scale=16) | kw)


def diagram_axes(t, rect, xlim, ylim):
    """Axes without ticks for drawing boxes and arrows, one data unit is equally long in x and y."""
    ax = t.axes(rect)
    ax.axis("off")
    ax.set(xlim=xlim, ylim=ylim, aspect="equal")
    return ax


# --- Text panel building blocks: plain str = paragraph ---
class H(str):
    """Heading."""


class M(str):
    """Centred formula."""


class Box(str):
    """Highlighted result."""


_TOKEN = re.compile(r"(?:[^\s$]*\$[^$]*\$)+[^\s$]*|\S+")   # words, never splitting $math$


def wrap(text, width):
    lines = []
    for raw in text.split("\n"):
        indent = "   " if raw.startswith("• ") else ""
        line, length = "", 0
        for token in _TOKEN.findall(raw):
            w = len(re.sub(r"\\[a-zA-Z]+|[{}^_$\\]", "", token))
            if line and length + 1 + w > width:
                lines.append(line)
                line, length = indent + token, len(indent) + w
            elif line:
                line, length = f"{line} {token}", length + 1 + w
            else:
                line, length = token, w
        lines.append(line)
    return "\n".join(lines)


# --- The course: a list of pages, grouped into parts ---
class Course:
    def __init__(self, title, glossary, sliders, doc):
        """glossary: [(pattern, symbol, meaning, formula, details)], explained under the text of every page
        that mentions the pattern. sliders: {key: (low, high, start, label, format[, step])} for the shared knobs."""
        self.title, self.glossary, self.slider_specs, self.doc = title, glossary, sliders, doc
        self.steps = []                  # (part, title, page)
        self.parts = []
        self.tutor = None

    def part(self, name):
        """All steps registered from now on belong to this part."""
        self.parts.append(name)

    def step(self, title):
        def register(page):
            self.steps.append((len(self.parts) - 1, title, page))
            return page
        return register

    def no(self, page):
        """The step number of a page, for references in the text."""
        return next(i for i, (_, _, p) in enumerate(self.steps) if p is page) + 1

    def glossary_in(self, text):
        """The glossary entries a text mentions, in order of first appearance."""
        found = [(match.start(), entry) for entry in self.glossary
                 if (match := re.search(rf"(?<![A-Za-z])(?:{entry[0]})(?![A-Za-z])", text))]
        return [entry for _, entry in sorted(found, key=lambda f: f[0])]

    def run(self):
        parser = argparse.ArgumentParser(description=self.doc, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("--save", metavar="DIR", help="save every step as a PNG into DIR instead of opening a window")
        parser.add_argument("--step", type=int, default=1, help="open the course at this step")
        args = parser.parse_args()

        self.tutor = Tutor(self)
        if args.save:
            out = Path(args.save)
            out.mkdir(parents=True, exist_ok=True)
            for i in range(len(self.steps)):
                self.tutor.show(i)
                self.tutor.fig.savefig(out / f"step_{i + 1:02d}.png")
            print(f"saved {len(self.steps)} steps to {out}")
        else:
            self.tutor.show(args.step - 1)
            plt.show()


class Tutor:
    def __init__(self, course):
        self.course = course
        steps = course.steps
        plt.rcParams.update({"toolbar": "None", "keymap.back": [], "keymap.forward": [], "keymap.home": [],
                             "keymap.save": [], "keymap.quit": [], "keymap.grid": [], "keymap.yscale": [],
                             "keymap.xscale": [], "keymap.pan": [], "keymap.zoom": [], "font.size": 10.5})
        self.fig = plt.figure(figsize=(15, 8.5), facecolor="white")
        self.fig.canvas.manager.set_window_title(course.title)
        self.title = self.fig.text(0.04, 0.945, "", fontsize=18, weight="bold", va="center")
        self.counter = self.fig.text(0.975, 0.945, "", fontsize=11, color="gray", ha="right", va="center")
        self.panel = self.fig.add_axes([0.645, 0.12, 0.33, 0.77])
        self.panel.axis("off")

        self.back = Button(self.fig.add_axes([0.645, 0.035, 0.07, 0.05]), "← Back")
        self.next = Button(self.fig.add_axes([0.905, 0.035, 0.07, 0.05]), "Next →")
        self.back.on_clicked(lambda _: self.go(self.pending - 1))
        self.next.on_clicked(lambda _: self.go(self.pending + 1))

        # progress bar: one segment per step, a gap between the parts
        self.bar_ax = self.fig.add_axes([0.725, 0.035, 0.17, 0.05])
        self.bar_ax.axis("off")
        part_of = np.array([part for part, _, _ in steps])
        self.bar_x = np.arange(len(steps)) + 0.8 * part_of
        self.bar_ax.set(xlim=(-0.6, self.bar_x[-1] + 0.6), ylim=(-1, 1.6))
        self.bar = self.bar_ax.bar(self.bar_x, 0.9, width=0.8, bottom=-0.45)
        self.part_label = self.bar_ax.text(0, 1.05, "", ha="center", va="center", fontsize=9, color="#555")

        self.popup = self.fig.text(0, 0, "", fontsize=10.5, linespacing=1.4, multialignment="left", zorder=100,
                                   visible=False, bbox=dict(boxstyle="round,pad=0.7", fc="#fffdf5", ec="#d9a400"))
        self.hoverable, self.hovered = None, None

        self.guess = {key: spec[2] for key, spec in course.slider_specs.items()}   # the reader's knobs, shared by all pages
        self.state = {}                  # anything else a page wants to keep when the reader comes back
        self.owned, self.timers, self.slider_for = [], [], {}
        self.index = self.pending = 0
        self.switch = self.fig.canvas.new_timer(interval=1)
        self.switch.single_shot = True
        self.switch.add_callback(lambda: self.show(self.pending))

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.fig.canvas.mpl_connect("resize_event", lambda _: self.show(self.index))  # re-flow text

    # navigation
    def go(self, index):
        """Show a step as soon as the current mouse or key event is handled. Matplotlib hands the event to all
        handlers connected when it arrived, also to those of widgets that show() removes, which then crash."""
        self.pending = int(np.clip(index, 0, len(self.course.steps) - 1))
        self.switch.start()

    def show(self, index):
        steps = self.course.steps
        self.index = self.pending = int(np.clip(index, 0, len(steps) - 1))
        for timer in self.timers:
            timer.stop()
        for item in self.owned:
            if isinstance(item, AxesWidget):
                item.disconnect_events()
                item.ax.remove()
            else:
                self.disconnect_handlers(item)
                item.remove()
        self.owned, self.timers, self.slider_for = [], [], {}
        self.hoverable, self.hovered = None, None    # collected again on the next mouse move
        self.popup.set_visible(False)
        self.panel.clear()
        self.panel.axis("off")

        part, title, page = steps[self.index]
        self.title.set_text(f"{self.index + 1}.  {title}")
        self.counter.set_text(f"part {part + 1} of {len(self.course.parts)}:  {self.course.parts[part]}"
                              f"     ·     step {self.index + 1} of {len(steps)}   (← → keys)")
        for i, (segment, (p, _, _)) in enumerate(zip(self.bar, steps)):
            segment.set_color("#333" if i == self.index else "#9bb7d4" if p == part else "#d5d5d5")
        in_part = self.bar_x[[i for i, (p, _, _) in enumerate(steps) if p == part]]
        self.part_label.set(text=f"part {part + 1}", x=in_part.mean())
        page(self)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key in ("right", "pagedown", " ", "enter"):
            self.go(self.pending + 1)
        elif event.key in ("left", "pageup", "backspace"):
            self.go(self.pending - 1)

    def on_click(self, event):
        if event.inaxes is self.bar_ax and event.xdata is not None:
            self.go(int(np.argmin(abs(self.bar_x - event.xdata))))

    def on_move(self, event):
        """Explain the abbreviations in the text under the mouse in a popup."""
        if self.hoverable is None:               # after drawing, so tick labels exist too
            self.hoverable = [(text, entries) for text in self.fig.findobj(Text)
                              if text is not self.popup and (entries := self.course.glossary_in(text.get_text()))]
        text, entries = next(((text, entries) for text, entries in self.hoverable if text.contains(event)[0]),
                             (None, None))
        if text is self.hovered:
            return
        self.hovered = text
        if text is not None:
            x, y = self.fig.transFigure.inverted().transform((event.x, event.y))
            right, top = x > 0.5, y > 0.5        # open towards the middle of the window
            popup = "\n\n".join(wrap(rf"$\mathbf{{{symbol}}}$ {meaning}" f"\n{formula}\n{details}", 56)
                                for _, symbol, meaning, formula, details in entries)
            self.popup.set(text=popup, x=x - 0.01 if right else x + 0.01, y=y - 0.02 if top else y + 0.02,
                           ha="right" if right else "left", va="top" if top else "bottom")
        self.popup.set_visible(text is not None)
        self.fig.canvas.draw_idle()

    def disconnect_handlers(self, artist):
        """3D axes connect mouse handlers to the canvas that would outlive artist.remove() and then crash."""
        callbacks = self.fig.canvas.callbacks
        for handlers in list(callbacks.callbacks.values()):
            for cid, ref in list(handlers.items()):
                if getattr(ref(), "__self__", None) is artist:
                    callbacks.disconnect(cid)

    # page building blocks
    def content(self, slider_rows=0):
        """Rectangle for a page's plot, leaving room for sliders underneath."""
        bottom = 0.09 + 0.045 * slider_rows
        return [0.06, bottom, 0.54, 0.88 - bottom]

    def split(self, slider_rows=0, gap=0.08, ratio=0.5):
        left, bottom, width, height = self.content(slider_rows)
        first = (width - gap) * ratio
        return [left, bottom, first, height], [left + first + gap, bottom, width - gap - first, height]

    def axes(self, rect, **kw):
        ax = self.fig.add_axes(rect, **kw)
        self.owned.append(ax)
        return ax

    def button(self, rect, label, on_click):
        button = Button(self.fig.add_axes(rect), label)

        def clicked(_):
            on_click()
            self.fig.canvas.draw_idle()

        button.on_clicked(clicked)
        self.owned.append(button)
        return button

    def buttons(self, actions, y=0.1, x=0.12, width=0.085, height=0.05, gap=0.01):
        """A row of buttons: [(label, function)]."""
        return [self.button([x + (width + gap) * k, y, width, height], label, action)
                for k, (label, action) in enumerate(actions)]

    def radio(self, rect, labels, on_change, active=0, title=None):
        ax = self.fig.add_axes(rect)
        ax.set_facecolor("#f7f7f7")
        for spine in ax.spines.values():
            spine.set_color("#ddd")
        if title:
            ax.set_title(title, fontsize=10.5, loc="left", color="#333")
        radio = RadioButtons(ax, labels, active=active, activecolor=C_KNOB)
        for label in radio.labels:
            label.set_fontsize(10.5)

        def clicked(label):
            on_change(labels.index(label))
            self.fig.canvas.draw_idle()

        radio.on_clicked(clicked)
        self.owned.append(radio)
        return radio

    def timer(self, interval, callback):
        """A repeating timer that stops when the reader leaves the page."""
        timer = self.fig.canvas.new_timer(interval=interval)

        def tick():
            callback()
            self.fig.canvas.draw_idle()

        timer.add_callback(tick)
        self.timers.append(timer)
        return timer

    def play_button(self, rect, advance, interval=40):
        """A button that calls advance() over and over until clicked again or the page changes."""
        timer = self.timer(interval, advance)
        running = [False]
        button = None

        def toggle():
            running[0] = not running[0]
            (timer.start if running[0] else timer.stop)()
            button.label.set_text("pause" if running[0] else "play ▸")

        button = self.button(rect, "play ▸", toggle)

        def stop():
            if running[0]:
                toggle()
        return stop

    def sliders(self, keys, on_change, columns=1):
        """Sliders for shared knobs below the plot, in one or two columns, filled row by row."""
        rows = -(-len(keys) // columns)
        for k, key in enumerate(keys):
            row, col = divmod(k, columns)
            lo, hi, _, label, fmt, *valstep = self.course.slider_specs[key]
            self.guess[key] = float(np.clip(self.guess[key], lo, hi))
            x, width = (0.12, 0.36) if columns == 1 else (0.1 + 0.29 * col, 0.17)
            ax = self.fig.add_axes([x, 0.03 + 0.045 * (rows - 1 - row), width, 0.028])
            slider = Slider(ax, label, lo, hi, valinit=self.guess[key], valfmt=fmt, color=C_KNOB,
                            valstep=valstep[0] if valstep else None)

            def changed(value, key=key):
                self.guess[key] = value
                on_change()
                self.fig.canvas.draw_idle()

            slider.on_changed(changed)
            self.owned.append(slider)
            self.slider_for[key] = slider
        on_change()

    def set_guess(self, **values):
        """Move knobs, through their sliders if the page has them."""
        for key, value in values.items():
            lo, hi, *_ = self.course.slider_specs[key]
            value = float(np.clip(value, lo, hi))
            if key in self.slider_for:
                self.slider_for[key].set_val(value)
            else:
                self.guess[key] = value

    def write(self, *blocks):
        """Lay out headings, paragraphs, formulas and boxes top to bottom in the right panel,
        with the abbreviations they use explained at the bottom."""
        renderer = self.fig.canvas.get_renderer()
        frame = self.panel.get_window_extent(renderer)
        px_per_pt = self.fig.dpi / 72
        width = max(30, int(frame.width / px_per_pt / (FS_TEXT * 0.53)))

        floor = 0.0                              # one line per abbreviation, stacked bottom up
        for _, symbol, meaning, *_ in reversed(self.course.glossary_in("\n".join(blocks))):
            note = self.panel.text(0, floor, wrap(rf"$\mathbf{{{symbol}}}$ {meaning}", int(width * FS_TEXT / FS_NOTE)),
                                   fontsize=FS_NOTE, color="#555", va="bottom", transform=self.panel.transAxes)
            floor += note.get_window_extent(renderer).height / frame.height + 0.006
        if floor:
            self.panel.plot([0, 1], [floor + 0.008] * 2, color="#ddd", linewidth=0.8, transform=self.panel.transAxes)
            floor += 0.016

        y = 1.0
        for block in blocks:
            pad = 0
            if isinstance(block, H):
                y -= 0.012
                kw = dict(x=0, s=block, fontsize=12.5, weight="bold", color="#1a1a1a")
            elif isinstance(block, M):
                kw = dict(x=0.5, s=block, fontsize=14.5, ha="center")
            elif isinstance(block, Box):
                pad = 0.6 * 13 * px_per_pt / frame.height
                kw = dict(x=0.5, s=block, fontsize=13, ha="center",
                          bbox=dict(boxstyle="round,pad=0.6", fc="#fff4cc", ec="#d9a400"))
            else:
                kw = dict(x=0, s=wrap(block, width), fontsize=FS_TEXT, linespacing=1.4, color="#222")
            text = self.panel.text(y=y - pad, va="top", transform=self.panel.transAxes, **kw)
            extent = text.get_window_extent(renderer)
            if extent.x1 > frame.x1 + 12 or extent.x0 < frame.x0 - 12:
                print(f"note: text panel of step {self.index + 1} is too narrow for: {block[:40]}", file=sys.stderr)
            y -= extent.height / frame.height + 2 * pad + 0.022
        if y < floor - 0.02:
            print(f"note: text panel of step {self.index + 1} overflows", file=sys.stderr)
