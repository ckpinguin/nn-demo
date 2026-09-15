"""
Backpropagation, step by step.

The second course after nn_tutor.py: how a neural network finds the slopes of its loss for all
knobs in a single pass backwards. From the slope of a curve and the chain rule, over one neuron
and a small network worked out with numbers, to the few lines of numpy in tinynet.py and deep
networks, in very small steps.

    python backprop_tutor.py                 # Back / Next buttons or the arrow keys
    python backprop_tutor.py --step 10       # open the course at step 10
    python backprop_tutor.py --save pages    # export every step as a PNG instead
"""
import inspect
import re
import textwrap
from functools import cache

import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from office import (C_FIT, C_HID, C_MEAN, C_NET, C_NO, C_YES, GT, GX, HLIM, T, TG, X1, X2, XG, Y, map_axes, n,
                    plot_map_points, plot_votes, temp_axes, wash)
from tinynet import Net, sigmoid, train
from tutorkit import Box, Course, H, M, arrow, diagram_axes, m, node, readout, style

C_SLOPE = "#c62828"                      # slopes are red everywhere in this course
C_NOW = "#fff4cc"                        # what the current step computes

# --- The worked example: one afternoon, one neuron, one small network ---
EX = int(np.flatnonzero((Y == 1) & (T >= 24))[0])      # 24.0 °C, and the staff said: comfortable
X_EX, Y_EX = float(X1[EX, 0]), int(Y[EX])
W_EX, B_EX = 2.0, 1.0                                  # the knobs of the single neuron
NET_EX = Net([[2.0], [1.5]], [1.0, -1.0], [3.0, -2.0], -1.0)   # the knobs of the network with two hidden neurons
KNOB_NAMES = ["w1", "b1", "w2", "b2", "v1", "v2", "c"]


def penalty(p, y):
    return -np.log(p if y else 1 - p)


def neuron(w, b, x, y):
    """Every value and every slope of one neuron with its penalty, forward and backward."""
    wx = w * x
    z = wx + b
    p = sigmoid(z)
    L = penalty(p, y)
    dp = -1 / p if y else 1 / (1 - p)
    dz = dp * p * (1 - p)
    return dict(x=x, w=w, wx=wx, b=b, z=z, p=p, L=L, dL=1.0, dp=dp, dz=dz, dwx=dz, db=dz, dw=dz * x, dx=dz * w)


def network(net, x, y):
    """Every value and every slope of the network with two hidden neurons for one afternoon."""
    (w1, w2), (b1, b2), (v1, v2), c = net.W[:, 0], net.b, net.v, net.c
    z1, z2 = w1 * x + b1, w2 * x + b2
    h1, h2 = sigmoid(z1), sigmoid(z2)
    z = v1 * h1 + v2 * h2 + c
    p = sigmoid(z)
    d = p - y                                           # step 1: blame of the output score
    dh1, dh2 = d * v1, d * v2                           # step 2: back along the output weights
    d1, d2 = dh1 * h1 * (1 - h1), dh2 * h2 * (1 - h2)   # step 3: through the hidden switches
    return dict(x=x, y=y, z1=z1, z2=z2, h1=h1, h2=h2, z=z, p=p, L=penalty(p, y), dp=-1 / p if y else 1 / (1 - p),
                d=d, dv1=d * h1, dv2=d * h2, dc=d, dh1=dh1, dh2=dh2, d1=d1, d2=d2,
                dw1=d1 * x, db1=d1, dw2=d2 * x, db2=d2, dx=d1 * w1 + d2 * w2, **dict(zip(KNOB_NAMES, knobs_of(net))))


def knobs_of(net):
    return [net.W[0, 0], net.b[0], net.W[1, 0], net.b[1], net.v[0], net.v[1], net.c]


def net_with(knobs):
    w1, b1, w2, b2, v1, v2, c = knobs
    return Net([[w1], [w2]], [b1, b2], [v1, v2], c)


def nudge(f, value, h=1e-5):
    """The slope of f at value, by nudging a tiny bit up and down."""
    return (f(value + h) - f(value - h)) / (2 * h)


def mm(value, digits=3):
    """A signed number for mathtext."""
    return m(value, digits)


SLIDERS = {     # key: (low, high, start, label, format[, step])
    "u": (-2, 3, 1.5, "$u$", "%.2f"),
    "h": (0.01, 1.5, 1.0, "nudge $h$", "%.2f"),
    "a": (-3, 3, 1.5, "input $a$", "%.2f"),
    "b": (-3, 3, 0.7, "input $b$", "%.2f"),
    "wt": (-3, 3, 2.0, "input $w$", "%.2f"),
    "z": (-6, 6, 1.0, "score $z$", "%.2f"),
    "p": (0.03, 0.97, 0.7, "chance $p$", "%.2f"),
    "uc": (-2, 2, 0.3, "input $u$", "%.2f"),
    "w": (-4, 4, W_EX, "weight $w$", "%.2f"),
    "bn": (-4, 4, B_EX, "bias $b$", "%.2f"),
    "depth": (2, 20, 10, "layers", "%d", 1),
    "wd": (0.5, 5, 1.0, "weight $w$", "%.2f"),
}

GLOSSARY = [    # (pattern, symbol, meaning, formula, details)
    (r"\\partial|∂", r"\partial", "slope: ∂L/∂w is how much L changes per nudge of w, everything else fixed",
     r"$\partial L / \partial w \approx (L(w + h) - L(w)) / h$  for a tiny $h$",
     "Read '∂L/∂w' as 'the slope of L for w'. The curly d reminds us that only w is nudged, all other "
     "knobs stay where they are. It is a name, not a fraction you could cancel."),
    (r"\\delta|δ", r"\delta", "blame of a neuron: the slope of the loss for the neuron's score z",
     r"$\delta = \partial L / \partial z$,   output neuron: $\delta = p - y$",
     "Positive: the score is too high, lowering it would lower the loss. Negative: too low. Every knob "
     "of the neuron gets its slope from this one number."),
    (r"\\sigma|σ", r"\sigma", "sigmoid, squashes any score into a number between 0 and 1",
     r"$\sigma(z) = 1 / (1 + e^{-z})$,   slope $\sigma(z)\,(1 - \sigma(z))$",
     "The slope is largest, 0.25, at z = 0, and almost 0 far out, where the neuron is very sure."),
    (r"\\ln", r"\ln", "natural logarithm, undoes e to the power of",
     r"$\ln(e^{x}) = x$,   slope of $\ln p$ is $1 / p$",
     "The penalty −ln p has the slope −1/p: steep when p is close to 0."),
    (r"e\^", "e", "Euler's number, about 2.718",
     r"$e^{-z} = 1 / e^{z}$", "e to the power of a big number is huge, to the power of a very negative number tiny."),
    (r"[Ll]oss", "L", "loss, the average penalty for the chances given to what really happened",
     r"$L = \frac{1}{n} \sum (-\ln q_i)$", "q is p for comfortable afternoons and 1 − p for the others."),
    (r"[Gg]radient", "gradient", "the list of the slopes of the loss, one for every knob",
     r"$(\partial L / \partial w_1,\ \partial L / \partial b_1,\ \dots)$",
     "Gradient descent moves every knob a little against its slope. Backpropagation computes the gradient."),
    (r"[Cc]hain rule", r"chain\ rule", "slopes along a chain of calculations multiply",
     r"$\partial c / \partial a = \partial c / \partial b \cdot \partial b / \partial a$",
     "If b reacts twice as strongly as a, and c three times as strongly as b, then c reacts six times as "
     "strongly as a."),
    (r"\\sum", r"\sum", "sum: add up the term for every afternoon",
     r"$\sum a_i = a_1 + a_2 + \dots + a_n$", "Divided by n, the sum becomes the average."),
    (r"@", "@", "matrix product in numpy: every row on the left with every row of weights, multiplied and added",
     r"$(X\,W^{T})_{ij} = x_{i1} w_{j1} + x_{i2} w_{j2}$",
     "One @ computes the weighted sums of all neurons of a layer for all afternoons at once."),
    (r"ReLU", "ReLU", "an activation function: negative scores become 0, positive ones pass",
     r"$\mathrm{ReLU}(z) = \max(0,\ z)$,   slope 0 or 1",
     "Because its slope is 1 wherever the neuron is active, blame passes through without shrinking."),
]

course = Course("Backpropagation, step by step", GLOSSARY, SLIDERS, __doc__)
step, no = course.step, course.no


# --- Drawing computations as boxes and wires ---
class Graph:
    """A computation as boxes and wires: the value on every wire in black, its slope in red."""

    def __init__(self, ax):
        self.ax, self.items, self.boxes = ax, {}, {}

    def box(self, key, x, y, label, width=1.3, height=1.0, fontsize=16, color=C_NET):
        patch = FancyBboxPatch((x - width / 2, y - height / 2), width, height, zorder=3, facecolor="white",
                               boxstyle="round,pad=0.02,rounding_size=0.12", edgecolor=color, linewidth=2)
        self.ax.add_patch(patch)
        self.ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, zorder=4)
        self.boxes[key] = patch

    def wire(self, key, start, end, value, slope, offset=0.42):
        """A horizontal wire with its value above and its slope below."""
        arrow(self.ax, start, end, color="#777", width=1.5)
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        self.label(key, (mx, my + offset), value, (mx, my - offset), slope)

    def label(self, key, value_at, value, slope_at, slope, fontsize=12, ha="center"):
        """A value and a slope somewhere in the drawing, shown and hidden together under one key."""
        white = dict(boxstyle="round,pad=0.15", fc="white", ec="none")
        v = self.ax.text(*value_at, value, ha=ha, va="center", fontsize=fontsize, color="#222", zorder=6, bbox=white)
        s = self.ax.text(*slope_at, slope, ha=ha, va="center", fontsize=fontsize, color=C_SLOPE, zorder=6, bbox=white)
        self.items.setdefault(key, []).append((v, s))

    def show(self, values=None, slopes=(), current=(), boxes=()):
        """values: keys whose values are shown (None = all), slopes: keys whose slopes are shown, current: highlighted."""
        for key, pairs in self.items.items():
            for v, s in pairs:
                v.set_visible(values is None or key in values)
                s.set_visible(key in slopes)
                now = key in current
                s.set_bbox(dict(boxstyle="round,pad=0.2", fc=C_NOW if now else "white", ec="#d9a400" if now else "none"))
                s.set_fontweight("bold" if now else "normal")
        for key, patch in self.boxes.items():
            patch.set_facecolor(C_NOW if key in boxes else "white")


def graph_axes(t, rect, xlim, ylim):
    ax = t.axes(rect)
    ax.axis("off")
    ax.set(xlim=xlim, ylim=ylim)
    return ax


def legend(ax, x, y):
    ax.text(x, y, "black: value", fontsize=10.5, color="#222", va="center")
    ax.text(x + 2.3, y, "red: slope of the loss for it", fontsize=10.5, color=C_SLOPE, va="center")


def neuron_axes(t, rect):
    return graph_axes(t, rect, (0, 16.2), (-0.2, 6.2))


def neuron_graph(ax, v):
    """One neuron and its penalty: x times w, plus b, sigmoid, -ln."""
    g, y0 = Graph(ax), 3.0
    g.box("times", 3.4, y0, r"$\times$")
    g.box("plus", 6.6, y0, "$+$")
    g.box("sigma", 9.8, y0, r"$\sigma$")
    g.box("ln", 13.1, y0, r"$-\ln$", width=1.5)
    g.wire("w", (0.6, y0), (2.75, y0), rf"$w = {mm(v['w'], 2)}$", rf"${mm(v['dw'])}$")
    arrow(ax, (3.4, 5.6), (3.4, y0 + 0.5), color="#777")
    g.label("x", (3.2, 4.75), rf"$x = {mm(v['x'], 2)}$", (3.6, 4.2), rf"${mm(v['dx'])}$", ha="right")
    for pair in g.items["x"]:
        pair[1].set_ha("left")
    g.wire("wx", (4.05, y0), (5.95, y0), rf"$wx = {mm(v['wx'], 2)}$", rf"${mm(v['dwx'])}$")
    arrow(ax, (6.6, 0.3), (6.6, y0 - 0.5), color="#777")
    g.label("b", (6.4, 1.45), rf"$b = {mm(v['b'], 2)}$", (6.8, 0.9), rf"${mm(v['db'])}$", ha="right")
    for pair in g.items["b"]:
        pair[1].set_ha("left")
    g.wire("z", (7.25, y0), (9.15, y0), rf"$z = {mm(v['z'], 2)}$", rf"${mm(v['dz'])}$")
    g.wire("p", (10.45, y0), (12.35, y0), rf"$p = {mm(v['p'])}$", rf"${mm(v['dp'])}$")
    g.wire("L", (13.85, y0), (15.9, y0), rf"$L = {mm(v['L'])}$", "$1$")
    legend(ax, 0.2, 5.9)
    return g


def network_axes(t, rect):
    return graph_axes(t, rect, (0, 18.6), (-0.4, 10.2))


def network_graph(ax, v):
    """The network with two hidden neurons and the penalty, as boxes: weigh and add (Σ), then switch (σ)."""
    g = Graph(ax)
    g.box("x", 0.9, 5.0, "$x$", width=1.0, color=C_MEAN)
    g.label("x", (0.9, 4.1), rf"${mm(v['x'], 2)}$", (0.9, 3.55), rf"${mm(v['dx'])}$", fontsize=11)
    for k, (yy, color) in enumerate(((8.0, C_HID[0]), (2.0, C_HID[1])), start=1):
        side = 1 if yy > 5 else -1
        g.box(f"sum{k}", 4.2, yy, r"$\Sigma$", color=color)
        g.box(f"sig{k}", 7.4, yy, r"$\sigma$", color=color)
        arrow(ax, (1.4, 5.0 + 0.35 * side), (3.55, yy - 0.4 * side), color="#777")
        g.wire(f"z{k}", (4.85, yy), (6.75, yy), rf"$z_{k} = {mm(v[f'z{k}'], 2)}$", rf"${mm(v[f'd{k}'])}$")
        arrow(ax, (8.05, yy - 0.3 * side), (10.35, 5.0 + 0.45 * side), color="#777")
        g.label(f"h{k}", (9.9, 5.0 + 2.1 * side), rf"$h_{k} = {mm(v[f'h{k}'])}$",
                (8.55, 5.0 + 1.3 * side), rf"${mm(v[f'dh{k}'])}$")
        for name, xx in ((f"w{k}", 3.4), (f"b{k}", 5.1)):
            g.label(f"knobs{k}", (xx, yy + side * 0.95), rf"${name[0]}_{k} = {v[name]:g}$".replace("-", "−"),
                    (xx, yy + side * 1.5), rf"${mm(v['d' + name])}$", fontsize=11)
    g.box("sum", 11.0, 5.0, r"$\Sigma$", height=1.3)
    g.box("sig", 13.7, 5.0, r"$\sigma$")
    g.box("ln", 16.2, 5.0, r"$-\ln$", width=1.4)
    g.wire("z", (11.65, 5.0), (13.05, 5.0), rf"$z = {mm(v['z'], 2)}$", rf"${mm(v['d'])}$", offset=0.95)
    g.wire("p", (14.35, 5.0), (15.5, 5.0), rf"$p = {mm(v['p'])}$", rf"${mm(v['dp'])}$", offset=0.8)
    g.wire("L", (16.9, 5.0), (18.0, 5.0), rf"$L = {mm(v['L'])}$", "$1$", offset=0.8)
    for name, symbol, xx in (("v1", "v_1", 12.5), ("v2", "v_2", 14.2), ("c", "c", 15.9)):
        g.label("knobs", (xx, 3.4), rf"${symbol} = {v[name]:g}$".replace("-", "−"), (xx, 2.85), rf"${mm(v['d' + name])}$",
                fontsize=11)
    legend(ax, 0.2, 10.0)
    return g


NET_STAGES = [                          # slopes known after each step of the backward pass
    ["L", "p", "z", "knobs"],
    ["h1", "h2"],
    ["z1", "z2"],
    ["knobs1", "knobs2"],
    ["x"],
]


def network_page(t, stage, lines, extra=None):
    """A step of the backward pass through the network: the graph, highlighted, and the calculation below it."""
    left, bottom, width, height = t.content()
    ax = network_axes(t, [left, bottom + 0.3 * height, width, 0.7 * height])
    v = network(NET_EX, X_EX, Y_EX)
    g = network_graph(ax, v)
    known = [key for keys in NET_STAGES[:stage] for key in keys]
    boxes = [["sum", "sig", "ln"], ["sum"], ["sig1", "sig2"], ["sum1", "sum2"], ["x"]][stage - 1]
    g.show(slopes=known, current=NET_STAGES[stage - 1], boxes=boxes)
    text_ax = t.axes([left, bottom, width, 0.28 * height])
    text_ax.axis("off")
    numbers = {key: mm(val) for key, val in v.items()} | (extra or {})
    for k, line in enumerate(lines):
        text_ax.text(0, 0.92 - k * 0.24, line.format(**numbers), fontsize=12.5, va="center", transform=text_ax.transAxes)
    return v


def code_of(function):
    """The source of a function as it is in the file, without its docstring."""
    source = textwrap.dedent(inspect.getsource(function)).strip()
    return re.sub(r'\n\s*"""[^"]*"""', "", source, count=1)


def code_box(t, left, top, width, code, title=None, marks=None, fontsize=11.5):
    """Source code in a grey box below top, as high as it needs to be. marks: {line number: short note} in a
    gutter on the left. Returns the figure y of the box's lower edge."""
    lines = code.split("\n")
    gutter = 1.15 if marks else 0.12                            # inches
    width_in = width * 15 - gutter - 0.15
    fontsize = min(fontsize, width_in * 72 / (0.61 * max(len(line) for line in lines)))
    line_h = fontsize * 1.6 / 72 / 8.5                          # figure height per line
    height = line_h * len(lines) + 0.025
    ax = t.axes([left, top - height, width, height])
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left", fontsize=11.5, color="#333")
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#f3f5f8", edgecolor="#d5dbe3"))
    x = gutter / (width * 15)
    for k, line in enumerate(lines):
        y = 1 - (0.0125 + (k + 0.5) * line_h) / height
        ax.text(x, y, line, family="monospace", fontsize=fontsize, va="center", color="#1a1a1a", transform=ax.transAxes)
        if marks and k + 1 in marks:
            ax.text(x - 0.08 / (width * 15), y, marks[k + 1], ha="right", va="center", fontsize=10, color=C_SLOPE,
                    weight="bold", transform=ax.transAxes)
    return top - height


# === The steps ===

course.part("Slopes")


@step("Where we are, and where we go")
def page_start(t):
    ax = network_axes(t, t.content())
    v = network(NET_EX, X_EX, Y_EX)
    g = network_graph(ax, v)
    g.show(values=[], slopes=[])
    ax.annotate("", (1.5, -0.1), (16.2, -0.1), arrowprops=dict(arrowstyle="-|>", color=C_SLOPE, linewidth=3,
                                                                 mutation_scale=25))
    ax.text(8.8, 0.35, "backpropagation: the slopes flow backwards", ha="center", fontsize=13, color=C_SLOPE,
            weight="bold")
    ax.annotate("", (16.2, 9.5), (1.5, 9.5), arrowprops=dict(arrowstyle="-|>", color="#555", linewidth=3,
                                                               mutation_scale=25))
    ax.text(8.8, 9.05, "forward pass: the values flow forwards", ha="center", fontsize=13, color="#555", weight="bold")
    t.write(
        "In nn_tutor.py a network learned by gradient descent: find the slope of the loss for every knob, "
        "then move every knob a little against its slope. Repeat.",
        "Finding the slopes by nudging costs one extra loss calculation per knob. Backpropagation finds all "
        "of them in one single pass backwards through the network. This course shows how.",
        H("What you need"),
        "• what a neuron, the loss and gradient descent are (parts 2 to 4 of nn_tutor.py),",
        "• +, −, × and ÷. Everything else is built up here, in small steps.",
        H("The plan"),
        "1. slopes of simple building blocks,  2. one neuron backwards,  3. a network backwards,  "
        "4. all afternoons at once, in code,  5. deep networks.",
        Box("left: the network of nn_tutor.py,\ndrawn as a chain of small calculations"),
    )


@step("A slope is a ratio of two nudges")
def page_slope(t):
    ax = style(t.axes(t.content(slider_rows=2)), title=r"The curve $f(u) = u^2$")
    grid = np.linspace(-2.2, 3.2, 300)
    ax.plot(grid, grid ** 2, color="#555", linewidth=2.5)
    ax.set(xlim=(-2.2, 3.2), ylim=(-1, 10), xlabel="$u$", ylabel="$f(u)$")
    secant, = ax.plot([], [], color=C_SLOPE, linewidth=2)
    step_x, = ax.plot([], [], color=C_NET, linewidth=2, linestyle="--")
    step_y, = ax.plot([], [], color=C_NET, linewidth=2, linestyle="--")
    dots, = ax.plot([], [], "o", color=C_NET, markersize=10, zorder=5)
    info = readout(ax, fontsize=12.5, family="monospace")

    def update():
        u, h = t.guess["u"], t.guess["h"]
        f0, f1 = u ** 2, (u + h) ** 2
        ratio = (f1 - f0) / h
        span = np.array([-2.2, 3.2])
        secant.set_data(span, f0 + ratio * (span - u))
        step_x.set_data([u, u + h], [f0, f0])
        step_y.set_data([u + h, u + h], [f0, f1])
        dots.set_data([u, u + h], [f0, f1])
        info.set_text(f"f(u)     = {f0:6.3f}\nf(u + h) = {f1:6.3f}\n\n"
                      f"ratio = ({f1:.3f} - {f0:.3f}) / {h:.2f}\n      = {ratio:.3f}\n\n"
                      f"slope 2u = {2 * u:.3f}")

    t.sliders(["u", "h"], update)
    t.write(
        "Before networks, one idea: the slope. Take a curve, here $f(u) = u^2$, and pick a point $u$.",
        "Nudge $u$ by a small amount $h$ (dashed, to the right) and see how much $f$ changes (dashed, up). "
        "Divide the two:",
        M(r"ratio $= \dfrac{f(u + h) - f(u)}{h}$"),
        "That's the steepness of the red line through both points.",
        H("Try it"),
        "Make $h$ smaller and smaller. The ratio settles at one number: the slope of the curve at $u$. "
        "For $u^2$ it is always $2u$.",
        "The slope says: nudge the input by a tiny bit, and the output moves by slope × that bit.",
        Box("slope = change of the output ÷ change of the input,\nfor a tiny nudge"),
    )


def formula_axes(t, rect):
    ax = t.axes(rect)
    ax.axis("off")
    return ax


@step("Slopes of a plus box")
def page_plus(t):
    left, bottom, width, height = t.content(slider_rows=2)
    ax = diagram_axes(t, [left, bottom + 0.45 * height, width, 0.55 * height], (0, 12), (0, 5))
    Graph(ax).box("plus", 6, 2.5, "$+$", width=1.6, height=1.4, fontsize=22)
    arrow(ax, (1.5, 4.0), (5.15, 2.9), width=2)
    arrow(ax, (1.5, 1.0), (5.15, 2.1), width=2)
    arrow(ax, (6.85, 2.5), (10, 2.5), width=2)
    a_text = ax.text(1.2, 4.2, "", fontsize=15, ha="right", va="center")
    b_text = ax.text(1.2, 0.8, "", fontsize=15, ha="right", va="center")
    s_text = ax.text(10.2, 2.5, "", fontsize=15, ha="left", va="center")
    lines_ax = formula_axes(t, [left, bottom, width, 0.42 * height])
    lines = [lines_ax.text(0.02, 0.9 - k * 0.22, "", fontsize=12.5, family="monospace", va="center",
                           transform=lines_ax.transAxes) for k in range(4)]

    def update():
        a, b = t.guess["a"], t.guess["b"]
        s = a + b
        a_text.set_text(f"$a = {mm(a, 2)}$")
        b_text.set_text(f"$b = {mm(b, 2)}$")
        s_text.set_text(f"$s = {mm(s, 2)}$")
        lines[0].set_text(f"                  s = {a:5.2f} + {b:5.2f} = {s:5.2f}")
        lines[1].set_text(f"nudge a by 0.01:  s = {a + 0.01:5.2f} + {b:5.2f} = {s + 0.01:5.2f}   change 0.01, slope 1")
        lines[2].set_text(f"nudge b by 0.01:  s = {a:5.2f} + {b + 0.01:5.2f} = {s + 0.01:5.2f}   change 0.01, slope 1")

    t.sliders(["a", "b"], update)
    t.write(
        "Every calculation in a network is made of a few simple boxes. Their slopes are easy. First the "
        "plus box: $s = a + b$.",
        "Nudge $a$ by 0.01: $s$ grows by exactly 0.01. The same for $b$. Move the sliders: it doesn't matter "
        "what $a$ and $b$ are.",
        "A plus box passes every nudge on unchanged. Its slope is 1 for both inputs.",
        H("A short way to write slopes"),
        "∂s/∂a is short for: the slope of $s$ when only $a$ is nudged. Read it 'the slope of s for a'. "
        "The curly ∂ reminds us that everything else stays fixed.",
        Box(r"$s = a + b$:     $\partial s / \partial a = 1$,    $\partial s / \partial b = 1$"),
    )


@step("Slopes of a times box")
def page_times(t):
    left, bottom, width, height = t.content(slider_rows=2)
    ax = diagram_axes(t, [left, bottom + 0.45 * height, width, 0.55 * height], (0, 12), (0, 5))
    Graph(ax).box("times", 6, 2.5, r"$\times$", width=1.6, height=1.4, fontsize=22)
    arrow(ax, (1.5, 4.0), (5.15, 2.9), width=2)
    arrow(ax, (1.5, 1.0), (5.15, 2.1), width=2)
    arrow(ax, (6.85, 2.5), (10, 2.5), width=2)
    a_text = ax.text(1.2, 4.2, "", fontsize=15, ha="right", va="center")
    w_text = ax.text(1.2, 0.8, "", fontsize=15, ha="right", va="center")
    s_text = ax.text(10.2, 2.5, "", fontsize=15, ha="left", va="center")
    lines_ax = formula_axes(t, [left, bottom, width, 0.42 * height])
    lines = [lines_ax.text(0.02, 0.9 - k * 0.22, "", fontsize=12.5, family="monospace", va="center",
                           transform=lines_ax.transAxes) for k in range(3)]

    def update():
        a, w = t.guess["a"], t.guess["wt"]
        s = w * a
        a_text.set_text(f"$a = {mm(a, 2)}$")
        w_text.set_text(f"$w = {mm(w, 2)}$")
        s_text.set_text(f"$s = {mm(s, 3)}$")
        lines[0].set_text(f"                  s = {w:5.2f} * {a:5.2f} = {s:7.4f}")
        lines[1].set_text(f"nudge a by 0.01:  s = {w:5.2f} * {a + 0.01:5.2f} = {w * (a + 0.01):7.4f}"
                          f"   change {w * 0.01:+.4f}, slope w")
        lines[2].set_text(f"nudge w by 0.01:  s = {w + 0.01:5.2f} * {a:5.2f} = {(w + 0.01) * a:7.4f}"
                          f"   change {a * 0.01:+.4f}, slope a")

    t.sliders(["a", "wt"], update)
    t.write(
        r"Now the times box: $s = w \cdot a$.",
        "Nudge $a$ by 0.01, and $s$ changes by $w$ times 0.01: the slope for $a$ is $w$. Nudge $w$ instead, and "
        "$s$ changes by $a$ times 0.01: the slope for $w$ is $a$.",
        "A times box multiplies every nudge by its other input.",
        Box(r"$s = w \cdot a$:     $\partial s / \partial a = w$,    $\partial s / \partial w = a$"),
        H("Why this one matters"),
        "Every weight of a network sits in a times box, multiplied with the input of its neuron. So the slope "
        "for a weight will always contain the input it multiplies.",
        "If that input is 0, nudging the weight changes nothing, and its slope is 0.",
    )


@step("The slope of the sigmoid")
def page_sigmoid_slope(t):
    left, bottom, width, height = t.content(slider_rows=1)
    ax = style(t.axes([left, bottom + 0.52 * height, width, 0.46 * height]), title=r"$\sigma(z)$ and its tangent")
    grid = np.linspace(-6, 6, 400)
    ax.plot(grid, sigmoid(grid), color="#555", linewidth=2.5)
    ax.set(xlim=(-6, 6), ylim=(-0.1, 1.1))
    ax.set_xticklabels([])
    tangent, = ax.plot([], [], color=C_SLOPE, linewidth=2)
    dot, = ax.plot([], [], "o", color=C_NET, markersize=10, zorder=5)
    ax2 = style(t.axes([left, bottom, width, 0.42 * height]), title=r"its slope $\sigma(z)\,(1 - \sigma(z))$")
    ax2.plot(grid, sigmoid(grid) * (1 - sigmoid(grid)), color=C_SLOPE, linewidth=2.5)
    ax2.set(xlim=(-6, 6), ylim=(0, 0.3), xlabel="score $z$")
    dot2, = ax2.plot([], [], "o", color=C_SLOPE, markersize=10, zorder=5)
    info = readout(ax, fontsize=12, family="monospace")

    def update():
        z = t.guess["z"]
        p = sigmoid(z)
        slope = p * (1 - p)
        near = np.array([z - 1.5, z + 1.5])
        tangent.set_data(near, p + slope * (near - z))
        dot.set_data([z], [p])
        dot2.set_data([z], [slope])
        info.set_text(f"σ(z) = {p:.4f}\nnudged:  (σ(z + 0.001) - σ(z)) / 0.001 = {(sigmoid(z + 0.001) - p) / 0.001:.4f}\n"
                      f"formula: σ(z) · (1 - σ(z))            = {slope:.4f}")

    t.sliders(["z"], update)
    t.write(
        "The switch box of a neuron is the sigmoid σ. Its slope has a beautiful formula:",
        M(r"$\dfrac{\partial \sigma}{\partial z} = \sigma(z)\,(1 - \sigma(z))$"),
        "You don't have to derive it. Move the slider and compare with the nudge: they agree.",
        "So once the forward pass has computed $p = \\sigma(z)$, the slope is just $p\\,(1 - p)$. No new "
        "calculation of σ needed.",
        H("Remember the shape"),
        "• The slope is largest at $z = 0$, and there it is 0.25.",
        "• Far out, where the neuron is very sure, the slope is almost 0: nudges hardly get through. That "
        f"will matter in step {no(page_vanish)}.",
    )


@step("The slope of the penalty")
def page_ln_slope(t):
    ax = style(t.axes(t.content(slider_rows=1)), title="The penalty for one afternoon")
    grid = np.linspace(0.005, 0.995, 400)
    ax.plot(grid, -np.log(grid), color=C_YES, linewidth=2.5, label=r"staff said comfortable: $-\ln p$")
    ax.plot(grid, -np.log(1 - grid), color=C_NO, linewidth=2.5, label=r"staff said not comfortable: $-\ln(1 - p)$")
    ax.set(xlim=(0, 1), ylim=(0, 4.5), xlabel="chance $p$ the network gives to 'comfortable'", ylabel="penalty")
    ax.legend(loc="upper center", frameon=False, fontsize=11)
    tangents = [ax.plot([], [], color=C_SLOPE, linewidth=2)[0] for _ in range(2)]
    dots = [ax.plot([], [], "o", color=color, markersize=10, markeredgecolor="white", zorder=5)[0]
            for color in (C_YES, C_NO)]
    info = readout(ax, y=0.78, fontsize=12, family="monospace")

    def update():
        p = t.guess["p"]
        near = np.array([p - 0.12, p + 0.12])
        for tangent, dot, value, slope in ((tangents[0], dots[0], -np.log(p), -1 / p),
                                           (tangents[1], dots[1], -np.log(1 - p), 1 / (1 - p))):
            tangent.set_data(near, value + slope * (near - p))
            dot.set_data([p], [value])
        info.set_text(f"comfortable:     slope -1/p     = {-1 / p:+.3f}\n"
                      f"not comfortable: slope 1/(1-p)  = {1 / (1 - p):+.3f}")

    t.sliders(["p"], update)
    t.write(
        "The last box of the chain is the penalty. Its slope, for the chance $p$:",
        M(r"$\dfrac{\partial(-\ln p)}{\partial p} = -\dfrac{1}{p}$      "
          r"$\dfrac{\partial(-\ln(1 - p))}{\partial p} = \dfrac{1}{1 - p}$"),
        "• Comfortable afternoon (blue): the slope is negative. A higher $p$ lowers the penalty.",
        "• Not comfortable (orange): the slope is positive. A higher $p$ raises the penalty.",
        "• The more wrong the network is, the steeper: at $p = 0.03$ for a comfortable afternoon the slope "
        "is −33.",
        "Now we know the slopes of all building blocks: plus, times, σ and the penalty. Next: how to combine "
        "them.",
    )


@step("Boxes in a row: slopes multiply")
def page_chain(t):
    left, bottom, width, height = t.content(slider_rows=1)
    ax = diagram_axes(t, [left, bottom + 0.5 * height, width, 0.5 * height], (0, 13), (0, 4.4))
    g = Graph(ax)
    g.box("times", 3.3, 2.2, r"$\times 3$", width=1.6)
    g.box("sigma", 7.8, 2.2, r"$\sigma$", width=1.6)
    arrow(ax, (0.9, 2.2), (2.5, 2.2), width=2)
    arrow(ax, (4.1, 2.2), (7.0, 2.2), width=2)
    arrow(ax, (8.6, 2.2), (11.2, 2.2), width=2)
    texts = [ax.text(xx, 3.1, "", ha="center", fontsize=13) for xx in (0.8, 5.55, 10.2)]
    slopes = [ax.text(xx, 1.0, "", ha="center", fontsize=12, color=C_SLOPE) for xx in (3.3, 7.8)]
    bars_ax = style(t.axes([left, bottom, width, 0.42 * height]), title="Size of a nudge of 0.01 on its way")
    bars = bars_ax.bar([0, 1, 2], [0, 0, 0], color=[C_MEAN, C_NET, C_SLOPE], width=0.55)
    bars_ax.set_xticks([0, 1, 2], ["nudge of $u$", "change of $v$", "change of $s$"], fontsize=11)
    bars_ax.set_ylim(0, 0.036)
    bar_labels = [bars_ax.text(k, 0, "", ha="center", va="bottom", fontsize=11) for k in range(3)]

    def update():
        u = t.guess["uc"]
        v = 3 * u
        s = sigmoid(v)
        local = s * (1 - s)
        texts[0].set_text(f"$u = {mm(u, 2)}$")
        texts[1].set_text(f"$v = {mm(v, 2)}$")
        texts[2].set_text(f"$s = {mm(s, 3)}$")
        slopes[0].set_text(r"slope $3$")
        slopes[1].set_text(rf"slope $s(1-s) = {local:.3f}$")
        changes = [0.01, 0.03, sigmoid(3 * (u + 0.01)) - s]
        for bar, label, size, k in zip(bars, bar_labels, changes, range(3)):
            bar.set_height(abs(size))
            label.set(text=f"{size:+.4f}", y=abs(size) + 0.0008)
        bars_ax.set_title(rf"A nudge of 0.01 on its way:  total slope $= 3 \times {local:.3f} = {3 * local:.3f}$",
                          loc="left", fontsize=11.5, color="#333")

    t.sliders(["uc"], update)
    t.write(
        "Put two boxes in a row: first $v = 3u$, then $s = \\sigma(v)$. How much does $s$ react to a nudge "
        "of $u$?",
        "Follow the nudge (bars): the times box makes it 3 times bigger. The σ box then multiplies it by its "
        "own slope $s\\,(1 - s)$.",
        "So the total slope is the product of the slopes of the two boxes. Like gears: if the first gear "
        "turns the second 3 times as fast, and the second turns the third 0.2 times as fast, then the third "
        "turns 3 × 0.2 = 0.6 times as fast as the first.",
        Box("chain rule: along a chain of boxes,\nthe slopes multiply"),
        "Move $u$: when $v$ is far from 0, the σ box is flat and swallows the nudge.",
    )


@step("The chain rule, written down")
def page_chain_rule(t):
    left, bottom, width, height = t.content()
    ax = neuron_axes(t, [left, bottom + 0.42 * height, width, 0.58 * height])
    v = neuron(W_EX, B_EX, X_EX, Y_EX)
    g = neuron_graph(ax, v)
    g.show(slopes=[])
    for xx, text in ((3.4, rf"for $w$: $x = {mm(v['x'], 2)}$"), (6.6, "1"),
                     (9.8, rf"$p(1-p) = {mm(v['p'] * (1 - v['p']))}$"), (13.1, rf"$-1/p = {mm(v['dp'])}$")):
        ax.text(xx, 1.95, text, ha="center", va="center", fontsize=11, color=C_SLOPE, zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_SLOPE, linewidth=0.8))
    lines_ax = formula_axes(t, [left, bottom, width, 0.38 * height])
    product = v["x"] * 1 * v["p"] * (1 - v["p"]) * v["dp"]
    by_nudge = nudge(lambda w: penalty(sigmoid(w * X_EX + B_EX), Y_EX), W_EX)
    lines_ax.text(0, 0.85, r"$\dfrac{\partial L}{\partial w} = \dfrac{\partial (wx)}{\partial w} \cdot "
                  r"\dfrac{\partial z}{\partial (wx)} \cdot \dfrac{\partial p}{\partial z} \cdot "
                  r"\dfrac{\partial L}{\partial p}$", fontsize=16, va="center", transform=lines_ax.transAxes)
    lines_ax.text(0, 0.5, rf"$= {mm(v['x'], 2)} \cdot 1 \cdot {mm(v['p'] * (1 - v['p']))} \cdot ({mm(v['dp'])}) = "
                  rf"{mm(product, 4)}$", fontsize=15, va="center", transform=lines_ax.transAxes)
    lines_ax.text(0, 0.18, f"by nudging w:  {by_nudge:+.4f}", fontsize=13, family="monospace", va="center",
                  transform=lines_ax.transAxes)
    t.write(
        "This chain is a whole neuron with its penalty, for one afternoon: "
        f"{T[EX]:.1f} °C, so $x = {X_EX:.1f}$, and the staff said comfortable. The knobs are $w = {W_EX:.0f}$ "
        f"and $b = {B_EX:.0f}$.",
        "Nudge the weight $w$. The nudge passes four boxes on its way to $L$, and each multiplies it by its "
        "slope (red frames under the boxes). So:",
        Box("chain rule: the slope of the end for the start\nis the product of all slopes in between"),
        "The formula (left) looks scary, but it only lists the four box slopes. Multiplied, they give the "
        "same number as nudging $w$.",
        "That is all the maths backpropagation needs. The rest is clever bookkeeping, and that's what the "
        "next part is about.",
    )


course.part("One neuron, backwards")


@step("The neuron as a chain of boxes")
def page_neuron_graph(t):
    left, bottom, width, height = t.content()
    ax = neuron_axes(t, [left, bottom + 0.3 * height, width, 0.7 * height])
    v = neuron(W_EX, B_EX, X_EX, Y_EX)
    neuron_graph(ax, v).show(slopes=[])
    t.write(
        "From now on we draw every calculation as boxes and wires. Values travel along the wires. Here is "
        "one neuron with its penalty:",
        "• the input $x$ and the weight $w$ go into a times box,",
        "• the bias $b$ is added, which gives the score $z$,",
        "• σ turns the score into the chance $p$,",
        "• the penalty box computes $L = -\\ln p$, because the staff said comfortable.",
        f"The numbers are the example of the last step: {T[EX]:.1f} °C, $w = {W_EX:.0f}$, $b = {B_EX:.0f}$.",
        "Two passes will run through this chain: first forwards for the values, then backwards for the "
        "slopes.",
    )


@step("Forward pass: values flow to the right")
def page_forward(t):
    stage = t.state.setdefault("forward", 1)
    left, bottom, width, height = t.content(slider_rows=1)
    ax = neuron_axes(t, [left, bottom + 0.3 * height, width, 0.7 * height])
    v = neuron(W_EX, B_EX, X_EX, Y_EX)
    g = neuron_graph(ax, v)
    line_ax = formula_axes(t, [left, bottom, width, 0.25 * height])
    line = line_ax.text(0, 0.6, "", fontsize=15, va="center", transform=line_ax.transAxes)
    steps = [("x", "w"), ("wx",), ("b", "z"), ("p",), ("L",)]
    formulas = ["the inputs: $x = {x}$ and the weight $w = {w}$",
                r"times box:  $w \cdot x = {w} \cdot {x} = {wx}$",
                "plus box:  $z = {wx} + {b} = {z}$",
                r"σ box:  $p = \sigma({z}) = {p}$",
                r"penalty box:  $L = -\ln {p} = {L}$"]

    def update():
        k = t.state["forward"]
        shown = [key for keys in steps[:k] for key in keys]
        g.show(values=shown, slopes=[])
        numbers = {key: mm(val, 3) for key, val in v.items()}
        line.set_text(formulas[k - 1].format(**numbers) if k else "")

    def advance(delta):
        t.state["forward"] = int(np.clip(t.state["forward"] + delta, 1, len(steps)))
        update()

    update()
    t.buttons([("next box", lambda: advance(1)), ("start over", lambda: advance(-9))], y=0.035, x=0.08, width=0.1)
    t.write(
        "The forward pass computes the values from left to right. Click 'next box' and follow them.",
        "Nothing new: it's exactly what the neuron did in nn_tutor.py. One detail is new, though:",
        H("Remember everything"),
        "Keep every value on every wire: $wx$, $z$, $p$. The backward pass will need them.",
        "For example, the slope of the σ box is $p\\,(1 - p)$, and the slope for $w$ in the times box is $x$. "
        "Both are already written on the wires after the forward pass.",
    )


@step("Backward pass: slopes flow to the left")
def page_backward(t):
    t.state.setdefault("backward", 1)
    left, bottom, width, height = t.content(slider_rows=1)
    ax = neuron_axes(t, [left, bottom + 0.3 * height, width, 0.7 * height])
    v = neuron(W_EX, B_EX, X_EX, Y_EX)
    g = neuron_graph(ax, v)
    line_ax = formula_axes(t, [left, bottom, width, 0.25 * height])
    line = line_ax.text(0, 0.6, "", fontsize=15, va="center", transform=line_ax.transAxes)
    steps = [("L",), ("p",), ("z",), ("b", "wx"), ("w", "x")]
    formulas = [r"start at the end:  $\partial L / \partial L = 1$",
                r"penalty box:  $1 \cdot (-1/p) = -1 / {p} = {dp}$",
                r"σ box:  ${dp} \cdot p\,(1-p) = {dp} \cdot {p} \cdot (1 - {p}) = {dz}$",
                r"plus box:  slope 1 for both inputs:  ${dz} \cdot 1 = {dz}$",
                r"times box:  $\partial L/\partial w = {dz} \cdot x = {dw}$,   $\partial L/\partial x = {dz} \cdot w = {dx}$"]

    def update():
        k = t.state["backward"]
        g.show(slopes=[key for keys in steps[:k] for key in keys], current=steps[k - 1])
        line.set_text(formulas[k - 1].format(**{key: mm(val) for key, val in v.items()}))

    def advance(delta):
        t.state["backward"] = int(np.clip(t.state["backward"] + delta, 1, len(steps)))
        update()

    update()
    t.buttons([("next box", lambda: advance(1)), ("start over", lambda: advance(-9))], y=0.035, x=0.08, width=0.1)
    t.write(
        "Now the slopes, from right to left (red, under the wires). Each red number is the slope of the loss "
        "for the value on that wire. Click 'next box'.",
        "• At the very end, $L$ reacts to a nudge of $L$ one to one: slope 1.",
        "• Every box takes the slope arriving from the right and multiplies it by its own slope. That gives "
        "the slope for its input.",
        Box("slope for the input =\nslope arriving × slope of the box"),
        "That's the chain rule, done one box at a time. The last click delivers the slopes of the two knobs "
        "$w$ and $b$.",
        "Each box needs just one multiplication, with numbers from the forward pass.",
    )


@step("A shortcut: blame = chance minus vote")
def page_shortcut(t):
    ax = style(t.axes(t.content(slider_rows=1)), title="Slope of the penalty for the score $z$")
    grid = np.linspace(-6, 6, 400)
    ax.plot(grid, sigmoid(grid) - 1, color=C_YES, linewidth=3, label=r"staff said comfortable ($y = 1$):  $p - 1$")
    ax.plot(grid, sigmoid(grid), color=C_NO, linewidth=3, label=r"staff said not comfortable ($y = 0$):  $p - 0$")
    ax.axhline(0, color="#333", linewidth=1)
    ax.set(xlim=(-6, 6), ylim=(-1.1, 1.1), xlabel="score $z$", ylabel=r"$\delta = \partial L / \partial z$")
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    dots = [ax.plot([], [], "o", color=color, markersize=11, markeredgecolor="white", zorder=5)[0]
            for color in (C_YES, C_NO)]
    info = readout(ax, x=0.98, y=0.82, ha="right", fontsize=12, family="monospace")

    def update():
        z = t.guess["z"]
        p = sigmoid(z)
        dots[0].set_data([z], [p - 1])
        dots[1].set_data([z], [p])
        info.set_text(f"p = σ({z:.2f}) = {p:.3f}\n\n"
                      f"y = 1: (-1/p) · p(1-p)    = {(-1 / p) * p * (1 - p):+.3f} = p - 1\n"
                      f"y = 0: (1/(1-p)) · p(1-p) = {(1 / (1 - p)) * p * (1 - p):+.3f} = p - 0")

    t.sliders(["z"], update)
    t.write(
        "The σ box and the penalty box always come together. Multiply their slopes, and almost everything "
        "cancels:",
        M(r"$y = 1$:   $-\dfrac{1}{p} \cdot p\,(1 - p) = p - 1$"),
        M(r"$y = 0$:   $\dfrac{1}{1 - p} \cdot p\,(1 - p) = p$"),
        "Both cases in one formula: the slope for the score is the network's chance minus the real vote.",
        Box(r"$\delta = p - y$"),
        "We call it δ (delta), the blame of the neuron. Positive: the score is too high. Negative: too low. "
        "The more wrong the chance, the bigger the blame.",
    )


@step("The knobs get their slopes: blame × input")
def page_knob_slopes(t):
    left, right = t.split(slider_rows=2, gap=0.07, ratio=0.55)
    ax = temp_axes(t, left, title="The neuron with your knobs", top=False)
    plot_votes(ax, alpha=0.5, label=False)
    curve, = ax.plot(TG, np.zeros_like(TG), color=C_NET, linewidth=3, zorder=5)
    mark, = ax.plot([], [], "o", color=C_SLOPE, markersize=16, markerfacecolor="none", markeredgewidth=2.5, zorder=6)
    table_ax = t.axes(right)
    table_ax.axis("off")
    table = table_ax.text(0, 0.98, "", fontsize=12.5, va="top", family="monospace", linespacing=1.8)

    def update():
        w, b, i = t.guess["w"], t.guess["bn"], t.state.get("afternoon", EX)
        curve.set_ydata(sigmoid(w * XG[:, 0] + b))
        v = neuron(w, b, X1[i, 0], Y[i])
        mark.set_data([T[i]], [Y[i]])
        table.set_text(f"afternoon {i + 1}: {T[i]:.1f} °C\n"
                       f"x = {v['x']:+.3f}   y = {Y[i]}\n\n"
                       f"p = σ({v['z']:+.2f}) = {v['p']:.3f}\n"
                       f"δ = p - y   = {v['dz']:+.3f}\n\n"
                       f"∂L/∂w = δ · x = {v['dw']:+.4f}\n"
                       f"∂L/∂b = δ     = {v['db']:+.4f}")

    def next_afternoon():
        t.state["afternoon"] = (t.state.get("afternoon", EX) + 9) % n
        update()

    t.sliders(["w", "bn"], update)
    t.button([0.52, 0.035, 0.08, 0.075], "next\nafternoon", next_afternoon)
    t.write(
        "With the blame δ, the rest of the neuron is quick:",
        "• The plus box passes the blame on unchanged to the bias:",
        M(r"$\partial L / \partial b = \delta$"),
        "• The times box multiplies it by the other input, $x$, for the weight:",
        M(r"$\partial L / \partial w = \delta \cdot x$"),
        "Blame times input. An afternoon at 22 °C has $x = 0$, so it gives the weight no slope at all: the "
        "weight had no effect on that afternoon.",
        H("Try it"),
        "Click 'next afternoon' and move the knobs. The red ring marks the afternoon in the table.",
    )


@step("Check: nudge the knobs")
def page_check_neuron(t):
    ax = t.axes(t.content())
    ax.axis("off")
    w, b = t.guess["w"], t.guess["bn"]
    rows = ["  °C   y     p      δ      ∂L/∂w backprop  nudging    ∂L/∂b backprop  nudging"]
    for i in np.linspace(0, n - 1, 12).astype(int):
        v = neuron(w, b, X1[i, 0], Y[i])
        dw = nudge(lambda k: penalty(sigmoid(k * X1[i, 0] + b), Y[i]), w)
        db = nudge(lambda k: penalty(sigmoid(w * X1[i, 0] + k), Y[i]), b)
        rows.append(f"{T[i]:5.1f}  {Y[i]}  {v['p']:.3f}  {v['dz']:+.3f}     {v['dw']:+.5f}     {dw:+.5f}"
                    f"         {v['db']:+.5f}     {db:+.5f}")
    ax.set_title(f"Twelve afternoons, knobs w = {w:.2f} and b = {b:.2f} (from the sliders of the last step)",
                 loc="left", fontsize=11.5, color="#333")
    ax.text(0, 0.95, "\n".join(rows).replace("-", "−"), family="monospace", fontsize=11, va="top", linespacing=2.0,
            bbox=dict(boxstyle="round,pad=0.6", fc="#f7f7f7", ec="#ddd"))
    t.write(
        "Is it right? Compare with the slow way: nudge each knob by a tiny bit up and down, compute the "
        "penalty twice, and divide the difference by the size of the nudge.",
        "The table shows twelve afternoons. Backpropagation and nudging agree to many decimals.",
        H("One neuron, backwards, in short"),
        "1. forward: compute and keep $z$ and $p$,",
        "2. blame of the score: $\\delta = p - y$,",
        "3. weight: $\\delta \\cdot x$,  bias: $\\delta$.",
        "Two multiplications for the whole backward pass. Nudging needs two extra forward passes per knob.",
    )


course.part("A network, backwards")


@step("The network as boxes")
def page_net_graph(t):
    left, bottom, width, height = t.content()
    ax = network_axes(t, [left, bottom + 0.3 * height, width, 0.7 * height])
    v = network(NET_EX, X_EX, Y_EX)
    network_graph(ax, v).show(slopes=[])
    text_ax = formula_axes(t, [left, bottom, width, 0.28 * height])
    for k, line in enumerate([rf"hidden: $h_1 = \sigma({mm(v['w1'], 1)} \cdot {mm(X_EX, 1)} + {mm(v['b1'], 1)}) = "
                              rf"\sigma({mm(v['z1'], 2)}) = {mm(v['h1'])}$,    $h_2 = \sigma({mm(v['z2'], 2)}) = {mm(v['h2'])}$",
                              rf"output: $p = \sigma({mm(v['v1'], 1)} \cdot {mm(v['h1'])} {mm(v['v2'], 1)} \cdot "
                              rf"{mm(v['h2'])} {mm(v['c'], 1)}) = \sigma({mm(v['z'], 2)}) = {mm(v['p'])}$",
                              rf"penalty: $L = -\ln {mm(v['p'])} = {mm(v['L'])}$"]):
        text_ax.text(0, 0.9 - k * 0.3, line, fontsize=13, va="center", transform=text_ax.transAxes)
    t.write(
        "Now the network from nn_tutor.py with two hidden neurons. Each neuron is drawn as two boxes: Σ "
        "weighs and adds its inputs, σ switches.",
        f"Same afternoon as before ({T[EX]:.1f} °C, comfortable), and seven knobs as if halfway through "
        "training:",
        M(r"$w_1 = 2,\ b_1 = 1,\ w_2 = 1.5,\ b_2 = -1,$"),
        M(r"$v_1 = 3,\ v_2 = -2,\ c = -1$"),
        "The forward pass has filled in every wire (calculation below).",
        H("The plan for the backward pass"),
        "Four steps, from the right to the left:",
        "1. the output neuron,  2. back along its weights,  3. through the hidden switches,  4. the hidden knobs.",
    )


@step("Backward step 1: the output neuron")
def page_back_output(t):
    network_page(t, 1, [r"blame of the output score:  $\delta = p - y = {p} - 1 = {d}$",
                        r"output knobs, blame × input:  $\partial L/\partial v_1 = \delta \cdot h_1 = {d} \cdot {h1} = {dv1}$,"
                        r"  $\partial L/\partial v_2 = \delta \cdot h_2 = {dv2}$",
                        r"output bias:  $\partial L/\partial c = \delta = {dc}$"])
    t.write(
        "The output neuron is a neuron like the one of the last part. Its inputs are just $h_1$ and $h_2$ "
        "instead of $x$.",
        "So everything from the last part applies (highlighted):",
        "• its blame is $\\delta = p - y$,",
        "• each of its weights gets blame × input: $\\delta \\cdot h_1$ and $\\delta \\cdot h_2$,",
        "• its bias gets the blame itself.",
        "The chance $p = 0.68$ is too low for a comfortable afternoon, so the blame is negative: the score "
        "should go up. Both slopes for $v_1$ and $v_2$ are negative, too: gradient descent will increase "
        "both weights.",
        "Three of seven slopes are done.",
    )


@step("Backward step 2: blame for the hidden answers")
def page_back_answers(t):
    network_page(t, 2, [r"times box $v_1 \cdot h_1$: the slope for $h_1$ is the other input, $v_1$",
                        r"$\partial L/\partial h_1 = \delta \cdot v_1 = {d} \cdot {v1} = {dh1}$",
                        r"$\partial L/\partial h_2 = \delta \cdot v_2 = {d} \cdot ({v2}) = {dh2}$"])
    t.write(
        "Inside the output neuron's Σ box, $h_1$ is multiplied by $v_1$. A times box sends a nudge on "
        "multiplied by its other input. So:",
        M(r"$\partial L / \partial h_1 = \delta \cdot v_1$      $\partial L / \partial h_2 = \delta \cdot v_2$"),
        "The blame travels back along the same arrows the values came forward, multiplied by the weights.",
        H("What it means"),
        "• $v_1 = 3$ is positive: a bigger $h_1$ would raise $p$. Since $p$ is too low, the slope for $h_1$ is "
        "negative: $h_1$ should go up.",
        "• $v_2 = -2$ is negative: a bigger $h_2$ would lower $p$. So $h_2$ should go down: positive slope.",
        "• A hidden neuron with a big weight gets a big share of the blame.",
    )


@step("Backward step 3: through the hidden switches")
def page_back_switches(t):
    network_page(t, 3, [r"σ box: the slope is $h\,(1 - h)$, with $h$ from the forward pass",
                        r"$\delta_1 = \partial L/\partial h_1 \cdot h_1 (1 - h_1) = {dh1} \cdot {h1} \cdot (1 - {h1}) = {d1}$",
                        r"$\delta_2 = \partial L/\partial h_2 \cdot h_2 (1 - h_2) = {dh2} \cdot {h2} \cdot (1 - {h2}) = {d2}$"])
    t.write(
        "Next come the σ boxes of the hidden neurons. Each multiplies the arriving slope by its own slope, "
        "$h\\,(1 - h)$:",
        M(r"$\delta_1 = \dfrac{\partial L}{\partial h_1} \cdot h_1 (1 - h_1)$"),
        "The result is the blame of the hidden neuron's score, δ₁ and δ₂, exactly like δ of the output neuron.",
        "Look at the sizes: $h_1 = 0.88$ is quite sure, so its σ box is flat, with slope 0.105. It shrinks the "
        "blame to a tenth. $h_2 = 0.44$ is near the middle, its slope 0.246 lets more through.",
        "A neuron that is very sure learns slowly.",
    )


@step("Backward step 4: the hidden knobs")
def page_back_hidden_knobs(t):
    network_page(t, 4, [r"hidden knobs, blame × input, and the input is $x = {x}$:",
                        r"$\partial L/\partial w_1 = \delta_1 \cdot x = {d1} \cdot {x} = {dw1}$,    "
                        r"$\partial L/\partial b_1 = \delta_1 = {db1}$",
                        r"$\partial L/\partial w_2 = \delta_2 \cdot x = {d2} \cdot {x} = {dw2}$,    "
                        r"$\partial L/\partial b_2 = \delta_2 = {db2}$"])
    t.write(
        "The hidden neurons are ordinary neurons too, with blame δ₁ and δ₂. So their knobs get the same rule "
        "as the output neuron's knobs:",
        M(r"$\partial L / \partial w_1 = \delta_1 \cdot x$      $\partial L / \partial b_1 = \delta_1$"),
        "All seven slopes are done, with a handful of multiplications, and every number needed was already on "
        "the wires.",
        H("The pattern"),
        "Every neuron, whichever layer it's in, does the same:",
        "1. receive the blame for its answer from the layer after it,",
        "2. multiply by the slope of its switch: that's its δ,",
        "3. knobs: δ × input, and δ for the bias,",
        "4. pass δ × weight back to the layer before it.",
    )


@step("All seven slopes, checked")
def page_check_net(t):
    left, right = t.split(gap=0.05, ratio=0.5)
    v = network(NET_EX, X_EX, Y_EX)
    knobs = knobs_of(NET_EX)

    def loss_at(k, value):
        changed = list(knobs)
        changed[k] = value
        return net_with(changed).loss(np.array([[X_EX]]), np.array([Y_EX]))

    ax = t.axes(left)
    ax.axis("off")
    ax.set_title(f"One afternoon ({T[EX]:.1f} °C)", loc="left", fontsize=11.5, color="#333")
    rows = ["knob    backprop     nudging"] + [f"{name:<5} {v['d' + name]:+.6f}   {nudge(lambda u: loss_at(k, u), knobs[k]):+.6f}"
                                              for k, name in enumerate(KNOB_NAMES)]
    ax.text(0, 0.95, "\n".join(rows).replace("-", "−"), family="monospace", fontsize=12, va="top", linespacing=1.9,
            bbox=dict(boxstyle="round,pad=0.6", fc="#f7f7f7", ec="#ddd"))

    grads = NET_EX.backprop(X1, Y)
    backprop = [grads.W[0, 0], grads.b[0], grads.W[1, 0], grads.b[1], grads.v[0], grads.v[1], grads.c]
    order_ = [0, 2, 1, 3, 4, 5, 6]              # Net.knobs(): w1, w2, b1, b2, v1, v2, c
    nudged = NET_EX.nudge_slopes(X1, Y)
    ax2 = t.axes(right)
    ax2.axis("off")
    ax2.set_title(f"All {n} afternoons (average)", loc="left", fontsize=11.5, color="#333")
    rows = ["knob    backprop     nudging"] + [f"{name:<5} {backprop[k]:+.6f}   {nudged[order_[k]]:+.6f}"
                                              for k, name in enumerate(KNOB_NAMES)]
    ax2.text(0, 0.95, "\n".join(rows).replace("-", "−"), family="monospace", fontsize=12, va="top", linespacing=1.9,
             bbox=dict(boxstyle="round,pad=0.6", fc="#f7f7f7", ec="#ddd"))
    t.write(
        "Left: the seven slopes of the last four steps, next to the slopes found by nudging each knob. They "
        "agree.",
        "Right: the same for the loss of all afternoons. More on that in the next part.",
        H("Count the work"),
        "Nudging: two extra forward passes for each of the 7 knobs, so 14 forward passes.",
        "Backpropagation: one forward pass and one backward pass. The backward pass does about as many "
        "multiplications as the forward pass.",
        "With 7 knobs that's a nice saving. With a billion knobs it's the difference between possible and "
        "impossible.",
    )


@step("A value used twice: add the slopes")
def page_fork(t):
    by_nudge = nudge(lambda u: NET_EX.loss(np.array([[u]]), np.array([Y_EX])), X_EX)
    network_page(t, 5, [r"$x$ feeds two neurons, so a nudge of $x$ reaches $L$ along two paths:",
                        r"$\partial L/\partial x = \delta_1 \cdot w_1 + \delta_2 \cdot w_2 = {d1} \cdot {w1} + {d2} \cdot {w2} = {dx}$",
                        "by nudging $x$:  {nudged}"], extra=dict(nudged=mm(by_nudge)))
    t.write(
        "One more rule. The input $x$ goes into both hidden neurons. Nudge it, and the nudge reaches $L$ along "
        "two paths, through neuron 1 and through neuron 2.",
        "The two effects simply add up:",
        M(r"$\dfrac{\partial L}{\partial x} = \delta_1 w_1 + \delta_2 w_2$"),
        "Each path contributes blame × weight, as in step 2.",
        H("Why this matters"),
        "We don't need the slope for $x$ here, since $x$ is not a knob. But in a network with two hidden "
        "layers, every answer of the first hidden layer feeds every neuron of the second. Its blame is the sum "
        "over all of them.",
        Box("a value used several times:\nadd up the slopes of all its uses"),
    )


course.part("All afternoons, in code")


@step("Many afternoons: average the slopes")
def page_average(t):
    left, bottom, width, height = t.content()
    ax = style(t.axes([left, bottom, 0.42, height]))
    choice = [t.state.get("avg_knob", 4)]
    bars = ax.bar(np.arange(n), np.zeros(n), color=np.where(Y == 1, C_YES, C_NO), width=0.8)
    mean_line = ax.axhline(0, color=C_SLOPE, linewidth=2.5)
    mean_label = ax.text(n - 1, 0, "", ha="right", va="bottom", fontsize=12, color=C_SLOPE, weight="bold",
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.axhline(0, color="#333", linewidth=1)
    ax.set(xlim=(-1, n), xlabel="afternoon (sorted by temperature)")
    per_afternoon = []
    for i in range(n):
        v = network(NET_EX, X1[i, 0], Y[i])
        per_afternoon.append([v["d" + name] for name in KNOB_NAMES])
    per_afternoon = np.array(per_afternoon)

    def update():
        k = choice[0]
        values = per_afternoon[:, k]
        for bar, value in zip(bars, values):
            bar.set_height(value)
        mean = values.mean()
        mean_line.set_ydata([mean, mean])
        mean_label.set(y=mean, text=f"average {mean:+.4f}")
        top = np.abs(values).max() * 1.1
        ax.set_ylim(-top, top)
        ax.set_title(f"Slope of each afternoon's penalty for {SLIDERS_LABEL[KNOB_NAMES[k]]}", loc="left",
                     fontsize=11.5, color="#333")

    def choose(k):
        choice[0] = t.state["avg_knob"] = k
        update()

    update()
    t.radio([0.5, 0.5, 0.1, 0.38], [SLIDERS_LABEL[name] for name in KNOB_NAMES], choose, active=choice[0], title="knob")
    t.write(
        "So far: one afternoon. But the loss is the average penalty over all afternoons:",
        M(r"$L = \dfrac{1}{n} \sum\ \mathrm{penalty}_i$"),
        "Nudge a knob, and every afternoon's penalty changes a bit. The loss changes by the average of those "
        "changes. So:",
        Box("slope of the loss = average of\nthe slopes of all afternoons"),
        "Each bar is one afternoon's slope, computed with the four backward steps (blue: comfortable, orange: "
        "not). The red line is their average: the slope gradient descent uses.",
        "The afternoons pull in different directions. The average says where most of them want the knob to go.",
    )


SLIDERS_LABEL = dict(w1="$w_1$", b1="$b_1$", w2="$w_2$", b2="$b_2$", v1="$v_1$", v2="$v_2$", c="$c$")
NET2_EX = train(Net.random(2, 4, 0), X2, Y, 2.0, 5000)


def matrix(ax, x, y, values, fmt="{:+.1f}", cell=(1.15, 0.55), highlight=(), color="#fff4cc", label=None,
           edge="#999"):
    """A small table of numbers drawn at x, y (top left corner). highlight: cells (row, col) or rows (row, None)."""
    rows, cols = values.shape
    for r in range(rows):
        for c in range(cols):
            lit = (r, c) in highlight or (r, None) in highlight or (None, c) in highlight
            ax.add_patch(Rectangle((x + c * cell[0], y - (r + 1) * cell[1]), cell[0], cell[1], facecolor=color if lit
                                   else "white", edgecolor=edge, linewidth=0.8, zorder=2))
            ax.text(x + (c + 0.5) * cell[0], y - (r + 0.5) * cell[1], fmt.format(values[r, c]).replace("-", "−"),
                    ha="center", va="center", fontsize=10, family="monospace", zorder=3)
    if label:
        ax.text(x + cols * cell[0] / 2, y + 0.35, label, ha="center", va="bottom", fontsize=13)


@step("Many neurons: a table of weights")
def page_matrix(t):
    ax = diagram_axes(t, t.content(), (0, 17), (-2.2, 7.8))
    net = NET2_EX
    Xs = X2[:5]
    Z = Xs @ net.W.T + net.b
    top = 6.4
    matrix(ax, 0.3, top, Xs, label="X", highlight=[(0, None)], fmt="{:+.2f}")
    ax.text(1.45, top - 5 * 0.55 - 0.3, "(140 rows, 5 shown)", ha="center", va="top", fontsize=10, color="#666")
    ax.text(3.25, top - 0.55, "@", fontsize=20, ha="center", va="center")
    matrix(ax, 3.9, top, net.W.T, label="W.T", highlight=[(None, 0)], fmt="{:+.2f}")
    ax.text(6.2, top - 2 * 0.55 - 0.3, "(one column per hidden neuron)", ha="center", va="top", fontsize=10,
            color="#666")
    ax.text(9.15, top - 0.55, "+", fontsize=20, ha="center", va="center")
    matrix(ax, 9.8, top, net.b[None, :], label="b", highlight=[(0, 0)], fmt="{:+.2f}")
    ax.text(3.25, 1.6, "=", fontsize=20, ha="center", va="center")
    matrix(ax, 3.9, 2.9, Z, label="Z: scores of the hidden neurons", highlight=[(0, 0)], fmt="{:+.2f}")
    ax.text(6.2, 2.9 - 5 * 0.55 - 0.3, "(140 rows, 4 columns)", ha="center", va="top", fontsize=10, color="#666")
    x1, x2 = Xs[0]
    ax.text(0.3, -1.7, rf"first afternoon, neuron 1:   ${mm(x1, 2)} \cdot ({mm(net.W[0, 0], 2)}) + ({mm(x2, 2)}) "
                       rf"\cdot ({mm(net.W[0, 1], 2)}) + ({mm(net.b[0], 2)}) = {mm(Z[0, 0], 2)}$", fontsize=13)
    t.write(
        "Real networks have many neurons and many afternoons. Writing a loop over all of them would be slow. "
        "Instead, numpy computes with tables of numbers, called matrices.",
        "Here the network with 2 inputs and 4 hidden neurons from nn_tutor.py:",
        "• X: one row per afternoon, one column per input ($x_1$, $x_2$),",
        "• W: one row per hidden neuron, with its two weights,",
        "• b: the four biases.",
        "The @ multiplies every afternoon with every neuron's weights and adds them up: one number per "
        "afternoon and neuron. Adding b gives all 560 hidden scores in one line.",
        "The yellow cells show one of them.",
    )


@step("The forward pass in code")
def page_code_forward(t):
    left, bottom, width, height = t.content()
    below = code_box(t, left, 0.86, width, code_of(Net.forward), title="tinynet.py: the real code both courses run",
                     marks={2: "hidden layer", 3: "output"})
    H_, p = NET2_EX.forward(X2)
    rows = [("X", f"{X2.shape[0]} × {X2.shape[1]}", "afternoons × inputs"),
            ("self.W", f"{NET2_EX.W.shape[0]} × {NET2_EX.W.shape[1]}", "hidden neurons × inputs"),
            ("self.b", f"{NET2_EX.b.shape[0]}", "one bias per hidden neuron"),
            ("H", f"{H_.shape[0]} × {H_.shape[1]}", "afternoons × hidden neurons"),
            ("self.v", f"{NET2_EX.v.shape[0]}", "one weight per hidden neuron"),
            ("p", f"{p.shape[0]}", "one chance per afternoon")]
    shapes = t.axes([left, bottom, width, below - bottom - 0.08])
    shapes.axis("off")
    shapes.set_title("Sizes, for the network with 2 inputs and 4 hidden neurons", loc="left", fontsize=11.5, color="#333")
    shapes.text(0, 0.97, "\n".join(f"{a:<8}{b:<10}{c}" for a, b, c in rows), family="monospace", fontsize=12,
                va="top", linespacing=1.9)
    t.write(
        "This is the forward pass of tinynet.py, word for word. It computes the whole network for all "
        "afternoons in two lines:",
        "• Line 2: all hidden scores (the table of the last step), then σ for every one of them. H holds the "
        "answers of the hidden neurons.",
        "• Line 3: every afternoon's hidden answers weighted with v, plus c, then σ: the chances $p$.",
        "Both results are returned, because the backward pass needs H: it holds the values on the wires.",
        "The sizes (bottom left) help to read the code: in X @ self.W.T, the 2 columns of X meet the 2 "
        "inputs of every hidden neuron.",
    )


@step("The backward pass in code")
def page_code_backward(t):
    left, bottom, width, height = t.content()
    below = code_box(t, left, 0.86, width, code_of(Net.backprop), title="tinynet.py",
                     marks={2: "forward", 4: "step 1", 5: "steps 2 + 3", 6: "step 4"})
    notes = t.axes([left, bottom, width, below - bottom - 0.06])
    notes.axis("off")
    notes.text(0, 0.95, "The slopes have the same shape as the knobs, so they are packed into a Net too.\n"
                        "Net.step() then moves every knob:   W - eta * slopes.W,   b - eta * slopes.b, ...\n\n"
                        "Sizes: e has 140 numbers, D is 140 × 4 like H, D.T @ X is 4 × 2 like W.",
               fontsize=12, va="top", linespacing=1.8)
    t.write(
        "And this is the backward pass. Compare with the four steps of the last part:",
        "• e = p − y: the blame δ of the output score, one number per afternoon (step 1).",
        "• np.outer(e, self.v): blame × weight, for every afternoon and hidden neuron (step 2). Then × H·(1 − H), "
        "the slope of σ (step 3). D holds the blames δ₁, δ₂, ... of all hidden neurons.",
        "• The last line: blame × input, averaged over the afternoons (step 4 and the output knobs). D.T @ X "
        "multiplies every blame with every input and adds up; divided by n it's the average.",
        "Six lines for any number of afternoons and hidden neurons.",
    )


@step("Gradient check: all 17 knobs")
def page_gradient_check(t):
    left, right = t.split(slider_rows=1.5, gap=0.08, ratio=0.55)
    ax = style(t.axes(left), title="Every dot is one knob")
    dots = ax.scatter([], [], s=60, color=C_NET, zorder=4)
    diagonal, = ax.plot([], [], color="#999", linestyle="--", linewidth=1.5)
    ax.set(xlabel="slope by nudging", ylabel="slope by backpropagation")
    info = readout(ax, fontsize=12, family="monospace")
    table_ax = t.axes(right)
    table_ax.axis("off")
    table = table_ax.text(0, 1, "", family="monospace", fontsize=10.5, va="top", linespacing=1.55)
    seed = [t.state.get("check_seed", 1)]

    def update():
        net = Net.random(2, 4, seed[0], spread=1.5)
        by_backprop = net.backprop(X2, Y).knobs()
        by_nudge = net.nudge_slopes(X2, Y)
        dots.set_offsets(np.column_stack([by_nudge, by_backprop]))
        lim = np.abs(np.concatenate([by_nudge, by_backprop])).max() * 1.15
        diagonal.set_data([-lim, lim], [-lim, lim])
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim))
        info.set_text(f"random knobs no. {seed[0]}\nbiggest difference: {np.abs(by_nudge - by_backprop).max():.1e}")
        names = [f"w{j + 1}{i + 1}" for j in range(4) for i in range(2)] + [f"b{j + 1}" for j in range(4)] + \
                [f"v{j + 1}" for j in range(4)] + ["c"]
        rows = ["knob   backprop    nudging"] + [f"{name:<4} {a:+.7f} {b:+.7f}" for name, a, b in
                                                zip(names, by_backprop, by_nudge)]
        table.set_text("\n".join(rows).replace("-", "−"))

    def new_knobs():
        seed[0] = t.state["check_seed"] = seed[0] + 1
        update()

    update()
    t.button([0.12, 0.035, 0.14, 0.05], "new random knobs", new_knobs)
    t.write(
        "The network with 2 inputs and 4 hidden neurons has 17 knobs. For random knobs, compute all 17 slopes "
        "of the loss twice: with backprop from the code of the last step, and by nudging every knob. (w12 "
        "is the weight of hidden neuron 1 for input 2.)",
        "If backprop is right, every dot lies on the diagonal. It does, and the biggest difference is tiny: "
        "rounding errors of the nudging.",
        H("Always do this"),
        "Backpropagation code is easy to get subtly wrong: a missing factor, a wrong row. The network then "
        "still learns a bit, just badly, and nobody notices.",
        "So people check new backprop code exactly like this, on a small network: a gradient check. Then "
        "they switch nudging off, because it's far too slow for real training.",
    )


@step("The training loop")
def page_training_loop(t):
    run = t.state.setdefault("loop", dict(net=Net.random(2, 4, 0), losses=[]))
    if not run["losses"]:
        run["losses"].append(run["net"].loss(X2, Y))
    left, bottom, width, height = t.content(slider_rows=1.5)
    below = code_box(t, left, 0.86, 0.29, code_of(train), title="tinynet.py")
    ax2 = style(t.axes([left, bottom, 0.29, below - bottom - 0.08]), title="Loss")
    line, = ax2.plot([], [], color=C_NET, linewidth=2)
    ax2.set(xlabel="step")
    ax = map_axes(t, [left + 0.34, bottom, 0.2, height], title="The network")
    shade = wash(ax)
    plot_map_points(ax, label=False, s=18)
    border = [None]

    def update():
        net = run["net"]
        p = net.forward(GX)[1]
        shade.set_data(p.reshape(GT.shape))
        if border[0] is not None:
            border[0].remove()
        border[0] = ax.contour(GT, np.linspace(*HLIM, GT.shape[0])[:, None] * np.ones_like(GT), p.reshape(GT.shape),
                               levels=[0.5], colors=C_FIT, linewidths=2)
        losses = run["losses"]
        line.set_data(np.arange(len(losses)), losses)
        ax2.set(xlim=(0, max(100, len(losses) - 1)), ylim=(0, max(0.9, max(losses) * 1.05)))
        ax.set_title(f"The network after {len(losses) - 1} steps", loc="left", fontsize=11.5, color="#333")

    def go(steps):
        for _ in range(steps):
            run["net"] = run["net"].step(run["net"].backprop(X2, Y), 2.0)
            run["losses"].append(run["net"].loss(X2, Y))
        update()

    def restart():
        stop()
        run.update(net=Net.random(2, 4, 0), losses=[Net.random(2, 4, 0).loss(X2, Y)])
        update()

    update()
    t.buttons([("100 steps", lambda: go(100)), ("1000 steps", lambda: go(1000))], y=0.035, x=0.08)
    stop = t.play_button([0.27, 0.035, 0.085, 0.05], lambda: go(25))
    t.button([0.365, 0.035, 0.085, 0.05], "reset", restart)
    t.write(
        "Put it all together, and training is a loop around one line (top left, the rest only keeps a "
        "history for plots):",
        "1. backprop: one forward and one backward pass give all slopes,",
        "2. step: every knob moves a little against its slope.",
        "That's the loop that trained every network in nn_tutor.py, and in principle every neural network in "
        "the world. Big systems add smarter steps, random batches of examples and graphics cards, but the "
        "heart is the same.",
        H("Try it"),
        "Click 'play' and watch the island form on the right.",
    )


course.part("Deep networks")

ACTIVATIONS = {
    "sigmoid": (sigmoid, lambda H_: H_ * (1 - H_)),
    "ReLU": (lambda z: np.maximum(0, z), lambda H_: (H_ > 0).astype(float)),
}


def deep_slopes(Ws, bs, X, y, act):
    """Backpropagation through any number of layers: the same few lines, once for every layer."""
    switch, switch_slope = ACTIVATIONS[act]
    H = [X]                                    # the answers of every layer
    for W, b in zip(Ws[:-1], bs[:-1]):
        H.append(switch(H[-1] @ W.T + b))
    p = sigmoid(H[-1] @ Ws[-1].T + bs[-1])[:, 0]
    D = (p - y)[:, None] / len(y)              # blame of the output score
    dWs, dbs = [], []
    for k in reversed(range(len(Ws))):         # last layer first
        dWs.insert(0, D.T @ H[k])              # weights: blame x input
        dbs.insert(0, D.sum(axis=0))           # biases: blame
        if k > 0:                              # blame of the layer below:
            D = (D @ Ws[k]) * switch_slope(H[k])
    return dWs, dbs


def deep_net(act, layers, width, seed):
    """Random starting knobs, scaled the way people start such networks (Xavier for sigmoid, He for ReLU)."""
    rng = np.random.default_rng(seed)
    sizes = [2] + [width] * layers + [1]
    gain = 2.0 if act == "ReLU" else 1.0
    return ([rng.normal(0, np.sqrt(gain / a), (b, a)) for a, b in zip(sizes, sizes[1:])],
            [np.zeros(b) for b in sizes[1:]])


@cache
def deep_check():
    """Largest difference between deep_slopes and nudging, for a small network with three hidden layers."""
    Ws, bs = deep_net("sigmoid", 3, 3, 0)

    def loss(Ws_, bs_):
        H_ = X2
        for W, b in zip(Ws_[:-1], bs_[:-1]):
            H_ = sigmoid(H_ @ W.T + b)
        z = (H_ @ Ws_[-1].T + bs_[-1])[:, 0]
        return np.mean(np.logaddexp(0, z) - Y * z)

    dWs, _ = deep_slopes(Ws, bs, X2, Y, "sigmoid")
    worst = 0.0
    for k, W in enumerate(Ws):
        for index in np.ndindex(W.shape):
            def at(value):
                changed = [w.copy() for w in Ws]
                changed[k][index] = value
                return loss(changed, bs)
            worst = max(worst, abs(nudge(at, W[index]) - dWs[k][index]))
    return worst


@step("Any number of layers: a loop")
def page_deep_code(t):
    left, bottom, width, height = t.content()
    code_box(t, left, 0.86, width, code_of(deep_slopes), title="backprop_tutor.py", fontsize=12.5,
             marks={3: "forward", 7: "start", 9: "backward", 10: "knobs", 13: "blame back"})
    t.write(
        "With more hidden layers nothing new happens. The four steps of every neuron repeat, layer by layer, "
        "from the last to the first:",
        "• forward: keep the answers H of every layer,",
        "• start with the blame p − y,",
        "• in every layer: weights get blame × input, biases the blame,",
        "• blame × weights, times the slope of the switch, is the blame of the layer below.",
        "D @ W adds up the blame from all neurons an answer feeds: the rule of adding slopes of a value used "
        "several times.",
        f"A gradient check for a network with three hidden layers: biggest difference to nudging "
        f"{deep_check():.0e}.",
        "(The division by len(y) happens right at the start, so the sums in the loop are already averages.)",
    )


@step("Blame fades in deep sigmoid networks")
def page_vanish(t):
    left, right = t.split(slider_rows=2, gap=0.07, ratio=0.62)
    ax = style(t.axes(left), title="Slope for the weight in each layer, compared to the last layer")
    bars = ax.bar([], [])
    ax.set_yscale("log")
    ax.set(ylabel="relative size of the slope", xlabel="layer (1 = next to the input)", ylim=(1e-13, 30))
    ax.axhline(1, color="#999", linewidth=1)
    diagram = diagram_axes(t, right, (0, 4), (-0.5, 10.5))
    info = readout(ax, x=0.02, y=0.97, fontsize=12, family="monospace")

    def update():
        nonlocal bars
        depth, w = int(t.guess["depth"]), t.guess["wd"]
        factor = w * 0.25
        sizes = factor ** (depth - np.arange(1, depth + 1))
        bars.remove()
        bars = ax.bar(np.arange(1, depth + 1), sizes, color=np.where(sizes < 1e-3, C_SLOPE, C_NET), width=0.7)
        ax.set_xlim(0.3, depth + 0.7)
        ax.set_xticks(range(1, depth + 1))
        info.set_text(f"per layer: w · 0.25 = {factor:.3f}\nlayer 1: {factor:.3f}^{depth - 1} = {sizes[0]:.1e}")
        diagram.clear()
        diagram.axis("off")
        diagram.set(xlim=(0, 4), ylim=(-0.5, 10.5))
        shown = min(depth, 8)
        for k in range(shown):
            yy = 0.4 + k * 9.2 / max(shown - 1, 1)
            node(diagram, 2, yy, "", color=C_HID[0], size=0.32)
            if k < shown - 1:
                arrow(diagram, (2, yy + 0.35), (2, yy + 9.2 / max(shown - 1, 1) - 0.35), color="#999")
        diagram.text(3.0, 9.6, "last layer", fontsize=10, va="center")
        diagram.text(3.0, 0.4, "layer 1", fontsize=10, va="center")
        if depth > shown:
            diagram.text(2.9, 5, f"({depth} layers,\n{shown} drawn)", fontsize=9.5, color="#777", va="center")
        diagram.annotate("", (0.8, 0.4), (0.8, 9.6), arrowprops=dict(arrowstyle="-|>", color=C_SLOPE, linewidth=2.5,
                                                                   mutation_scale=18))
        diagram.text(0.45, 5, "blame", rotation=90, color=C_SLOPE, fontsize=11, ha="center", va="center")

    t.sliders(["depth", "wd"], update)
    t.write(
        "Picture a deep network as a tall stack of layers, one neuron each, all with the same weight $w$. On "
        "its way down, the blame passes every layer, and each one multiplies it (steps 2 and 3 of the "
        "backward pass) by",
        M(r"weight $\times$ slope of σ"),
        "The slope of σ is at most 0.25. Even in the best case, with every neuron exactly in the middle of its "
        "S, a weight of 1 shrinks the blame to a quarter in every layer.",
        "After 10 layers only a few millionths are left. The first layers get slopes of almost 0 and hardly "
        "learn: vanishing gradients.",
        H("Try it"),
        "Can bigger weights help? A weight of 4 keeps the size. But big weights push the neurons into the flat "
        "ends of σ, where the slope is far below 0.25. It doesn't work in practice.",
    )


@cache
def deep_sizes():
    """Average size of the weight slopes per layer at the start, 10 hidden layers of 8 neurons, 20 random starts."""
    result = {}
    for act in ACTIVATIONS:
        sizes = []
        for seed in range(20):
            Ws, bs = deep_net(act, 10, 8, seed)
            dWs, _ = deep_slopes(Ws, bs, X2, Y, act)
            sizes.append([np.abs(dW).mean() for dW in dWs])
        result[act] = np.mean(sizes, axis=0)
    return result


@step("ReLU keeps the blame alive")
def page_relu_deep(t):
    sizes = deep_sizes()
    left, right = t.split(gap=0.07, ratio=0.62)
    ax = style(t.axes(left), title="Real networks: 10 hidden layers of 8 neurons, at the start of training")
    layers = np.arange(1, 12)
    ax.bar(layers - 0.2, sizes["sigmoid"], width=0.4, color="#999", label="sigmoid")
    ax.bar(layers + 0.2, sizes["ReLU"], width=0.4, color=C_NET, label="ReLU")
    ax.set_yscale("log")
    ax.set_xticks(layers, [str(k) for k in range(1, 11)] + ["out"])
    ax.set(xlabel="layer (1 = next to the input)", ylabel="average size of the weight slopes", ylim=(1e-10, 1))
    ax.legend(loc="upper left", frameon=False)
    ax2 = style(t.axes([right[0], right[1] + 0.45 * right[3], right[2], 0.5 * right[3]]), title="Slope of the switch")
    grid = np.linspace(-4, 4, 400)
    ax2.plot(grid, sigmoid(grid) * (1 - sigmoid(grid)), color="#999", linewidth=2.5, label="sigmoid")
    ax2.plot(grid, (grid > 0).astype(float), color=C_NET, linewidth=2.5, label="ReLU")
    ax2.set(xlim=(-4, 4), ylim=(-0.05, 1.15), xlabel="score $z$")
    ax2.legend(loc="upper left", frameon=False)
    ratio = sizes["sigmoid"][-1] / sizes["sigmoid"][0]
    t.write(
        "These bars are measured, not imagined: random starting networks with 10 hidden layers on the office "
        f"data, backprop with the loop of step {no(page_deep_code)}, averaged over 20 starts.",
        f"• sigmoid (grey): the first layer's slopes are about {ratio / 1e6:.0f} million times smaller than the "
        "output's. "
        "Those layers would practically never learn.",
        "• ReLU (purple): the slopes stay about the same size through all layers.",
        H("Why"),
        "The slope of ReLU is exactly 1 for every active neuron (top right). The blame is only multiplied by "
        "the weights, and with a well-chosen random start they keep its size on average.",
        "That's one of the reasons why networks with dozens or hundreds of layers became possible.",
    )


@step("Summary: backpropagation in five rules")
def page_summary(t):
    left, bottom, width, height = t.content()
    ax = network_axes(t, [left, bottom + 0.3 * height, width, 0.7 * height])
    v = network(NET_EX, X_EX, Y_EX)
    g = network_graph(ax, v)
    g.show(slopes=[key for keys in NET_STAGES for key in keys])
    text_ax = formula_axes(t, [left, bottom, width, 0.28 * height])
    text_ax.text(0, 0.8, "Forward: values flow right, and are kept.", fontsize=13, transform=text_ax.transAxes)
    text_ax.text(0, 0.5, "Backward: slopes flow left, each box multiplies by its own slope.", fontsize=13,
                 color=C_SLOPE, transform=text_ax.transAxes)
    t.write(
        H("Backpropagation in five rules"),
        "1. Forward pass: compute every value and keep it.",
        "2. Start at the end: the blame of the output score is $\\delta = p - y$.",
        "3. Knobs: weight gets δ × input, bias gets δ.",
        "4. Back along a weight: δ × weight. Through a switch: × its slope, σ: $h(1-h)$, ReLU: 0 or 1.",
        "5. A value used several times: add up the slopes.",
        "Average over all examples, step downhill, repeat.",
        H("Where to go from here"),
        "• Run nn.py and compare its knobs with tinynet.py.",
        "• Read tinynet.py: every line should now make sense.",
        "• Libraries like PyTorch apply exactly these rules automatically to any chain of boxes you write "
        "('autograd').",
    )


if __name__ == "__main__":
    course.run()
