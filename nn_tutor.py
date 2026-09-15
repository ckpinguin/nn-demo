"""
Neural networks, step by step.

A course for beginners in very small steps: from a single artificial neuron to a network
with a hidden layer that learns by itself, on the office-climate data that nn.py uses.
How the slopes for learning are found (backpropagation) has a course of its own:
backprop_tutor.py.

    python nn_tutor.py                 # Back / Next buttons or the arrow keys
    python nn_tutor.py --step 20       # open the course at step 20
    python nn_tutor.py --save pages    # export every step as a PNG instead
"""
from functools import cache

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle

from office import (C_FIT, C_HID, C_MEAN, C_NET, C_NO, C_WRONG, C_YES, GH, GT, GX, HLIM, HU, JIT, PROB, T, TG, TLIM,
                    X1, X2, X2T, XG, Y, YT, clip_line, map_axes, n, n_test, plot_map_points, plot_test_votes,
                    plot_votes, strip, switch_temp, temp_axes, to_temp, to_x, wash)
from tinynet import Net, X_train, rescale, sigmoid, train, y_train
from tutorkit import Box, Course, H, M, arrow, box, diagram_axes, m, node, readout, rings, set_points, style


def fit_neuron(X, labels, eta=1.0, steps=3000):
    """The best single neuron (logistic regression), by gradient descent from 0."""
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        e = sigmoid(X @ w + b) - labels
        w, b = w - eta * X.T @ e / len(labels), b - eta * e.mean()
    return w, b


def log_loss(p, labels):
    return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))


# --- Everything the course computes in advance ---
(ONE_W,), ONE_B = fit_neuron(X1, Y)      # one neuron, temperature only
NET1 = train(Net.random(1, 2, 0), X1, Y, 2.0, 3000)            # two hidden neurons, temperature only
SUGGEST = dict(w1=4.0, b1=4.0, w2=4.0, b2=-4.0, v1=8.0, v2=-8.0, c=-4.0)
KNOBS = ["w1", "b1", "w2", "b2", "v1", "v2", "c"]
THREE = (16, 22, 29)                     # a cold, a nice and a hot afternoon


def net_of(g):
    """The one-input network with the reader's seven knobs."""
    return Net([[g["w1"]], [g["w2"]]], [g["b1"], g["b2"]], [g["v1"], g["v2"]], g["c"])


def fewest_switch_mistakes():
    """The fewest wrong votes a single neuron with a hard switch can reach: one threshold, either direction."""
    cuts = np.concatenate([[-np.inf], (X1[1:, 0] + X1[:-1, 0]) / 2, [np.inf]])
    return min(min(((X1[:, 0] >= cut) != Y).sum(), ((X1[:, 0] < cut) != Y).sum()) for cut in cuts)


BEST_SWITCH = fewest_switch_mistakes()
EX = int(np.flatnonzero((Y == 1) & (T >= 24))[0])                 # example afternoon: a comfortable one at about 24 °C

SLIDERS = {     # key: (low, high, start, label, format)
    "w": (-4, 4, 1.5, r"weight $w$", "%.2f"),
    "b": (-6, 6, 0.0, r"bias $b$", "%.2f"),
    "z": (-7, 7, 1.5, r"score $z$", "%.2f"),
    "w1": (-10, 10, 1.5, r"$w_1$", "%.2f"),
    "b1": (-10, 10, 0.5, r"$b_1$", "%.2f"),
    "w2": (-10, 10, -1.0, r"$w_2$", "%.2f"),
    "b2": (-10, 10, 1.0, r"$b_2$", "%.2f"),
    "v1": (-12, 12, 1.0, r"$v_1$", "%.2f"),
    "v2": (-12, 12, 1.0, r"$v_2$", "%.2f"),
    "c": (-12, 12, 0.0, r"$c$", "%.2f"),
    "a1": (-8, 8, 1.5, r"$w_1$ (temperature)", "%.2f"),
    "a2": (-8, 8, -1.0, r"$w_2$ (humidity)", "%.2f"),
    "a0": (-8, 8, 0.5, r"bias $b$", "%.2f"),
    "c4": (-8, 20, 4.0, r"output bias $c$", "%.1f"),
    "snap": (0, 30000, 20000, "training steps", "%d", 500),
}

GLOSSARY = [    # (pattern, symbol, meaning, formula, details)
    (r"\\sigma|σ", r"\sigma", "sigmoid, squashes any score into a number between 0 and 1",
     r"$\sigma(z) = 1 / (1 + e^{-z})$",
     "Exactly 0.5 at z = 0. Close to 0 far below, close to 1 far above, but it never reaches 0 or 1. "
     "Also called the logistic function."),
    (r"e\^", "e", "Euler's number, about 2.718",
     r"$e^{2} = e \cdot e = 7.39$,   $e^{0} = 1$,   $e^{-2} = 1 / e^{2} = 0.14$",
     "A constant like π. e to the power of a big number is huge, to the power of a very negative "
     "number it is tiny, but always positive."),
    (r"\\ln", r"\ln", "natural logarithm, undoes e to the power of",
     r"$\ln(e^{x}) = x$,   $\ln 1 = 0$,   $\ln 0.5 = -0.69$",
     "Negative for numbers between 0 and 1, and it heads to minus infinity as the number shrinks to 0."),
    (r"\\sum", r"\sum", "sum: add up the term for every afternoon",
     r"$\sum a_i = a_1 + a_2 + \dots + a_n$",
     "The index i counts through all n afternoons. Divided by n, the sum becomes the average."),
    (r"\\eta|η", r"\eta", "learning rate, the step size of gradient descent",
     r"knob $\leftarrow$ knob $-\ \eta \cdot$ slope",
     "Too small and training crawls. Too big and the steps overshoot, jump around, or push the neurons "
     "into the flat ends of the sigmoid, where learning stops."),
    (r"[Ll]oss", "L", "loss, the average penalty for the chances given to what really happened",
     r"$L = \frac{1}{n} \sum (-\ln q_i)$",
     "Also called log-loss or cross-entropy. 0 would mean every afternoon got chance 1 for the real "
     "vote. Always saying 50% gives ln 2 = 0.69. Confident mistakes make it explode."),
    (r"[Gg]radient", "gradient", "the list of the slopes of the loss, one for every knob",
     r"$(\mathrm{slope\ for\ } w_1,\ \mathrm{slope\ for\ } b_1,\ \dots)$",
     "It points in the direction in which the loss grows fastest. Gradient descent steps the "
     "opposite way, downhill."),
    (r"[Aa]ccuracy", "accuracy", "share of the afternoons classified correctly",
     r"$\mathrm{accuracy} = \mathrm{correct} / n$",
     "Easy to read, but coarse: it ignores how sure the network was. A 51% and a 99% answer count the "
     "same as long as they are on the right side of 50%."),
    (r"ReLU", "ReLU", "rectified linear unit, an activation function: negative scores become 0",
     r"$\mathrm{ReLU}(z) = \max(0,\ z)$",
     "The most common activation in modern networks: fast to compute, and its slope is 1 for every "
     "positive score, so learning signals don't fade in deep networks."),
    (r"[Oo]verfitting", "overfitting", "learning the training examples by heart, noise included",
     r"training loss $\downarrow$,   test loss $\uparrow$",
     "The network gets better on the afternoons it trains on and worse on new ones. Fewer knobs, more "
     "data, stopping early or punishing big weights help."),
]

course = Course("Neural networks, step by step", GLOSSARY, SLIDERS, __doc__)
step, no = course.step, course.no


# --- Plot building blocks ---
def sigmoid_glyph(ax, x, y, size=0.55, color=C_NET):
    s = np.linspace(-1, 1, 40)
    ax.plot(x + size * s, y + size * (sigmoid(5 * s) - 0.5) * 1.4, color=color, linewidth=2, zorder=5)


def knob_glyph(ax, x, y, angle, radius=0.32, color=C_NET):
    ax.add_patch(Circle((x, y), radius, facecolor="#f3ecf7", edgecolor=color, linewidth=2, zorder=5))
    ax.plot([x, x + 0.8 * radius * np.cos(angle)], [y, y + 0.8 * radius * np.sin(angle)],
            color=color, linewidth=2.5, zorder=6)


NET_POS = {"x": (1.3, 3.5), "h1": (5.8, 5.6), "h2": (5.8, 1.4), "p": (10.3, 3.5)}


def draw_network(ax, g, radius=0.62, values=None):
    """The one-input network with its seven knobs on the arrows and next to the neurons.
    Returns the texts inside the neurons, so pages can fill in values."""
    (x0, y0), (h1x, h1y), (h2x, h2y), (px, py) = NET_POS.values()
    label_box = dict(boxstyle="round,pad=0.15", fc="white", ec="none")
    for (ax_, ay), (bx, by), text, color, t_ in [((x0, y0), (h1x, h1y), rf"$w_1 = {m(g['w1'], 2)}$", C_HID[0], 0.45),
                                                 ((x0, y0), (h2x, h2y), rf"$w_2 = {m(g['w2'], 2)}$", C_HID[1], 0.45),
                                                 ((h1x, h1y), (px, py), rf"$v_1 = {m(g['v1'], 2)}$", C_HID[0], 0.5),
                                                 ((h2x, h2y), (px, py), rf"$v_2 = {m(g['v2'], 2)}$", C_HID[1], 0.5)]:
        d = np.hypot(bx - ax_, by - ay)
        ux, uy = (bx - ax_) / d, (by - ay) / d
        arrow(ax, (ax_ + radius * ux, ay + radius * uy), (bx - radius * ux, by - radius * uy), color="#777")
        ax.text(ax_ + t_ * (bx - ax_), ay + t_ * (by - ay) + 0.05, text, ha="center", va="center", fontsize=12,
                color=color, zorder=6, bbox=label_box)
    texts = {
        "x": node(ax, x0, y0, "$x$", color=C_MEAN, size=radius, fontsize=16),
        "h1": node(ax, h1x, h1y, "$h_1$", color=C_HID[0], size=radius, fontsize=16),
        "h2": node(ax, h2x, h2y, "$h_2$", color=C_HID[1], size=radius, fontsize=16),
        "p": node(ax, px, py, "$p$", color=C_NET, size=radius, fontsize=16),
    }
    ax.text(h1x, h1y + radius + 0.35, rf"$b_1 = {m(g['b1'], 2)}$", ha="center", fontsize=12, color=C_HID[0])
    ax.text(h2x, h2y - radius - 0.35, rf"$b_2 = {m(g['b2'], 2)}$", ha="center", va="top", fontsize=12, color=C_HID[1])
    ax.text(px, py + radius + 0.35, rf"$c = {m(g['c'], 2)}$", ha="center", fontsize=12, color=C_NET)
    return texts


# === The steps ===

course.part("The question")


@step("The question")
def page_question(t):
    ax = style(t.axes(t.content()), title=f"{n + n_test} afternoons in an office: the temperature at 3 pm and the staff's vote")
    ax.set(xlim=TLIM, ylim=(-0.45, 1.75), xlabel="temperature at 3 pm (°C)", ylabel="vote $y$")
    ax.set_yticks([0, 1])
    plot_votes(ax)
    plot_test_votes(ax, alpha=0.45)
    ax.legend(loc="upper center", ncols=2, frameon=False, fontsize=10.5)
    t.write(
        H("What we want"),
        "Every dot is one afternoon in an office. It shows the temperature at 3 pm and how the staff "
        "voted: blue means comfortable, orange means not comfortable.",
        "Too cold is uncomfortable, too warm as well. In between most people are happy, but not always.",
        "We want a machine that reads the thermometer and tells us the chance that the afternoon feels "
        "comfortable. We will build it out of artificial neurons: a neural network.",
        H("How this course works"),
        "In very small steps. Click Next or press the → key to go on, and ← to go back. Many steps have "
        "sliders and buttons: play with them, nothing can break.",
        "Symbols and special words are explained at the bottom of a page. Hover over them with the mouse "
        "for more details.",
        Box("no maths needed beyond $+$, $-$, $\\times$ and $\\div$"),
    )


@step("Learning from examples")
def page_examples(t):
    ax = diagram_axes(t, t.content(), (0, 14), (0, 9.4))
    box(ax, 2, 7.6, 2.6, 1.4, "24.3 °C", color=C_MEAN, fontsize=15)
    ax.text(2, 6.5, "input", ha="center", va="top", fontsize=11, color="#555")
    arrow(ax, (3.4, 7.6), (4.7, 7.6), width=2)
    box(ax, 7, 7.6, 4.2, 2.4, "", color=C_NET)
    ax.text(7, 8.4, "model", ha="center", va="center", fontsize=14, weight="bold", color=C_NET, zorder=6)
    for k, angle in enumerate([2.2, 0.6, 1.4]):
        knob_glyph(ax, 5.9 + 1.1 * k, 7.25, angle)
    ax.text(7, 6.62, "knobs", ha="center", va="center", fontsize=10, color="#555", zorder=6)
    arrow(ax, (9.2, 7.6), (10.5, 7.6), width=2)
    box(ax, 12, 7.6, 2.6, 1.4, "81 %\ncomfortable", color=C_YES, fontsize=13)
    ax.text(12, 6.5, "answer", ha="center", va="top", fontsize=11, color="#555")

    ax.text(0.4, 4.9, "examples with a known vote", fontsize=12, weight="bold", color="#333")
    picks = [int(np.argmin(abs(T - temp))) for temp in (16.1, 19.7, 22.4, 25.0, 28.6)]
    for row, i in enumerate(picks):
        yy = 4.1 - 0.72 * row
        ax.text(0.6, yy, f"{T[i]:.1f} °C", fontsize=12, family="monospace", va="center")
        ax.text(2.7, yy, "comfortable" if Y[i] else "not comfortable", fontsize=12, va="center",
                color=C_YES if Y[i] else C_NO, weight="bold")
    ax.annotate("", (6.6, 6.3), (5.3, 3.0), arrowprops=dict(arrowstyle="-|>", color=C_NET, linewidth=2,
                                                            connectionstyle="arc3,rad=-0.2", mutation_scale=18))
    ax.text(6.7, 4.1, "training:\nturn the knobs until the\nanswers fit the examples", fontsize=12, color=C_NET,
            va="center")
    box(ax, 12, 1.6, 3.4, 1.8, f"{n_test} test afternoons\nkept aside", color="#999", fc="#f4f4f4", fontsize=12)
    ax.text(12, 0.35, "for an honest check at the end", ha="center", fontsize=10.5, color="#777")
    t.write(
        "A model is a recipe that turns an input into an answer, here: temperature in, chance of "
        "'comfortable' out.",
        "Inside the recipe are numbers we can change, the knobs. Turn them, and the answers change.",
        "Training means: turn the knobs until the model's answers fit the examples, the afternoons where "
        "we already know how the staff voted. Then the model can answer for new afternoons too.",
        H("Training and test data"),
        f"Before training, {n_test} of the {n + n_test} afternoons are put aside (squares on the first page). "
        f"The model never sees them while it learns, only the other {n}.",
        "At the end they show honestly how well the model works on afternoons it has never seen, just "
        "like an exam with new questions.",
        Box(f"{n} afternoons to learn from, {n_test} for the test"),
    )


course.part("One neuron")


@step("Small numbers around zero")
def page_rescale(t):
    ax = t.axes(t.content())
    ax.axis("off")
    ax.set(xlim=(10.6, 32), ylim=(-1.9, 3.4))
    for yy in (2, 0):
        ax.plot(TLIM, [yy, yy], color="#333", linewidth=1.5)
    for temp in range(14, 32):
        ax.plot([temp, temp], [2, 2 - (0.12 if temp % 2 else 0.2)], color="#333", linewidth=1)
    for temp in range(14, 32, 2):
        ax.text(temp, 1.7, f"{temp}", ha="center", va="top", fontsize=11)
    for xv in np.arange(-2, 2.26, 0.25):
        ax.plot([to_temp(xv)] * 2, [0, 0.12 if xv % 0.5 else 0.2], color="#333", linewidth=1)
    for xv in np.arange(-2, 2.1, 0.5):
        ax.text(to_temp(xv), -0.3, m(xv, 1).replace("-", "−"), ha="center", va="top", fontsize=11)
    for temp, xv in ((14, -2), (18, -1), (22, 0), (26, 1), (30, 2)):
        ax.plot([temp, temp], [1.2, 0.35], color="#bbb", linewidth=1, linestyle="--")
    ax.scatter(T, 2.25 + 0.5 * (JIT + 0.03) / 0.06, s=26, c=np.where(Y == 1, C_YES, C_NO), edgecolor="white",
               linewidth=0.5, zorder=4)
    ax.text(10.7, 2, "temperature\n(°C)", ha="left", va="center", fontsize=11.5, weight="bold")
    ax.text(10.7, 0, "$x$", ha="left", va="center", fontsize=16, weight="bold")
    ax.plot([24.6, 24.6], [2, 0], color=C_NET, linewidth=2.5, zorder=5)
    ax.text(24.75, 1.0, "24.6 °C  →  $x = 0.65$", color=C_NET, fontsize=12.5, va="center", weight="bold")
    ax.text(21.9, -1.3, r"$x = (\mathrm{temperature} - 22) \div 4$", fontsize=16, ha="center")
    t.write(
        "Before a neuron sees the temperature, we turn it into a small number around 0. Neurons work best "
        "that way, and the numbers are easier to compute with:",
        M(r"$x = \dfrac{\mathrm{temperature} - 22}{4}$"),
        "• 22 °C becomes $x = 0$, about the middle of the data.",
        "• Every 4 °C warmer adds 1: 26 °C becomes 1, 30 °C becomes 2.",
        "• Colder than 22 °C gives negative numbers: 18 °C becomes −1.",
        "It's like measuring the same thing with a different ruler: the dots stay where they are, only the "
        "numbers change. And from $x$ we can always go back to °C.",
        "From now on the neuron sees $x$. The plots keep showing °C at the bottom and $x$ at the top.",
        Box(r"24.6 °C:  $x = (24.6 - 22) \div 4 = 0.65$"),
    )


@step("Knob 1: the weight")
def page_weight(t):
    ax = temp_axes(t, t.content(slider_rows=1), title="The score of your neuron for every temperature",
                   ylabel="score $z$", ylim=(-9, 9))
    ax.axhline(0, color="#333", linewidth=1)
    line, = ax.plot(TG, np.zeros_like(TG), color=C_NET, linewidth=3)
    guide, = ax.plot([], [], color=C_NET, linestyle=":", linewidth=1.5)
    dot, = ax.plot([], [], "o", color=C_NET, markersize=11, zorder=5)
    info = readout(ax, fontsize=13)

    def update():
        w = t.guess["w"]
        line.set_ydata(w * XG[:, 0])
        z = w * 0.65
        guide.set_data([24.6, 24.6, TLIM[0]], [0, z, z])
        dot.set_data([24.6], [z])
        info.set_text("24.6 °C:  $x = 0.65$\n" rf"$z = w \cdot x = {m(w, 2)} \cdot 0.65 = {m(z, 2)}$")

    t.sliders(["w"], update)
    t.write(
        "A neuron starts with a very simple calculation: it multiplies its input by a number, the "
        "weight $w$.",
        M(r"$z = w \cdot x$"),
        "The result $z$ is called the score. The line shows the score for every temperature. Move the slider:",
        "• $w$ positive: the warmer, the higher the score.",
        "• $w$ negative: the warmer, the lower the score.",
        "• $w = 0$: the temperature doesn't matter at all.",
        "• The bigger the weight (ignoring its sign), the steeper the line: small temperature changes make "
        "a big difference.",
        H("A problem"),
        "At 22 °C, $x = 0$, so the score is always 0, whatever the weight. All lines go through that point. "
        "That's too rigid: the neuron needs a second knob.",
    )


@step("Knob 2: the bias")
def page_bias(t):
    ax = temp_axes(t, t.content(slider_rows=2), title="The score with weight and bias", ylabel="score $z$",
                   ylim=(-9, 9))
    shade = strip(ax, (-9, 9))
    ax.axhline(0, color="#333", linewidth=1)
    line, = ax.plot(TG, np.zeros_like(TG), color=C_NET, linewidth=3)
    cross = ax.axvline(22, color=C_NET, linestyle="--", linewidth=1.5)
    label = ax.text(22, 8.3, "", ha="center", fontsize=11.5, color=C_NET,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))
    info = readout(ax, y=0.2, fontsize=13)

    def update():
        w, b = t.guess["w"], t.guess["b"]
        z = w * XG[:, 0] + b
        line.set_ydata(z)
        shade.set_data((z >= 0)[None, :].astype(float))
        at = switch_temp(w, b)
        inside = at is not None and TLIM[0] < at < TLIM[1]
        cross.set_visible(inside)
        label.set_visible(inside)
        if inside:
            cross.set_xdata([at, at])
            label.set(x=at, text=f"z = 0 at {at:.1f} °C")
        info.set_text(rf"24.6 °C:  $z = {m(w, 2)} \cdot 0.65 + ({m(b, 2)}) = {m(w * 0.65 + b, 2)}$")

    t.sliders(["w", "b"], update)
    t.write(
        "The second knob is simply added to the score: the bias $b$.",
        M(r"$z = w \cdot x + b$"),
        "• $b$ lifts the whole line up or pushes it down, without changing its steepness.",
        "• That moves the point where the score crosses 0 (dashed line).",
        "The crossing point matters: on one side the score is positive (blue band), on the other side "
        "negative (orange band). In the next step the neuron will say 'yes' on one side and 'no' on the other.",
        H("Try it"),
        "Put the crossing point at 18 °C, with positive scores on the warm side.",
        "Tip: 18 °C is $x = -1$. The score there is $w \\cdot (-1) + b$, which is 0 when $b = w$.",
    )


@step("A switch turns the score into a vote")
def page_switch(t):
    ax = temp_axes(t, t.content(slider_rows=2), title="Your neuron's vote (line) and the staff's votes (dots)",
                   ylabel="vote", ylim=(-0.3, 1.3))
    ax.set_yticks([0, 1])
    shade = strip(ax, (-0.3, 1.3))
    plot_votes(ax)
    vote, = ax.plot(TG, np.zeros_like(TG), color=C_NET, linewidth=2.5, zorder=3)
    wrong = rings(ax, s=150)
    info = ax.text(0.02, 0.5, "", transform=ax.transAxes, fontsize=13, va="center", zorder=10,
                   bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bbb"))

    def update():
        w, b = t.guess["w"], t.guess["b"]
        votes = (w * XG[:, 0] + b >= 0).astype(float)
        vote.set_ydata(votes)
        shade.set_data(votes[None, :])
        bad = (w * X1[:, 0] + b >= 0) != Y
        set_points(wrong, T[bad], Y[bad] + JIT[bad])
        info.set_text(f"wrong votes: {bad.sum()} of {n}")

    t.sliders(["w", "b"], update)
    t.write(
        "Now the neuron makes a decision. A switch turns the score into a vote:",
        M(r"$\hat{y} = 1$ (comfortable)  if  $z \geq 0$,   otherwise  $\hat{y} = 0$"),
        r"The little hat on $\hat{y}$ means: the neuron's guess. Plain $y$ is the real vote of the staff.",
        "The purple line is the neuron's vote for every temperature. Red rings mark afternoons where it "
        "disagrees with the staff.",
        H("Try it"),
        f"Find knobs with as few wrong votes as possible. The best possible is {BEST_SWITCH}.",
        "Why so many? This neuron can only say 'warmer than … is comfortable' or 'colder than … is "
        "comfortable'. It can never say 'between 18 and 26 °C'. Keep that in mind.",
    )


@step("A soft switch: the sigmoid")
def page_sigmoid(t):
    ax = style(t.axes(t.content(slider_rows=1)), title="Hard switch vs. soft switch")
    grid = np.linspace(-7, 7, 400)
    ax.plot([-7, 0, 0, 7], [0, 0, 1, 1], color="#999", linewidth=2, linestyle="--", label="hard switch")
    ax.plot(grid, sigmoid(grid), color=C_NET, linewidth=3, label=r"soft switch: sigmoid $\sigma(z)$")
    ax.axhline(0.5, color="#bbb", linewidth=1)
    ax.set(xlim=(-7, 7), ylim=(-0.05, 1.08), xlabel="score $z$", ylabel="output")
    guide, = ax.plot([], [], color=C_NET, linestyle=":", linewidth=1.5)
    dot, = ax.plot([], [], "o", color=C_NET, markersize=11, zorder=5)
    ax.legend(loc="center right", frameon=False, fontsize=11)
    info = readout(ax, fontsize=14)

    def update():
        z = t.guess["z"]
        p = sigmoid(z)
        guide.set_data([z, z, -7], [0, p, p])
        dot.set_data([z], [p])
        info.set_text(rf"$\sigma({m(z, 2)}) = {p:.3f}$" f"\n= {p:.0%} chance of comfortable")

    t.sliders(["z"], update)
    t.write(
        "The hard switch jumps from 0 to 1. It can't say 'probably', and an afternoon just below the "
        "border gets the same 0 as a freezing one.",
        "The sigmoid σ is a soft switch. It bends the score smoothly into a number between 0 and 1:",
        M(r"$\sigma(z) = \dfrac{1}{1 + e^{-z}}$"),
        "You don't need to compute it by hand. Just remember its shape, and move the slider:",
        "• very negative score: close to 0,",
        "• score 0: exactly 0.5, a coin flip,",
        "• very positive score: close to 1.",
        "It never quite reaches 0 or 1, so we can read it as a chance:",
        Box(r"$p = \sigma(z)$ = chance of 'comfortable'"),
    )


@step("The artificial neuron")
def page_neuron(t):
    ax = diagram_axes(t, t.content(slider_rows=2), (0, 12.4), (-0.2, 9))
    yc = 6.9
    node(ax, 1.2, yc, "$x$", color=C_MEAN, size=0.6, fontsize=17)
    node(ax, 5.2, yc, r"$+$", color=C_NET, size=0.6, fontsize=20)
    box(ax, 8.1, yc, 1.4, 1.2, "", color=C_NET)
    sigmoid_glyph(ax, 8.1, yc)
    node(ax, 11.2, yc, "", color=C_NET, size=0.6)
    arrow(ax, (1.8, yc), (4.6, yc), width=2)
    arrow(ax, (5.8, yc), (7.4, yc), width=2)
    arrow(ax, (8.8, yc), (10.6, yc), width=2)
    arrow(ax, (5.2, yc - 1.9), (5.2, yc - 0.6), width=2)
    w_label = ax.text(3.2, yc + 0.45, "", ha="center", fontsize=13, color=C_NET)
    b_label = ax.text(5.2, yc - 2.2, "", ha="center", va="top", fontsize=13, color=C_NET)
    z_label = ax.text(6.6, yc + 0.45, "", ha="center", fontsize=12, color=C_NET)
    x_label = ax.text(1.2, yc - 0.9, "", ha="center", va="top", fontsize=12, color="#555")
    p_text = ax.text(11.2, yc, "", ha="center", va="center", fontsize=13, weight="bold", zorder=5)
    ax.text(11.2, yc + 0.9, "chance $p$", ha="center", fontsize=11, color="#555")
    ax.text(8.1, yc + 0.9, r"$\sigma$", ha="center", fontsize=13, color="#555")
    ax.plot([0, 12.4], [3.4, 3.4], color="#ddd", linewidth=1)
    lines = [ax.text(0.1, yy, "", fontsize=13, va="center") for yy in (2.7, 1.9, 1.1, 0.3)]

    def update():
        w, b, i = t.guess["w"], t.guess["b"], t.state.get("afternoon", EX)
        x = X1[i, 0]
        z = w * x + b
        p = sigmoid(z)
        w_label.set_text(rf"$\times\ w = {m(w, 2)}$")
        b_label.set_text(rf"$b = {m(b, 2)}$")
        z_label.set_text(rf"$z = {m(z, 2)}$")
        x_label.set_text(rf"${m(x, 3)}$")
        p_text.set_text(f"{p:.2f}")
        p_text.set_color(C_YES if p >= 0.5 else C_NO)
        truth = "comfortable" if Y[i] else "not comfortable"
        lines[0].set_text(f"Afternoon {i + 1} of {n}: {T[i]:.1f} °C, the staff said: {truth}")
        lines[1].set_text(rf"1. rescale:  $x = ({T[i]:.1f} - 22) \div 4 = {m(x, 3)}$")
        lines[2].set_text(rf"2. weigh and add:  $z = w \cdot x + b = {m(w, 2)} \cdot ({m(x, 3)}) + ({m(b, 2)}) = {m(z, 2)}$")
        lines[3].set_text(rf"3. squash:  $p = \sigma({m(z, 2)}) = {p:.2f}$,  a {p:.0%} chance of comfortable")

    def next_afternoon():
        t.state["afternoon"] = (t.state.get("afternoon", EX) + 7) % n
        update()

    t.sliders(["w", "b"], update)
    t.button([0.52, 0.035, 0.08, 0.075], "next\nafternoon", next_afternoon)
    t.write(
        H("Put the pieces together"),
        "That's all an artificial neuron does, in three small steps:",
        "1. multiply the input by the weight,",
        "2. add the bias,",
        "3. squash the score with the sigmoid.",
        M(r"$p = \sigma(w \cdot x + b)$"),
        "The idea comes from nerve cells in the brain: they collect signals and fire when the signals get "
        "strong enough. Artificial neurons are much, much simpler.",
        H("Try it"),
        "Click 'next afternoon' to feed the neuron one afternoon after another, and change the knobs. "
        "Watch the numbers flow through the three steps.",
    )


@step("The neuron's answer for every temperature")
def page_s_curve(t):
    ax = temp_axes(t, t.content(slider_rows=2), title="Your neuron's chance of comfortable")
    plot_votes(ax)
    ax.axhline(0.5, color="#bbb", linewidth=1)
    curve, = ax.plot(TG, np.zeros_like(TG), color=C_NET, linewidth=3, zorder=5)
    mid, = ax.plot([], [], "o", color=C_NET, markersize=10, zorder=6)
    label = ax.text(0, 0.56, "", fontsize=11.5, color=C_NET, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.legend(loc="center right", frameon=False)

    def update():
        w, b = t.guess["w"], t.guess["b"]
        curve.set_ydata(sigmoid(w * XG[:, 0] + b))
        at = switch_temp(w, b)
        inside = at is not None and TLIM[0] < at < TLIM[1]
        mid.set_data(([at], [0.5]) if inside else ([], []))
        label.set_visible(inside)
        if inside:
            label.set(x=at + 0.3, text=f"50% at {at:.1f} °C")

    t.sliders(["w", "b"], update)
    t.write(
        "Instead of one afternoon at a time, compute $p$ for every temperature and draw it. You get an "
        "S-shaped curve.",
        "• $w$ sets how steep the S is, and whether it goes up ($w > 0$) or down ($w < 0$).",
        "• $b$ slides the S to the left or right.",
        "• The curve passes 50% where the score is 0, the crossing point of the last steps.",
        "The dots are the real votes: comfortable ones at the top (1), the others at the bottom (0). A good "
        "curve is high where the blue dots are and low where the orange ones are.",
        H("Try it"),
        "Can you make the curve high in the middle and low on both sides?",
    )


@step("One neuron is not enough")
def page_not_enough(t):
    ax = temp_axes(t, t.content(), title="The best a single neuron can do")
    plot_votes(ax)
    ax.axhline(0.5, color="#bbb", linewidth=1)
    best = sigmoid(ONE_W * XG[:, 0] + ONE_B)
    ax.plot(TG, best, color=C_FIT, linewidth=3, label="best single neuron", zorder=5)
    ax.plot(TG, NET1.forward(XG)[1], color="#888", linewidth=2.5, linestyle="--",
            label="what we would like", zorder=5)
    ax.legend(loc="center right", frameon=False)
    t.write(
        "However you turn the two knobs, the S goes up once or down once. It can't go up and then down "
        "again.",
        f"The red curve is the best single neuron, found by the computer. It is nearly flat: between "
        f"{best.min():.0%} and {best.max():.0%} everywhere. Rising would be wrong for the warm afternoons, "
        "falling wrong for the cold ones, so it hardly dares to do either.",
        f"That's roughly the share of comfortable afternoons, {Y.mean():.0%}. The neuron has practically "
        "given up on the temperature!",
        "What we would like is the dashed curve: low when it's cold, high in the middle, and low again when "
        "it's warm. A bump.",
        Box("one neuron = one S-curve\nfor a bump we need more neurons"),
    )


course.part("Neurons working together")


@step("Idea: split the question in two")
def page_split(t):
    left, bottom, width, height = t.content()
    gap = 0.045
    h = (height - 2 * gap) / 3
    h1 = sigmoid(4 * XG[:, 0] + 4)
    h2 = sigmoid(4 * XG[:, 0] - 4)
    panels = [("neuron 1: warm enough?", h1, C_HID[0]), ("neuron 2: too warm?", h2, C_HID[1]),
              ("warm enough, but not too warm:  neuron 1 minus neuron 2", h1 - h2, C_NET)]
    for k, (title, curve, color) in enumerate(panels):
        ax = temp_axes(t, [left, bottom + (2 - k) * (h + gap), width, h], ylabel="", top=False, xlabel=k == 2)
        ax.plot(TG, curve, color=color, linewidth=3)
        ax.fill_between(TG, 0, curve, color=color, alpha=0.12)
        ax.set_yticks([0, 1])
        ax.text(0.01, 0.97, title, transform=ax.transAxes, fontsize=12, weight="bold", color=color, va="top")
        if k < 2:
            ax.set_xticklabels([])
        for temp in (18, 26):
            ax.axvline(temp, color="#999", linestyle=":", linewidth=1)
    t.write(
        "'Is it comfortable?' is too hard for one neuron. Split it into two easy questions, each of which "
        "a single S-curve can answer:",
        "• Neuron 1: is it warm enough? Its S rises at 18 °C.",
        "• Neuron 2: is it too warm? Its S rises at 26 °C.",
        "Now combine the two answers: take 'warm enough' minus 'too warm'.",
        "• Below 18 °C both say 0:  0 − 0 = 0.",
        "• Between 18 and 26 °C:  1 − 0 = 1.",
        "• Above 26 °C both say 1:  1 − 1 = 0.",
        Box("two S-curves, subtracted: a bump!"),
        "In the next steps you build exactly this out of three neurons.",
    )


@step("Two neurons look at the temperature")
def page_hidden(t):
    ax = temp_axes(t, t.content(slider_rows=3), title="The answers of your two neurons", ylabel="answer $h$")
    plot_votes(ax, alpha=0.25, label=False)
    curves = [ax.plot(TG, np.zeros_like(TG), color=color, linewidth=3, label=label)[0]
              for color, label in ((C_HID[0], "$h_1$ (neuron 1)"), (C_HID[1], "$h_2$ (neuron 2)"))]
    marks = [ax.axvline(22, color=color, linestyle="--", linewidth=1.2) for color in C_HID[:2]]
    ax.legend(loc="center right", frameon=False, fontsize=11)
    info = readout(ax, y=0.8, fontsize=12)

    def update():
        g = t.guess
        texts = []
        for k, (curve, mark) in enumerate(zip(curves, marks)):
            w, b = g[f"w{k + 1}"], g[f"b{k + 1}"]
            curve.set_ydata(sigmoid(w * XG[:, 0] + b))
            at = switch_temp(w, b)
            mark.set_visible(at is not None and TLIM[0] < at < TLIM[1])
            if mark.get_visible():
                mark.set_xdata([at, at])
            direction = "rises" if w > 0 else "falls" if w < 0 else "is flat"
            texts.append(f"neuron {k + 1} {direction}" + (f" at {at:.1f} °C" if mark.get_visible() else ""))
        info.set_text("\n".join(texts))

    def show_me():
        t.set_guess(**{key: SUGGEST[key] for key in ("w1", "b1", "w2", "b2")})

    t.sliders(["w1", "b1", "w2", "b2"], update, columns=2)
    t.buttons([("show me", show_me)], y=0.03 + 2 * 0.045, x=0.1, width=0.1, height=0.035)
    t.write(
        "Two neurons, both looking at the same input $x$. Each has its own weight and its own bias. The "
        "small number tells the two apart:",
        M(r"$h_1 = \sigma(w_1 x + b_1)$      $h_2 = \sigma(w_2 x + b_2)$"),
        "Nothing new: each one is the neuron of the last part. We call their answers $h$, you'll see why "
        "in a moment.",
        H("Try it"),
        "Make neuron 1 (green) rise at 18 °C and neuron 2 (red) rise at 26 °C.",
        "Tip: rising means a positive weight. 18 °C is $x = -1$, so choose $b_1 = w_1$. 26 °C is $x = 1$, "
        "so choose $b_2 = -w_2$. A weight of about 4 gives a nice steep S.",
        "Stuck? Click 'show me'.",
    )


@step("A third neuron combines the answers")
def page_output(t):
    left, right = t.split(slider_rows=3, gap=0.04, ratio=0.56)
    ax = temp_axes(t, left, title="The chance your three neurons give")
    plot_votes(ax, alpha=0.6, label=False)
    faint = [ax.plot(TG, np.zeros_like(TG), color=color, linewidth=1.5, alpha=0.6)[0] for color in C_HID[:2]]
    curve, = ax.plot(TG, np.zeros_like(TG), color=C_NET, linewidth=3.5, zorder=5)
    for temp in THREE:
        ax.axvline(temp, color="#999", linestyle=":", linewidth=1.2)
    table_ax = t.axes(right)
    table_ax.axis("off")
    table_ax.set_title("Three afternoons", loc="left", fontsize=11.5, color="#333")
    table = table_ax.text(0, 0.95, "", family="monospace", fontsize=11, va="top", linespacing=1.9)

    def update():
        g = t.guess
        net = net_of(g)
        H_, p = net.forward(XG)
        for line, k in zip(faint, (0, 1)):
            line.set_ydata(H_[:, k])
        curve.set_ydata(p)
        Hs, ps = net.forward(to_x(THREE)[:, None])
        zs = Hs @ net.v + net.c
        rows = ["    " + "".join(f"{temp:>6} °C" for temp in THREE),
                "h1  " + "".join(f"{v:9.2f}" for v in Hs[:, 0]),
                "h2  " + "".join(f"{v:9.2f}" for v in Hs[:, 1]),
                "z   " + "".join(f"{v:9.2f}" for v in zs),
                "p   " + "".join(f"{v:9.2f}" for v in ps)]
        table.set_text("\n".join(rows).replace("-", "−"))

    def show_me():
        t.set_guess(**SUGGEST)

    t.sliders(["v1", "v2", "c"], update)
    t.button([0.52, 0.035, 0.08, 0.06], "show me", show_me)
    t.write(
        "A third neuron takes the two answers $h_1$ and $h_2$ as its inputs. It has a weight for each, "
        "$v_1$ and $v_2$, and its own bias $c$:",
        M(r"$p = \sigma(v_1 h_1 + v_2 h_2 + c)$"),
        "Its answer $p$ is the chance of comfortable. Try $v_1 = 8$, $v_2 = -8$, $c = -4$, or click "
        "'show me'. The table shows why this makes a bump:",
        "• cold: $h_1 \\approx 0$, $h_2 \\approx 0$, so $z \\approx -4$ and $p$ is small.",
        "• nice: $h_1 \\approx 1$, $h_2 \\approx 0$, so $z \\approx 8 - 4 = 4$ and $p$ is large.",
        "• hot: $h_1 \\approx 1$, $h_2 \\approx 1$, so $z \\approx 8 - 8 - 4 = -4$, small again.",
        "A positive weight means: this answer speaks for 'comfortable'. A negative weight: against it.",
    )


@step("Your first neural network")
def page_network(t):
    ax = diagram_axes(t, t.content(), (0, 11.6), (-0.6, 8.2))
    draw_network(ax, t.guess)
    for xx, name in ((1.3, "input layer"), (5.8, "hidden layer"), (10.3, "output layer")):
        ax.text(xx, -0.4, name, ha="center", fontsize=12, weight="bold", color="#555")
    ax.text(1.3, 2.6, "temperature", ha="center", va="top", fontsize=10.5, color="#777")
    ax.text(10.3, 2.6, "chance of\ncomfortable", ha="center", va="top", fontsize=10.5, color="#777")
    inset = temp_axes(t, [0.42, 0.66, 0.18, 0.2], ylabel="", top=False)
    inset.plot(TG, net_of(t.guess).forward(XG)[1], color=C_NET, linewidth=2.5)
    inset.set_xticks([16, 20, 24, 28])
    inset.set_xlabel("°C", fontsize=9)
    inset.tick_params(labelsize=8)
    inset.set_title("p with these knobs", fontsize=10, loc="left")
    t.write(
        "Three neurons wired together: that's a neural network. Its neurons are arranged in layers:",
        "• input layer: just the number $x$, nothing is computed there,",
        "• hidden layer: the neurons with $h_1$ and $h_2$. Hidden, because their answers never show up in "
        "the data: nobody tells them what to compute,",
        "• output layer: $p$, the answer of the network.",
        H("Count the knobs"),
        "Every arrow has a weight, every neuron a bias: 4 weights and 3 biases, 7 knobs in total. They "
        "show the values from your sliders.",
        "Big networks work exactly like this one. They just have more inputs, more neurons and more layers. "
        "The biggest have billions of knobs.",
        Box(r"7 knobs:  $w_1,\ b_1,\ w_2,\ b_2,\ v_1,\ v_2,\ c$"),
    )


@step("Follow one afternoon through the network")
def page_forward(t):
    left, bottom, width, height = t.content()
    ax = diagram_axes(t, [left, bottom + 0.36 * height, width, 0.64 * height], (0, 11.6), (0.2, 7))
    g = t.guess
    texts = draw_network(ax, g)
    lines_ax = t.axes([left, bottom, width, 0.33 * height])
    lines_ax.axis("off")
    lines = [lines_ax.text(0, yy, "", fontsize=12.5, va="center", transform=lines_ax.transAxes)
             for yy in (0.96, 0.77, 0.58, 0.39, 0.2)]
    net = net_of(g)

    def update():
        i = t.state.get("afternoon", EX)
        x = X1[i, 0]
        H_, p = net.forward(np.array([[x]]))
        (h1, h2), p = H_[0], p[0]
        texts["x"].set(text=f"{x:.2f}", fontsize=13)
        texts["h1"].set(text=f"{h1:.2f}", fontsize=13)
        texts["h2"].set(text=f"{h2:.2f}", fontsize=13)
        texts["p"].set(text=f"{p:.2f}", fontsize=13, color=C_YES if p >= 0.5 else C_NO, weight="bold")
        z1, z2 = g["w1"] * x + g["b1"], g["w2"] * x + g["b2"]
        z = g["v1"] * h1 + g["v2"] * h2 + g["c"]
        truth = "comfortable" if Y[i] else "not comfortable"
        lines[0].set_text(f"Afternoon {i + 1} of {n}: {T[i]:.1f} °C  (the staff said: {truth})")
        lines[1].set_text(rf"input:  $x = ({T[i]:.1f} - 22) \div 4 = {m(x, 3)}$")
        lines[2].set_text(rf"hidden:  $h_1 = \sigma({m(g['w1'], 2)} \cdot ({m(x, 3)}) + ({m(g['b1'], 2)})) = "
                          rf"\sigma({m(z1, 2)}) = {h1:.2f}$     $h_2 = \sigma({m(z2, 2)}) = {h2:.2f}$")
        lines[3].set_text(rf"output:  $p = \sigma({m(g['v1'], 2)} \cdot {h1:.2f} + ({m(g['v2'], 2)}) \cdot {h2:.2f} "
                          rf"+ ({m(g['c'], 2)})) = \sigma({m(z, 2)}) = {p:.2f}$")
        lines[4].set_text(f"answer: a {p:.0%} chance of comfortable")

    def next_afternoon():
        t.state["afternoon"] = (t.state.get("afternoon", EX) + 7) % n
        update()

    update()
    t.button([0.06, 0.03, 0.1, 0.05], "next afternoon", next_afternoon)
    t.write(
        "How does the network compute its answer? From left to right, one layer after the other. This is "
        "called the forward pass:",
        "1. Rescale the temperature to $x$.",
        "2. Hidden layer: both neurons compute their answer from $x$.",
        "3. Output layer: the last neuron computes $p$ from $h_1$ and $h_2$.",
        "Every layer only needs the numbers of the layer before. The neurons show their answers; the "
        "calculation is written out underneath.",
        H("Try it"),
        "Click 'next afternoon' and follow the numbers. (The knobs are the ones you set with the sliders. "
        f"If it looks odd, go back to step {no(page_output)} and click 'show me'.)",
    )


@step("Why the squashing in the middle matters")
def page_linear(t):
    left, bottom, width, height = t.content(slider_rows=4)
    top_ax = temp_axes(t, [left, bottom + 0.58 * height, 0.4, 0.42 * height], title="Hidden answers",
                       ylabel="$h$", ylim=(-0.1, 1.1), xlabel=False)
    top_ax.set_xticklabels([])
    hidden = [top_ax.plot(TG, np.zeros_like(TG), color=color, linewidth=2.5)[0] for color in C_HID[:2]]
    ax = temp_axes(t, [left, bottom, 0.4, 0.47 * height], title="Chance of comfortable", top=False)
    plot_votes(ax, alpha=0.5, label=False)
    curve, = ax.plot(TG, np.zeros_like(TG), color=C_NET, linewidth=3, zorder=5)
    use_sigmoid = [t.state.get("squash", True)]

    def update():
        net = net_of(t.guess)
        Z = XG @ net.W.T + net.b
        H_ = sigmoid(Z) if use_sigmoid[0] else Z
        for line, k in zip(hidden, (0, 1)):
            line.set_ydata(H_[:, k])
        top_ax.set_ylim((-0.1, 1.1) if use_sigmoid[0] else (min(-1, H_.min() * 1.1), max(1, H_.max() * 1.1)))
        curve.set_ydata(sigmoid(H_ @ net.v + net.c))

    def choose(k):
        use_sigmoid[0] = t.state["squash"] = k == 0
        update()

    t.radio([0.49, 0.62, 0.11, 0.12], ["with σ", "without σ"], choose, active=0 if use_sigmoid[0] else 1,
            title="hidden neurons")
    t.sliders(KNOBS, update, columns=2)
    t.button([0.49, 0.45, 0.1, 0.05], "show me", lambda: t.set_guess(**SUGGEST))
    t.write(
        "Does the hidden layer really need the sigmoid? What if the hidden neurons passed on their score "
        "$w x + b$ directly?",
        "Switch it off ('without σ') and try to build a bump with the sliders. You can't!",
        "Without the squashing, every hidden answer is a straight line (top). And adding up straight lines "
        "always gives a straight line again:",
        M(r"$v_1(w_1 x + b_1) + v_2(w_2 x + b_2) + c$"),
        M(r"$= (v_1 w_1 + v_2 w_2)\,x + (v_1 b_1 + v_2 b_2 + c)$"),
        "That's just one weight times $x$ plus one bias: the whole network collapses into a single neuron, "
        "and a single neuron makes a single S.",
        Box("the bend in the hidden layer\ngives the network its power"),
        "A function like σ that bends the score is called an activation function.",
    )


course.part("How a network learns")

ETA = 2.0                                # learning rate of all training runs, unless a page says otherwise
PLAIN = dict(w1="w1", b1="b1", w2="w2", b2="b2", v1="v1", v2="v2", c="c")
KNOB_COLORS = [C_HID[0], C_HID[0], C_HID[1], C_HID[1], C_HID[0], C_HID[1], C_NET]


def knob_dict(net):
    """The seven knobs of a one-input network by name, in the order of the sliders."""
    return dict(w1=net.W[0, 0], b1=net.b[0], w2=net.W[1, 0], b2=net.b[1], v1=net.v[0], v2=net.v[1], c=net.c)


def afternoon_penalties(net, X, labels):
    """-ln q for every afternoon, computed from the score so that q never rounds to 0."""
    z = sigmoid(X @ net.W.T + net.b) @ net.v + net.c
    return np.logaddexp(0, z) - labels * z


L_SUGGEST = net_of(SUGGEST).loss(X1, Y)


@step("How wrong is one answer? The penalty")
def page_penalty(t):
    left, right = t.split(gap=0.06, ratio=0.6)
    ax = style(t.axes(left), title=r"Penalty $-\ln q$ for one afternoon")
    qs = np.linspace(0.004, 1, 400)
    ax.plot(qs, -np.log(qs), color="#555", linewidth=2.5)
    ax.set(xlim=(0, 1.02), ylim=(0, 5.6), xlabel="q = chance the network gave to the real vote", ylabel="penalty")
    for q, color in ((0.8, C_YES), (0.2, C_NO)):
        ax.plot([q, q, 0], [0, -np.log(q), -np.log(q)], color=color, linestyle=":", linewidth=1.5)
        ax.plot([q], [-np.log(q)], "o", color=color, markersize=12, zorder=5)
    ax.annotate("sure and right:\nno penalty", (0.99, 0.02), xytext=(0.72, 1.3), fontsize=11,
                arrowprops=dict(arrowstyle="->", color="#555"))
    ax.annotate("sure but wrong:\nhuge penalty", (0.012, 4.4), xytext=(0.12, 4.6), fontsize=11,
                arrowprops=dict(arrowstyle="->", color="#555"))
    tab = t.axes(right)
    tab.axis("off")
    tab.set_title("The network says: 80% comfortable", loc="left", fontsize=11.5, color="#333")
    tab.text(0, 0.92, "staff said comfortable:", fontsize=12, color=C_YES, weight="bold", va="top")
    tab.text(0, 0.85, r"$q = p = 0.8$,  penalty $-\ln 0.8 = 0.22$", fontsize=12.5, va="top")
    tab.text(0, 0.72, "staff said not comfortable:", fontsize=12, color=C_NO, weight="bold", va="top")
    tab.text(0, 0.65, r"$q = 1 - p = 0.2$,  penalty $-\ln 0.2 = 1.61$", fontsize=12.5, va="top")
    rows = ["    q    penalty"] + [f"{q:5.2f}   {abs(np.log(q)):6.2f}" for q in (1, 0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01)]
    tab.text(0, 0.5, "\n".join(rows), family="monospace", fontsize=12, va="top", linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.5", fc="#f7f7f7", ec="#ddd"))
    t.write(
        "So far you turned the knobs by hand and judged the curve by eye. A computer needs a number that "
        "says how wrong the network is. Start with a single afternoon.",
        "Let $q$ be the chance the network gave to what really happened:",
        "• the staff said comfortable: $q = p$,",
        "• the staff said not comfortable: $q = 1 - p$.",
        "The penalty for this afternoon is",
        M(r"$-\ln q$"),
        "• $q = 1$: sure and right. No penalty at all.",
        "• $q = 0.5$: a coin flip. Penalty 0.69.",
        "• $q$ close to 0: sure, but wrong. The penalty shoots up.",
        "You don't have to compute $\\ln$ yourself: just read the curve or the table.",
        Box("good answers cost little,\nconfident mistakes cost a lot"),
    )


@step("The loss: one number for the whole network")
def page_loss(t):
    left, right = t.split(slider_rows=5, gap=0.07)
    ax = temp_axes(t, left, title="Your network, dot size = penalty", top=False)
    curve, = ax.plot(TG, np.zeros_like(TG), color=C_NET, linewidth=3, zorder=5)
    dots = ax.scatter(T, Y + JIT, s=30, c=np.where(Y == 1, C_YES, C_NO), edgecolor="white", linewidth=0.6, zorder=4)
    ax2 = style(t.axes(right), title=r"Penalty $-\ln q$ of every afternoon")
    qs = np.linspace(0.004, 1, 400)
    ax2.plot(qs, -np.log(qs), color="#555", linewidth=2)
    ax2.set(xlim=(0, 1.02), ylim=(0, 5.3), xlabel="q = chance given to the real vote", ylabel="penalty")
    marks = ax2.scatter(np.zeros(n), np.zeros(n), s=40, c=np.where(Y == 1, C_YES, C_NO), edgecolor="white", zorder=4)
    info = readout(ax2, x=0.97, ha="right", fontsize=13)

    def update():
        net = net_of(t.guess)
        curve.set_ydata(net.forward(XG)[1])
        pen = afternoon_penalties(net, X1, Y)
        dots.set_sizes(12 + 80 * np.minimum(pen, 4))
        marks.set_offsets(np.column_stack([np.exp(-pen), np.minimum(pen, 5.2)]))
        info.set_text(f"your loss: L = {pen.mean():.3f}")

    t.sliders(KNOBS, update, columns=2)
    t.buttons([("show me", lambda: t.set_guess(**SUGGEST))], y=0.03 + 4 * 0.045, x=0.1, width=0.1, height=0.035)
    t.write(
        f"Now compute the penalty for all {n} afternoons and take the average. That's the loss $L$:",
        M(r"$L = \dfrac{1}{n} \sum\ (-\ln q_i)$"),
        "One number for the whole network: the lower, the better. The big dots on the left are the "
        "expensive afternoons.",
        "The loss is also called log-loss or cross-entropy. If the network always said 50%, it would be "
        "0.69. 0 is impossible here, because some afternoons with the same temperature got different votes.",
        H("Try it"),
        f"Push the loss down with the seven sliders. 'show me' gives {L_SUGGEST:.3f}. Can you beat that?",
        "Tedious, isn't it? Seven knobs that all depend on each other, and big networks have millions. "
        "Let's teach the computer to do it.",
    )


def one_knob_page(t, title, zoom):
    """Shared layout of the next two steps: a loss curve over one chosen knob, radio buttons, all seven sliders."""
    left, bottom, _, height = t.content(slider_rows=5)
    ax = style(t.axes([left, bottom, 0.42, height]), title=title)
    choice = [t.state.get("knob", 4)]

    def choose(k):
        choice[0] = t.state["knob"] = k
        refresh[0]()

    refresh = [lambda: None]
    t.radio([0.51, bottom + height - 0.34, 0.09, 0.34], [SLIDERS[k][3] for k in KNOBS], choose,
            active=choice[0], title="knob")
    return ax, choice, refresh


@step("One knob at a time")
def page_one_knob(t):
    ax, choice, refresh = one_knob_page(t, "The loss as you turn one knob, the others stay put", zoom=False)
    curve, = ax.plot([], [], color="#555", linewidth=2.5)
    dot, = ax.plot([], [], "o", color=C_NET, markersize=12, zorder=5)
    info = readout(ax, x=0.97, ha="right", fontsize=12)

    def update():
        key = KNOBS[choice[0]]
        lo, hi, *_ = SLIDERS[key]
        values = np.linspace(lo, hi, 161)
        losses = np.array([net_of(t.guess | {key: v}).loss(X1, Y) for v in values])
        top = max(1.0, min(3.5, losses.max() * 1.05))
        now = net_of(t.guess).loss(X1, Y)
        curve.set_data(values, losses)
        dot.set_data([t.guess[key]], [min(now, top)])
        ax.set(xlim=(lo, hi), ylim=(0, top), xlabel=f"knob {SLIDERS[key][3]}", ylabel="loss $L$")
        info.set_text(f"L = {now:.3f}")

    refresh[0] = update
    t.sliders(KNOBS, update, columns=2)
    t.write(
        "Freeze six knobs and turn only one. Then the loss depends on a single number, and we can draw it "
        "as a curve.",
        "Pick a knob on the right and move its slider: the dot runs along the curve. Move one of the other "
        "knobs, and the whole curve changes its shape.",
        "Learning means: walk downhill on this curve. From where the dot is, is downhill to the left or to "
        "the right?",
        "Your eyes see it at once, because the whole curve is drawn. But drawing it took 161 loss "
        "calculations, for a single knob. That's far too expensive for a computer that has to do this "
        "over and over for millions of knobs.",
        Box("we need a cheap way to find downhill"),
    )


@step("Which way is downhill? Nudge the knob")
def page_nudge(t):
    ax, choice, refresh = one_knob_page(t, "Zoomed in around your knob", zoom=True)
    curve, = ax.plot([], [], color="#bbb", linewidth=2.5)
    tangent, = ax.plot([], [], color=C_FIT, linewidth=2)
    dot, = ax.plot([], [], "o", color=C_NET, markersize=12, zorder=5)
    dot2, = ax.plot([], [], "o", color=C_FIT, markersize=7, zorder=6)
    info = readout(ax, family="monospace", fontsize=11)

    def slope_at(g, key):
        return (net_of(g | {key: g[key] + 0.01}).loss(X1, Y) - net_of(g).loss(X1, Y)) / 0.01

    def update():
        key = KNOBS[choice[0]]
        g, v = t.guess, t.guess[key]
        values = np.linspace(v - 1.5, v + 1.5, 121)
        losses = np.array([net_of(g | {key: u}).loss(X1, Y) for u in values])
        L0, L1 = net_of(g).loss(X1, Y), net_of(g | {key: v + 0.01}).loss(X1, Y)
        slope = (L1 - L0) / 0.01
        curve.set_data(values, losses)
        near = np.array([v - 0.8, v + 0.8])
        tangent.set_data(near, L0 + slope * (near - v))
        dot.set_data([v], [L0])
        dot2.set_data([v + 0.01], [L1])
        spread = max(losses.max() - losses.min(), 0.02)
        ax.set(xlim=(v - 1.5, v + 1.5), ylim=(losses.min() - 0.3 * spread, losses.max() + 0.6 * spread),
               xlabel=f"knob {SLIDERS[key][3]}", ylabel="loss $L$")
        name = PLAIN[key]
        info.set_text(f"L({name} = {v:6.2f}) = {L0:.4f}\nL({name} = {v + 0.01:6.2f}) = {L1:.4f}\n"
                      f"slope = {(L1 - L0):+.4f} / 0.01 = {slope:+.3f}\n\n"
                      f"step: {name} <- {v:.2f} - {ETA:.0f} * ({slope:+.3f})\n"
                      f"               = {v - ETA * slope:.2f}")

    def downhill(count):
        key = KNOBS[choice[0]]
        g = dict(t.guess)
        for _ in range(count):
            g[key] = float(np.clip(g[key] - ETA * slope_at(g, key), *SLIDERS[key][:2]))
        t.set_guess(**{key: g[key]})

    refresh[0] = update
    t.sliders(KNOBS, update, columns=2)
    t.buttons([("step downhill", lambda: downhill(1)), ("10 steps", lambda: downhill(10))],
              y=0.03 + 4 * 0.045, x=0.1, width=0.1, height=0.035)
    t.write(
        "Nudge the knob a tiny bit, say by 0.01, and compute the loss once more. The change, divided by "
        "the nudge, is the slope (red line):",
        M(r"slope $= \dfrac{L(\mathrm{knob} + 0.01) - L(\mathrm{knob})}{0.01}$"),
        "• slope negative: to the right the loss goes down. Turn the knob up.",
        "• slope positive: to the right the loss goes up. Turn the knob down.",
        "• slope zero: flat ground, the bottom of the valley.",
        "Both cases in one rule: move the knob against its slope.",
        M(r"knob $\leftarrow$ knob $-\ \eta \cdot$ slope"),
        f"The learning rate η sets the size of the steps, here $\\eta = {ETA:.0f}$. A steep slope gives a big step, "
        "a gentle slope a small one.",
        "Click 'step downhill' a few times. Then pick another knob.",
    )


def new_run(seed):
    net = Net.random(1, 2, seed)
    return dict(net=net, losses=[net.loss(X1, Y)], seed=seed)


def advance(run, X, labels, steps):
    for _ in range(steps):
        run["net"] = run["net"].step(run["net"].backprop(X, labels), ETA)
        run["losses"].append(run["net"].loss(X, labels))


@step("All knobs at once: gradient descent")
def page_descent(t):
    run = t.state.setdefault("gd1", new_run(0))
    left, right = t.split(slider_rows=2, gap=0.07, ratio=0.5)
    ax = temp_axes(t, left, title="The network while it learns", top=False)
    plot_votes(ax, alpha=0.6, label=False)
    hidden = [ax.plot(TG, np.zeros_like(TG), color=color, linewidth=1.5, alpha=0.8)[0] for color in C_HID[:2]]
    curve, = ax.plot(TG, np.zeros_like(TG), color=C_NET, linewidth=3.5, zorder=5)
    info = readout(ax, y=0.66, family="monospace", fontsize=11)

    x, bottom, width, height = right
    ax2 = style(t.axes([x, bottom + 0.58 * height, width, 0.38 * height]), title="Slope of the loss for every knob")
    bars = ax2.bar(range(7), np.zeros(7), color=KNOB_COLORS)
    ax2.axhline(0, color="#333", linewidth=1)
    ax2.set_xticks(range(7), [SLIDERS[k][3] for k in KNOBS], fontsize=12)
    ax3 = style(t.axes([x, bottom, width, 0.4 * height]), title="Loss")
    line, = ax3.plot([], [], color=C_NET, linewidth=2)
    ax3.set(xlabel="step", ylabel="$L$")

    def update():
        net = run["net"]
        H_, p = net.forward(XG)
        for h_line, k in zip(hidden, (0, 1)):
            h_line.set_ydata(H_[:, k])
        curve.set_ydata(p)
        slopes = knob_dict(net.backprop(X1, Y))
        values = np.array([slopes[k] for k in KNOBS])
        for bar, value in zip(bars, values):
            bar.set_height(value)
        scale = max(np.abs(values).max() * 1.15, 1e-4)
        ax2.set_ylim(-scale, scale)
        losses = run["losses"]
        line.set_data(np.arange(len(losses)), losses)
        ax3.set(xlim=(0, max(50, len(losses) - 1)), ylim=(0.38, max(0.75, max(losses) + 0.02)))
        knobs = knob_dict(net)
        info.set_text(f"step {len(losses) - 1}\nL  = {losses[-1]:.4f}\n\n"
                      + "\n".join(f"{PLAIN[k]:<2} = {knobs[k]:+6.2f}" for k in KNOBS))

    def go(steps):
        advance(run, X1, Y, steps)
        update()

    def restart():
        stop()
        run.update(new_run(run["seed"] + 1))
        update()

    update()
    t.buttons([("1 step", lambda: go(1)), ("10 steps", lambda: go(10)), ("100 steps", lambda: go(100))],
              y=0.06, x=0.08)
    stop = t.play_button([0.365, 0.06, 0.085, 0.05], lambda: go(10))
    t.button([0.46, 0.06, 0.12, 0.05], "new random start", restart)
    t.write(
        "Now do that for all seven knobs together, and repeat:",
        "1. Find the slope of the loss for every knob (bars, top right).",
        "2. Move every knob a little against its own slope.",
        "3. Back to 1.",
        "This is gradient descent. Almost every neural network learns this way. The list of all slopes is "
        "called the gradient.",
        H("Try it"),
        "Click 'play' and watch. For a while hardly anything happens. Then suddenly the bump forms, the loss "
        "(bottom right) drops and levels off: the network has reached the bottom of a valley.",
        "The knobs start at random values, and nobody tells the hidden neurons what to do. They find "
        "'too cold' and 'too warm' all by themselves.",
    )


ETAS = [0.1, 0.5, 2.0, 10.0, 30.0]


@cache
def eta_runs():
    """The same start, trained for 3000 steps with every learning rate: (loss every 10 steps, final knobs)."""
    runs = {}
    for eta in ETAS:
        net, history = train(Net.random(1, 2, 0), X1, Y, eta, 3000, every=10)
        runs[eta] = (np.array([h.loss(X1, Y) for h in history]), net)
    return runs


@step("The learning rate")
def page_learning_rate(t):
    runs = eta_runs()
    left, right = t.split(gap=0.07, ratio=0.5)
    ax = style(t.axes(left), title="Loss during training, same start")
    lines = {eta: ax.plot(np.arange(301) * 10, runs[eta][0], linewidth=1.5, color="#bbb")[0] for eta in ETAS}
    ax.set(xlim=(0, 3000), ylim=(0.35, 1.5), xlabel="step", ylabel="loss $L$")
    label = ax.text(2950, 0, "", ha="right", va="bottom", fontsize=13, color=C_NET, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85), zorder=5)
    ax2 = temp_axes(t, [right[0], right[1], right[2], right[3] - 0.18], title="The network after 3000 steps", top=False)
    plot_votes(ax2, alpha=0.6, label=False)
    curve, = ax2.plot(TG, np.zeros_like(TG), color=C_NET, linewidth=3, zorder=5)
    choice = [t.state.get("eta", 2)]

    def update():
        eta = ETAS[choice[0]]
        for e, line in lines.items():
            line.set(color=C_NET if e == eta else "#e2e2e2", linewidth=3 if e == eta else 0.8, zorder=3 if e == eta else 2)
        end = runs[eta][0][-1]
        label.set(text=f"η = {eta:g}", y=min(end, 1.4) + 0.03)
        curve.set_ydata(runs[eta][1].forward(XG)[1])

    def choose(k):
        choice[0] = t.state["eta"] = k
        update()

    update()
    t.radio([right[0], right[1] + right[3] - 0.13, 0.16, 0.13], [f"η = {eta:g}" for eta in ETAS], choose,
            active=choice[0], title="learning rate")
    t.write(
        "The learning rate η decides how far the knobs move in one step. Pick one of five values on the "
        "right. All runs start with the same random knobs.",
        "• η = 0.1: baby steps. After 3000 steps the network has hardly started.",
        "• η = 0.5 and 2: fine. 2 gets there faster.",
        "• η = 10: much too big. The first steps shoot far past the valley and push both hidden neurons "
        "into the flat end of the sigmoid. There the slopes are almost 0, and learning stops for good.",
        "• η = 30: jumps around wildly and never settles.",
        "There is no formula for the best η. People try a few values. Modern training tools also adapt "
        "the step size while they learn.",
    )


@step("Where to start? Not with all knobs at zero")
def page_symmetry(t):
    runs = t.state.setdefault("twins", [dict(net=Net([[0], [0]], [0, 0], [0, 0], 0), losses=[np.log(2)]), new_run(0)])
    left, right = t.split(slider_rows=1.5, gap=0.06)
    panels = []
    for rect, title in ((left, "Start: all seven knobs 0"), (right, "Start: random knobs")):
        ax = temp_axes(t, rect, title=title, top=False)
        plot_votes(ax, alpha=0.5, label=False)
        h1, = ax.plot(TG, np.zeros_like(TG), color=C_HID[0], linewidth=4, alpha=0.7, label="$h_1$")
        h2, = ax.plot(TG, np.zeros_like(TG), color=C_HID[1], linewidth=2, linestyle="--", label="$h_2$")
        curve, = ax.plot(TG, np.zeros_like(TG), color=C_NET, linewidth=3, zorder=5, label="$p$")
        ax.legend(loc="center right", frameon=False)
        panels.append((ax, h1, h2, curve, readout(ax, y=0.66, fontsize=11)))

    def update():
        for run, (ax, h1, h2, curve, info) in zip(runs, panels):
            H_, p = run["net"].forward(XG)
            h1.set_ydata(H_[:, 0])
            h2.set_ydata(H_[:, 1])
            curve.set_ydata(p)
            w = run["net"].W[:, 0]
            info.set_text(f"step {len(run['losses']) - 1},  L = {run['losses'][-1]:.3f}\n"
                          f"w1 = {w[0]:+.2f},  w2 = {w[1]:+.2f}")

    def go(steps):
        for run in runs:
            advance(run, X1, Y, steps)
        update()

    def restart():
        stop()
        runs[:] = [dict(net=Net([[0], [0]], [0, 0], [0, 0], 0), losses=[np.log(2)]), new_run(0)]
        update()

    update()
    t.buttons([("100 steps", lambda: go(100))], y=0.035, x=0.08)
    stop = t.play_button([0.175, 0.035, 0.085, 0.05], lambda: go(20))
    t.button([0.27, 0.035, 0.1, 0.05], "back to start", restart)
    t.write(
        "Why start with random knobs? Try the obvious alternative: all knobs at 0 (left), and click 'play'.",
        "Both hidden neurons start out identical. So they get exactly the same slopes, take exactly the "
        "same steps, and stay twins forever: their curves lie on top of each other.",
        "Two twins are no better than a single neuron. The bump never forms, the loss gets stuck.",
        "Random starting knobs make the hidden neurons different from the very beginning, so each one can "
        "find its own job (right).",
        Box("start with small random knobs\nto break the symmetry"),
    )


@step("Nudging is slow: backpropagation")
def page_why_backprop(t):
    left, right = t.split(gap=0.06, ratio=0.55)
    ax = style(t.axes(left), title="Loss calculations for one step of gradient descent")
    sizes = [("our network", 7), ("17 knobs", 17), ("161 knobs", 161), ("image network", 25_000_000),
             ("language model", 100_000_000_000)]
    xs = np.arange(len(sizes))
    ax.bar(xs - 0.2, [k + 1 for _, k in sizes], width=0.38, color="#bbb", label="nudging every knob")
    ax.bar(xs + 0.2, [3] * len(sizes), width=0.38, color=C_NET, label="backpropagation (about 3)")
    ax.set_yscale("log")
    ax.set_xticks(xs, ["our\nnetwork\n7", "4 hidden\n\n17", "40 hidden\n\n161", "image\nnetwork\n25 million",
                       "language\nmodel\n100 billion"], fontsize=9.5)
    ax.set(ylabel="loss calculations per step", ylim=(1, 1e13))
    ax.set_xlabel("knobs", fontsize=10)
    ax.legend(loc="upper left", frameon=False)

    net = net_of(t.guess)
    by_nudge = net.nudge_slopes(X1, Y)
    names = ["w1", "w2", "b1", "b2", "v1", "v2", "c"]      # the order of Net.knobs()
    slopes = dict(zip(names, by_nudge))
    by_backprop = knob_dict(net.backprop(X1, Y))
    tab = t.axes(right)
    tab.axis("off")
    tab.set_title("Slopes for your knobs, two ways", loc="left", fontsize=11.5, color="#333")
    rows = ["knob    nudging   backprop"] + [f"{PLAIN[k]:<5} {slopes[k]:+9.5f}  {by_backprop[k]:+9.5f}" for k in KNOBS]
    tab.text(0, 0.93, "\n".join(rows).replace("-", "−"), family="monospace", fontsize=12, va="top", linespacing=1.7,
             bbox=dict(boxstyle="round,pad=0.6", fc="#f7f7f7", ec="#ddd"))
    t.write(
        "To find seven slopes by nudging, the loss has to be computed eight times: once as it is, and once "
        "more for every nudged knob. And that for every single step of training.",
        "A network with a billion knobs would need a billion loss calculations per step. Hopeless.",
        H("Backpropagation"),
        "Backpropagation finds all slopes at once. It runs forward through the network once, then backward "
        "once, passing the blame for the loss from layer to layer. Together that costs about as much as "
        "three loss calculations, however many knobs there are.",
        "The last pages already used it behind the scenes. The table shows: same slopes as nudging, just "
        "much cheaper.",
        Box("how it works, step by step:\npython backprop_tutor.py"),
    )


course.part("Two inputs")

(LINE_W1, LINE_W2), LINE_B = fit_neuron(X2, Y)      # one neuron, temperature and humidity
NET2 = train(Net.random(2, 4, 0), X2, Y, ETA, 5000)  # 4 hidden neurons, temperature and humidity
HAND2 = Net([[-6, 0], [6, 3], [0, -6], [0, 6]], [-6, -6.75, -8.4, -7.8], [-8, -8, -8, -8], 4)
WALLS = ["too cold", "too warm", "too dry", "too humid"]
ALARM = LinearSegmentedColormap.from_list("alarm", ["white", "#e05a4f"])



def alarm_name(w1, w2):
    """What the side of a hidden neuron's line with high answers means, from the direction of its weights."""
    big, small = sorted([abs(w1), abs(w2)], reverse=True)
    if small < 0.6 * big:
        return ("too warm" if w1 > 0 else "too cold") if abs(w1) > abs(w2) else ("too humid" if w2 > 0 else "too dry")
    return f"{'warm' if w1 > 0 else 'cold'} and {'humid' if w2 > 0 else 'dry'}"


def contour_50(ax, p, old=None, **kw):
    """The 50% border of a map of chances, replacing the previous one."""
    if old is not None:
        old.remove()
    return ax.contour(GT, GH, p.reshape(GT.shape), levels=[0.5], **(dict(colors=C_FIT, linewidths=2.5) | kw))


@step("Temperature is not the whole story")
def page_humidity(t):
    left, right = t.split(gap=0.07, ratio=0.45)
    ax = temp_axes(t, left, title="The trained network from part 4", top=False)
    plot_votes(ax, label=False)
    ax.plot(TG, NET1.forward(XG)[1], color=C_FIT, linewidth=3, zorder=5)
    bad = (NET1.forward(X1)[1] >= 0.5) != Y
    ring = rings(ax, s=140)
    set_points(ring, T[bad], Y[bad] + JIT[bad])
    ax2 = map_axes(t, right, title="The same afternoons, with the humidity")
    plot_map_points(ax2)
    ring2 = rings(ax2, s=140)
    set_points(ring2, T[bad], HU[bad])
    ax2.legend(loc="upper right", frameon=True, fontsize=9.5)
    t.write(
        f"The network from part 4 still gets {bad.sum()} of {n} afternoons wrong (red rings). Most of them "
        "have perfectly nice temperatures, and still people were unhappy.",
        "The office also has a hygrometer, which measures the relative humidity of the air. Put it on a "
        "second axis (right), and the reason shows: the rings are at very dry or very humid afternoons.",
        "Comfortable afternoons form an island: not too cold, not too warm, not too dry, not too humid.",
        H("A second input"),
        "The humidity gets rescaled just like the temperature, to a small number around 0:",
        M(r"$x_1 = \dfrac{\mathrm{temperature} - 22}{4}$      $x_2 = \dfrac{\mathrm{humidity} - 50}{20}$"),
        "50% becomes 0, 70% becomes 1, and 30% becomes −1.",
    )


@step("One neuron with two inputs draws a straight line")
def page_line2d(t):
    ax = map_axes(t, t.content(slider_rows=3), title="Chance of comfortable from one neuron")
    shade = wash(ax)
    plot_map_points(ax)
    line, = ax.plot([], [], color=C_NET, linewidth=3)
    low, = ax.plot([], [], color=C_NET, linewidth=1.2, linestyle="--")
    high, = ax.plot([], [], color=C_NET, linewidth=1.2, linestyle="--")
    wrong = rings(ax, s=130)
    info = readout(ax, x=0.98, ha="right", fontsize=12)

    def update():
        a1, a2, a0 = t.guess["a1"], t.guess["a2"], t.guess["a0"]
        shade.set_data(sigmoid(GX @ [a1, a2] + a0).reshape(GT.shape))
        line.set_data(*clip_line(a1, a2, a0))
        low.set_data(*clip_line(a1, a2, a0, -np.log(9)))
        high.set_data(*clip_line(a1, a2, a0, np.log(9)))
        bad = (X2 @ [a1, a2] + a0 >= 0) != Y
        set_points(wrong, T[bad], HU[bad])
        info.set_text(f"wrong: {bad.sum()} of {n}")

    t.sliders(["a1", "a2", "a0"], update)
    t.write(
        "A neuron with two inputs has a weight for each of them, and one bias:",
        M(r"$p = \sigma(w_1 x_1 + w_2 x_2 + b)$"),
        "The map shows its answer for every temperature and humidity: blue means comfortable, orange not.",
        "The border where $p = 50\\%$ is where $w_1 x_1 + w_2 x_2 + b = 0$. That's always a straight line "
        "(solid). The dashed lines are 10% and 90%.",
        "• $w_1$ and $w_2$ turn the line,",
        "• $b$ shifts it.",
        H("Try it"),
        "Can you get the blue afternoons on one side and the orange ones on the other?",
        f"No: a straight line can't cut out an island. The best single neuron still gets "
        f"{((X2 @ [LINE_W1, LINE_W2] + LINE_B >= 0) != Y).sum()} afternoons wrong.",
    )


def alarm_map(ax, w1, w2, b, color, s=18):
    """A hidden neuron on the map: red where it rings, its line, and the data faintly."""
    ax.imshow(sigmoid(GX @ [w1, w2] + b).reshape(GT.shape), extent=(*TLIM, *HLIM), origin="lower", aspect="auto",
              cmap=ALARM, vmin=0, vmax=1, alpha=0.6, zorder=0, interpolation="bilinear")
    ax.plot(*clip_line(w1, w2, b), color=color, linewidth=2.5, linestyle="--")
    plot_map_points(ax, s=s, alpha=0.5, label=False)


@step("A hidden neuron is a wall with an alarm")
def page_wall(t):
    ax = map_axes(t, t.content(slider_rows=4), title="One neuron: red where its answer is close to 1")
    shade = ax.imshow(np.zeros(GT.shape), extent=(*TLIM, *HLIM), origin="lower", aspect="auto", cmap=ALARM,
                      vmin=0, vmax=1, alpha=0.6, zorder=0, interpolation="bilinear")
    plot_map_points(ax, alpha=0.6, label=False)
    line, = ax.plot([], [], color=C_NET, linewidth=3, linestyle="--")
    pointer = ax.annotate("", (0, 0), (0, 0), arrowprops=dict(arrowstyle="-|>", color="#b3261e", linewidth=2.5,
                                                              mutation_scale=22), zorder=7)
    label = ax.text(0, 0, "", fontsize=13, weight="bold", color="#b3261e", ha="center", va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

    def update():
        a1, a2, a0 = t.guess["a1"], t.guess["a2"], t.guess["a0"]
        shade.set_data(sigmoid(GX @ [a1, a2] + a0).reshape(GT.shape))
        xs, hs = clip_line(a1, a2, a0)
        line.set_data(xs, hs)
        visible = len(xs) == 2
        pointer.set_visible(visible)
        label.set_visible(visible)
        if visible:
            mid_t, mid_h = np.mean(xs), np.mean(hs)
            du, dv = a1 / 4 * 17, a2 / 20 * 80            # direction of rising answers, in fractions of the plot
            size = np.hypot(du, dv)
            du, dv = du / size, dv / size
            tip = (mid_t + 0.12 * du * 17, mid_h + 0.12 * dv * 80)
            pointer.xy, pointer.xyann = tip, (mid_t, mid_h)
            label.set(x=mid_t + 0.2 * du * 17, y=mid_h + 0.2 * dv * 80, text=alarm_name(a1, a2))

    def preset(k):
        (w1, w2), b = HAND2.W[k], HAND2.b[k]
        t.set_guess(a1=w1, a2=w2, a0=b)

    t.sliders(["a1", "a2", "a0"], update)
    t.buttons([(name, lambda k=k: preset(k)) for k, name in enumerate(WALLS)], y=0.03 + 3 * 0.045, x=0.12,
              width=0.085, height=0.035)
    t.write(
        "In a network, a hidden neuron doesn't have to answer the whole question. It is enough if it "
        "watches one side of its line, like a wall with an alarm: its answer is close to 1 on one side "
        "(red) and close to 0 on the other.",
        "Click the four buttons. Each one sets the knobs for a different alarm:",
        "• too cold: $w_1$ negative, $w_2 = 0$. Only the temperature counts, so the wall stands upright.",
        "• too dry and too humid: $w_1 = 0$. Only the humidity counts, so the wall lies flat.",
        "• too warm: humid air feels warmer, so humidity counts a little too, and the wall is tilted.",
        "The arrow points to the side where the alarm rings. A steeper neuron (bigger weights) gives a "
        "sharper wall.",
    )


@step("Four walls around the island")
def page_four_walls(t):
    left, bottom, width, height = t.content()
    gap_x, gap_y = 0.06, 0.08
    w, h = (width - gap_x) / 2, (height - gap_y) / 2
    for k in range(4):
        row, col = divmod(k, 2)
        rect = [left + col * (w + gap_x), bottom + (1 - row) * (h + gap_y), w, h]
        (w1, w2), b = HAND2.W[k], HAND2.b[k]
        ax = map_axes(t, rect, title=rf"neuron {k + 1}, {WALLS[k]}:  $h_{k + 1} = \sigma({m(w1)}\,x_1 {w2:+g}\,x_2 {b:+g})$")
        alarm_map(ax, w1, w2, b, C_HID[k])
        ax.title.set_fontsize(10.5)
        if row == 0:
            ax.set_xlabel("")
        if col == 1:
            ax.set_ylabel("")
    t.write(
        "Four hidden neurons, four walls. Each one watches one border of the comfortable island:",
        "• neuron 1 rings below 18 °C ($x_1 < -1$),",
        "• neuron 2 rings when it feels warmer than 26.5 °C,",
        "• neuron 3 rings below 22% humidity,",
        "• neuron 4 rings above 76% humidity.",
        "These knobs were set by hand, by someone who knows what comfortable means. Weights of 6 make the "
        "walls fairly sharp. The biases put them in the right place.",
        "Now the output neuron only has to answer one simple question: does any alarm ring?",
    )


def draw_net_241(ax, labels, v_text, c_text, colors=C_HID):
    """A small diagram of the network with 2 inputs, 4 hidden neurons and 1 output."""
    ins, hid, out = [(1, 5.2), (1, 2.8)], [(4, 7.1), (4, 5.0), (4, 3.0), (4, 0.9)], (7.2, 4.0)
    for (xi, yi), name in zip(ins, ("$x_1$", "$x_2$")):
        for xh, yh in hid:
            ax.plot([xi + 0.45, xh - 0.45], [yi, yh], color="#ccc", linewidth=1, zorder=1)
        node(ax, xi, yi, name, color=C_MEAN, size=0.45, fontsize=13)
    for (xh, yh), text, color in zip(hid, labels, colors):
        arrow(ax, (xh + 0.45, yh), (out[0] - 0.5, out[1] + 0.25 * (yh - out[1]) / 3), color=color)
        node(ax, xh, yh, "", color=color, size=0.45)
        ax.text(xh, yh + 0.62, text, ha="center", fontsize=9.5, color=color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
    for (xh, yh), text, color in zip(hid, v_text, colors):
        ax.text((xh + out[0]) / 2 + 0.1, (yh + out[1]) / 2 + 0.25, text, ha="center", fontsize=10.5, color=color,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
    node(ax, *out, "$p$", color=C_NET, size=0.5, fontsize=14)
    ax.text(out[0], out[1] - 0.95, c_text, ha="center", fontsize=11, color=C_NET)


@step("The output neuron: comfortable if no alarm rings")
def page_combine(t):
    left, right = t.split(slider_rows=1, gap=0.03, ratio=0.66)
    ax = map_axes(t, left, title="Chance of comfortable from the whole network")
    shade = wash(ax)
    plot_map_points(ax)
    for k in range(4):
        ax.plot(*clip_line(*HAND2.W[k], HAND2.b[k]), color=C_HID[k], linewidth=1.3, linestyle="--")
    border = [None]
    info = readout(ax, x=0.98, ha="right", fontsize=12)
    diagram = diagram_axes(t, [right[0], right[1] + 0.1, right[2], right[3] - 0.2], (0, 8.2), (-0.8, 8.2))
    c_label = [None]

    def update():
        c = t.guess["c4"]
        net = Net(HAND2.W, HAND2.b, HAND2.v, c)
        p = net.forward(GX)[1]
        shade.set_data(p.reshape(GT.shape))
        border[0] = contour_50(ax, p, border[0])
        bad = (net.forward(X2)[1] >= 0.5) != Y
        info.set_text(f"wrong: {bad.sum()} of {n}")
        if c_label[0] is not None:
            c_label[0].set_text(rf"$c = {m(c, 1)}$")

    draw_net_241(diagram, WALLS, ["$-8$"] * 4, "")
    c_label[0] = diagram.text(7.2, 3.05, "", ha="center", va="top", fontsize=11, color=C_NET)
    t.sliders(["c4"], update)
    t.write(
        "The output neuron listens to all four alarms, each with the weight −8:",
        M(r"$p = \sigma(-8 h_1 - 8 h_2 - 8 h_3 - 8 h_4 + c)$"),
        "With $c = 4$:",
        "• no alarm rings: $z = 4$, so $p = 98\\%$,",
        "• one alarm rings: $z = -8 + 4 = -4$, so $p = 2\\%$.",
        "Comfortable only if no alarm rings: the island appears. Now move the slider:",
        "• $c$ between 0 and 8: the island.",
        "• $c$ above 8: one alarm is forgiven, the island grows arms.",
        "• $c$ below 0: nothing is comfortable.",
        Box("hidden layer: draws lines\noutput neuron: combines them"),
    )


def new_run2(seed):
    net = Net.random(2, 4, seed)
    return dict(net=net, losses=[net.loss(X2, Y)], seed=seed)


@step("Let the network find the walls itself")
def page_train2d(t):
    run = t.state.setdefault("gd2", new_run2(0))
    left, right = t.split(slider_rows=1.5, gap=0.06, ratio=0.62)
    ax = map_axes(t, left, title="The network while it learns")
    shade = wash(ax)
    plot_map_points(ax, label=False)
    walls = [ax.plot([], [], color=color, linewidth=2, linestyle="--")[0] for color in C_HID]
    border = [None]
    info = readout(ax, x=0.98, ha="right", family="monospace", fontsize=11)
    ax2 = style(t.axes([right[0], right[1] + 0.5 * right[3], right[2], right[3] * 0.45]), title="Loss")
    table_ax = t.axes([right[0], right[1], right[2], right[3] * 0.38])
    table_ax.axis("off")
    table = table_ax.text(0, 1, "", family="monospace", fontsize=10.5, va="top", linespacing=1.6)
    line, = ax2.plot([], [], color=C_NET, linewidth=2)
    ax2.set(xlabel="step", ylabel="$L$")

    def update():
        net = run["net"]
        p = net.forward(GX)[1]
        shade.set_data(p.reshape(GT.shape))
        border[0] = contour_50(ax, p, border[0])
        for k, wall in enumerate(walls):
            wall.set_data(*clip_line(*net.W[k], net.b[k]))
        losses = run["losses"]
        line.set_data(np.arange(len(losses)), losses)
        ax2.set(xlim=(0, max(100, len(losses) - 1)), ylim=(0, max(0.8, max(losses) * 1.05)))
        wrong = ((net.forward(X2)[1] >= 0.5) != Y).sum()
        info.set_text(f"step {len(losses) - 1:5d}\nL = {losses[-1]:.3f}\nwrong: {wrong}")
        rows = ["neuron   w1     w2     b      v"] + [
            f"  {k + 1}   {net.W[k, 0]:+6.1f} {net.W[k, 1]:+6.1f} {net.b[k]:+6.1f} {net.v[k]:+6.1f}" for k in range(4)]
        table.set_text("\n".join(rows + [f"output bias c = {net.c:+.1f}"]).replace("-", "−"))

    def go(steps):
        advance(run, X2, Y, steps)
        update()

    def restart():
        stop()
        run.update(new_run2(run["seed"] + 1))
        update()

    update()
    t.buttons([("10 steps", lambda: go(10)), ("100 steps", lambda: go(100)), ("1000 steps", lambda: go(1000))],
              y=0.035, x=0.08)
    stop = t.play_button([0.365, 0.035, 0.085, 0.05], lambda: go(25))
    t.button([0.46, 0.035, 0.12, 0.05], "new random start", restart)
    t.write(
        "Hand-built walls are fine for a toy. A real network has to find its own. This one has 2 inputs, "
        "4 hidden neurons with 2 weights and a bias each, and an output neuron with 4 weights and a bias:",
        M(r"$4 \cdot 3 + 5 = 17$ knobs"),
        f"It learns exactly like before: random start, gradient descent with $\\eta = {ETA:.0f}$. The dashed lines "
        "are the walls of the four hidden neurons, the red line the 50% border of the network.",
        H("Try it"),
        "Click 'play'. The walls wander around and the island appears. After a few thousand steps the "
        "network has found all four walls.",
        "'new random start' begins again with other random knobs. Does it always find the same island?",
    )


@step("What the hidden neurons learned")
def page_learned(t):
    left, bottom, width, height = t.content()
    gap_x, gap_y = 0.06, 0.08
    w, h = (width - gap_x) / 2, (height - gap_y) / 2
    rank = {"too cold": 0, "too warm": 1, "too dry": 2}
    order_ = sorted(range(4), key=lambda j: rank.get(alarm_name(*NET2.W[j]), 3))
    for k, j in enumerate(order_):
        row, col = divmod(k, 2)
        rect = [left + col * (w + gap_x), bottom + (1 - row) * (h + gap_y), w, h]
        (w1, w2), b, v = NET2.W[j], NET2.b[j], NET2.v[j]
        name = alarm_name(w1, w2) if v < 0 else "for comfortable"
        ax = map_axes(t, rect, title=rf"neuron {j + 1}: {name}   ($v_{j + 1} = {m(v, 1)}$)")
        alarm_map(ax, w1, w2, b, C_HID[j])
        ax.title.set_fontsize(10.5)
        if row == 0:
            ax.set_xlabel("")
        if col == 1:
            ax.set_ylabel("")
    t.write(
        f"The network after {5000} steps from the first random start. Each map shows one hidden neuron: "
        "red where its answer is close to 1.",
        "All four weights to the output neuron are negative: every hidden neuron has become an alarm, just "
        "like the hand-built ones.",
        "And it found the same walls: too cold, too dry, too warm, and a tilted wall for warm and humid "
        "air. Nobody told it what comfortable means. It only saw the votes.",
        f"The output bias is $c = {NET2.c:.1f}$: comfortable if no alarm rings.",
        H("Don't expect this every time"),
        "Here the hidden neurons are easy to read. In big networks most hidden neurons do something no human "
        "can put into words, and they still work.",
    )


HIDDEN_CHOICES = [1, 2, 3, 4, 10]


@cache
def hidden_runs():
    runs = {}
    for size in HIDDEN_CHOICES:
        net = train(Net.random(2, size, 0), X2, Y, ETA, 5000)
        runs[size] = (net, net.loss(X2, Y), net.loss(X2T, YT), ((net.forward(X2T)[1] >= 0.5) == YT).mean())
    return runs


@step("How many hidden neurons?")
def page_how_many(t):
    runs = hidden_runs()
    ax = map_axes(t, [0.06, 0.09, 0.4, 0.79], title="The trained network")
    shade = wash(ax)
    plot_map_points(ax, label=False)
    plot_map_points(ax, test=True, label=False, alpha=0.6)
    lines = []
    border = [None]
    info = readout(ax, x=0.98, ha="right", family="monospace", fontsize=11)
    choice = [t.state.get("hidden", 3)]

    def update():
        size = HIDDEN_CHOICES[choice[0]]
        net, train_loss, test_loss, acc = runs[size]
        p = net.forward(GX)[1]
        shade.set_data(p.reshape(GT.shape))
        border[0] = contour_50(ax, p, border[0])
        for line in lines:
            line.remove()
        lines[:] = [ax.plot(*clip_line(*net.W[k], net.b[k]), color="#555", linewidth=1, linestyle="--")[0]
                    for k in range(size)]
        info.set_text(f"{size} hidden, {4 * size + 1} knobs\ntraining loss {train_loss:.3f}\n"
                      f"test loss     {test_loss:.3f}\ntest accuracy {acc:.0%}")

    def choose(k):
        choice[0] = t.state["hidden"] = k
        update()

    update()
    t.radio([0.48, 0.55, 0.12, 0.3], [f"{size} hidden" for size in HIDDEN_CHOICES], choose, active=choice[0],
            title="network")
    t.write(
        "More hidden neurons, more walls. Each network was trained for 5000 steps (squares: test afternoons).",
        "• 1 hidden neuron: one wall. The border is a straight line, like a single neuron.",
        "• 2: two walls, the border is a wedge.",
        "• 3: three walls, a triangle.",
        "• 4: four walls. The island fits well.",
        "• 10: more walls than needed. The border starts to bend around single afternoons.",
        "The numbers show the loss on the training afternoons and on the test afternoons, and the share of "
        "test afternoons on the right side of the border (accuracy).",
        "With 10 neurons the training loss is lowest, but the test loss is worse than with 4. More on that in "
        "two steps.",
    )


@cache
def start_runs():
    """A network with 3 hidden neurons, trained from 12 different random starts."""
    return [train(Net.random(2, 3, seed), X2, Y, ETA, 3000) for seed in range(12)]


@step("Different starts, different results")
def page_starts(t):
    nets = start_runs()
    losses = np.array([net.loss(X2, Y) for net in nets])
    left, bottom, width, height = t.content()
    cols, rows, gap = 4, 3, 0.012
    w, h = (width - (cols - 1) * gap) / cols, (height - (rows - 1) * 0.04) / rows
    for k, (net, loss_) in enumerate(zip(nets, losses)):
        row, col = divmod(k, cols)
        ax = t.axes([left + col * (w + gap), bottom + (rows - 1 - row) * (h + 0.04), w, h])
        good = loss_ < losses.min() + 0.01
        ax.imshow(net.forward(GX)[1].reshape(GT.shape), extent=(*TLIM, *HLIM), origin="lower", aspect="auto",
                  cmap=PROB, vmin=0, vmax=1, alpha=0.5, interpolation="bilinear")
        ax.scatter(T, HU, s=5, c=np.where(Y == 1, C_YES, C_NO))
        ax.set(xticks=[], yticks=[], xlim=TLIM, ylim=HLIM)
        ax.set_title(f"start {k + 1}:  L = {loss_:.3f}", fontsize=10, color="#1b7f3b" if good else C_WRONG,
                     weight="bold" if good else "normal")
        for spine in ax.spines.values():
            spine.set(color="#1b7f3b" if good else "#ddd", linewidth=2 if good else 1)
    n_good = int((losses < losses.min() + 0.01).sum())
    t.write(
        "The same network with 3 hidden neurons, trained 12 times from 12 different random starts, each for "
        "3000 steps.",
        f"{n_good} runs found the triangle (green frame). The others got stuck with a worse shape: one wall "
        "is in a useless place, and the loss is higher.",
        "The reason: the loss of a network is not a simple bowl with one bottom. It's a landscape with "
        "several valleys, and gradient descent walks down into whichever valley is nearest to its start.",
        H("What people do about it"),
        "• train from a few different starts and keep the best,",
        "• use more neurons than strictly needed: with more walls, bad valleys get rare,",
        "• use smarter walking methods than plain gradient descent.",
    )


@cache
def overfit_run():
    """40 hidden neurons trained for a long time: the knobs every 500 steps, and their losses."""
    _, history = train(Net.random(2, 40, 0), X2, Y, ETA, 30000, every=500)
    return history, np.array([h.loss(X2, Y) for h in history]), np.array([h.loss(X2T, YT) for h in history])


@step("Too flexible: overfitting")
def page_overfitting(t):
    history, train_losses, test_losses = overfit_run()
    best = int(np.argmin(test_losses))
    left, right = t.split(slider_rows=1, gap=0.07, ratio=0.52)
    ax = map_axes(t, left, title="40 hidden neurons")
    shade = wash(ax)
    plot_map_points(ax, label=False)
    plot_map_points(ax, test=True, label=False, alpha=0.5, s=28)
    border = [None]
    ax2 = style(t.axes([right[0], right[1] + 0.2, right[2], right[3] - 0.3]), title="Loss during training")
    steps = np.arange(len(history)) * 500
    ax2.plot(steps, train_losses, color=C_YES, linewidth=2.5, label="training afternoons")
    ax2.plot(steps, test_losses, color=C_NO, linewidth=2.5, label="test afternoons")
    ax2.axvline(best * 500, color="#999", linestyle=":", linewidth=1.5)
    ax2.text(best * 500 + 300, 0.95, "lowest test loss", fontsize=10, color="#777")
    ax2.set(xlim=(0, 30000), ylim=(0, 1.1), xlabel="step", ylabel="$L$")
    ax2.legend(loc="center right", frameon=False)
    marker = ax2.axvline(0, color=C_NET, linewidth=2)
    info = readout(ax, x=0.98, ha="right", family="monospace", fontsize=11)

    def update():
        k = int(round(t.guess["snap"] / 500))
        p = history[k].forward(GX)[1]
        shade.set_data(p.reshape(GT.shape))
        border[0] = contour_50(ax, p, border[0])
        marker.set_xdata([k * 500, k * 500])
        info.set_text(f"step {k * 500}\ntraining loss {train_losses[k]:.3f}\ntest loss     {test_losses[k]:.3f}")

    t.sliders(["snap"], update)
    t.write(
        f"A network with 40 hidden neurons has {4 * 40 + 1} knobs. That's enough to learn every single training "
        "afternoon by heart.",
        "Drag the slider through its training:",
        "• The training loss (blue) keeps falling.",
        f"• The test loss (orange) falls at first, but after about {best * 500} steps it rises again.",
        "The network has started to learn the noise: single grumpy votes on nice afternoons. Its map grows "
        "strange bays and islands around them, and new afternoons get worse answers. This is overfitting.",
        H("What helps"),
        "• stop training when the test loss starts rising (in practice with a third set of afternoons),",
        "• fewer neurons, or more data,",
        "• punishing big weights while training.",
    )


@cache
def sklearn_net():
    """The network of nn.py, fitted by scikit-learn."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(4,), activation="logistic", solver="sgd", batch_size=n,
                          learning_rate_init=2.0, momentum=0, alpha=0, max_iter=5000, tol=0, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(rescale(X_train), y_train)
    p = model.predict_proba(X2T)[:, 1]
    return ((p >= 0.5) == YT).mean(), log_loss(p, YT)


@step("The honest test")
def page_test(t):
    ax = map_axes(t, [0.06, 0.09, 0.33, 0.79], title="Test afternoons, never seen in training")
    p_grid = NET2.forward(GX)[1]
    wash(ax).set_data(p_grid.reshape(GT.shape))
    contour_50(ax, p_grid)
    plot_map_points(ax, alpha=0.2, label=False)
    plot_map_points(ax, test=True, s=60)
    new = rescale([[24, 60]])
    H_new, p_new = NET2.forward(new)
    ax.plot([24], [60], "D", color="#222", markersize=10, zorder=6)
    ax.annotate(f"24 °C, 60 %\n→ {p_new[0]:.0%}", (24, 60), xytext=(10, 10), textcoords="offset points",
                fontsize=11, weight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.legend(loc="upper left", fontsize=9, frameon=True)

    p_test = NET2.forward(X2T)[1]
    votes = (p_test >= 0.5).astype(int)
    acc, loss_test = (votes == YT).mean(), log_loss(p_test, YT)
    counts = np.array([[((YT == a) & (votes == b)).sum() for b in (0, 1)] for a in (0, 1)])
    cm = t.axes([0.465, 0.52, 0.12, 0.28])
    cm.imshow(counts, cmap="Greens", vmin=0, vmax=counts.max() * 1.6)
    for a in (0, 1):
        for b in (0, 1):
            cm.text(b, a, counts[a, b], ha="center", va="center", fontsize=18, weight="bold")
    cm.set_xticks([0, 1], ["not", "comf."])
    cm.set_yticks([0, 1], ["not", "comf."])
    cm.set(xlabel="network says", ylabel="staff said")
    cm.set_title("Who is right?", fontsize=11.5, color="#333")

    one_acc = ((X2T @ [LINE_W1, LINE_W2] + LINE_B >= 0) == YT).mean()
    sk_acc, sk_loss = sklearn_net()
    table = t.axes([0.43, 0.1, 0.18, 0.3])
    table.axis("off")
    rows = ["             accuracy  loss",
            f"network      {acc:6.0%}  {loss_test:5.2f}",
            f"one neuron   {one_acc:6.0%}  {log_loss(sigmoid(X2T @ [LINE_W1, LINE_W2] + LINE_B), YT):5.2f}",
            f"nn.py        {sk_acc:6.0%}  {sk_loss:5.2f}"]
    table.text(0, 0.9, "\n".join(rows), family="monospace", fontsize=10.5, va="top", linespacing=1.7,
               bbox=dict(boxstyle="round,pad=0.5", fc="#f7f7f7", ec="#ddd"))
    table.set_title("On the test afternoons", loc="left", fontsize=11.5, color="#333")
    z_new = H_new[0] @ NET2.v + NET2.c
    loud = int(np.argmax(H_new[0]))
    t.write(
        f"Now the {n_test} test afternoons come back. The network has never seen them, so its results are an "
        "honest preview of how it will do in the future.",
        f"Accuracy {acc:.0%}: {int(round(acc * n_test))} of {n_test} test afternoons are on the right side of the "
        f"border. The table shows which mistakes happened. The loss on them is {loss_test:.2f}, close to the "
        f"{NET2.loss(X2, Y):.2f} on the training afternoons: no overfitting.",
        f"A single neuron gets only {one_acc:.0%} right, less than always saying 'not comfortable'. The "
        "scikit-learn network in nn.py, trained the same way, does as well as ours.",
        H("A new afternoon: 24 °C and 60 %"),
        rf"rescale: $x_1 = 0.5$, $x_2 = 0.5$. Hidden layer: $h = ({', '.join(f'{v:.2f}' for v in H_new[0])})$. "
        f"Only the '{alarm_name(*NET2.W[loud])}' alarm rings a little, with {H_new[0][loud]:.2f}.",
        M(rf"$p = \sigma({m(z_new, 2)}) = {p_new[0]:.2f}$"),
    )


course.part("Outlook")


def relu(z):
    return np.maximum(0, z)


@step("Other switches: ReLU")
def page_relu(t):
    left, right = t.split(gap=0.07, ratio=0.4)
    ax = style(t.axes(left), title="Two activation functions")
    grid = np.linspace(-4, 4, 400)
    ax.plot(grid, sigmoid(grid), color="#999", linewidth=2.5, label=r"sigmoid $\sigma(z)$")
    ax.plot(grid, relu(grid), color=C_NET, linewidth=3, label=r"ReLU $= \max(0, z)$")
    ax.set(xlim=(-4, 4), ylim=(-0.5, 4), xlabel="score $z$", ylabel="answer")
    ax.axhline(0, color="#333", linewidth=1)
    ax.legend(loc="upper left", frameon=False)

    x, bottom, width, height = right
    ax2 = temp_axes(t, [x, bottom + 0.55 * height, width, 0.42 * height], title="Hidden neurons with ReLU",
                    ylabel="answer", ylim=(-0.3, 5), top=False, xlabel=False)
    r1, r2 = relu(-4 * XG[:, 0] - 4), relu(4 * XG[:, 0] - 4)
    ax2.plot(TG, r1, color=C_HID[0], linewidth=3, label=r"$h_1 = \mathrm{ReLU}(-4x - 4)$: how much too cold")
    ax2.plot(TG, r2, color=C_HID[1], linewidth=3, label=r"$h_2 = \mathrm{ReLU}(4x - 4)$: how much too warm")
    ax2.legend(loc="upper center", frameon=False, fontsize=9.5)
    ax2.set_xticklabels([])
    ax3 = temp_axes(t, [x, bottom, width, 0.45 * height], title=r"Output $p = \sigma(4 - 3 h_1 - 3 h_2)$", top=False)
    plot_votes(ax3, alpha=0.5, label=False)
    ax3.plot(TG, sigmoid(4 - 3 * r1 - 3 * r2), color=C_NET, linewidth=3)
    t.write(
        "The sigmoid is not the only possible bend. Most modern networks use an even simpler one, the ReLU:",
        M(r"$\mathrm{ReLU}(z) = \max(0,\ z)$"),
        "Negative scores become 0, positive scores pass unchanged. A hinge instead of an S.",
        "A ReLU neuron doesn't ring an alarm, it measures how far past its wall an afternoon is: 'how many "
        "degrees too cold'. The output neuron turns that into a bump again (right).",
        H("Why ReLU?"),
        "• It is extremely cheap to compute.",
        "• For positive scores its slope is always 1. The sigmoid is flat at both ends, and in networks with "
        "many layers its small slopes multiply to almost nothing. backprop_tutor.py shows why that matters.",
        "Everything else stays the same: layers, loss, gradient descent.",
    )


@step("Deeper networks")
def page_deeper(t):
    ax = diagram_axes(t, t.content(), (0, 13), (-1, 9))
    layers = [(1, 2, "input"), (4, 5, "hidden layer 1"), (7, 5, "hidden layer 2"), (10, 3, "hidden layer 3"),
              (12.3, 1, "output")]
    positions = [[(x, 4 + (k - (size - 1) / 2) * 1.6) for k in range(size)] for x, size, _ in layers]
    for a, b in zip(positions, positions[1:]):
        for xa, ya in a:
            for xb, yb in b:
                ax.plot([xa + 0.4, xb - 0.4], [ya, yb], color="#ccc", linewidth=0.8, zorder=1)
    colors = [C_MEAN, C_HID[0], C_HID[1], C_HID[2], C_NET]
    for (x, size, name), pos, color in zip(layers, positions, colors):
        for xx, yy in pos:
            node(ax, xx, yy, "", color=color, size=0.4)
        ax.text(x, -0.6, name, ha="center", fontsize=11, weight="bold", color="#555")
    ax.text(4, 8.4, "simple features", ha="center", fontsize=11, color="#555")
    ax.text(7, 8.4, "combinations", ha="center", fontsize=11, color="#555")
    ax.text(10, 8.4, "combinations of\ncombinations", ha="center", fontsize=11, color="#555")
    t.write(
        "Our network had one hidden layer. Nothing stops us from stacking more: every layer takes the answers "
        "of the layer before as its inputs. That's deep learning.",
        "Each layer can build on what the previous one found:",
        "• In a network that recognizes photos, the first layer finds edges, the next one corners and "
        "circles, then eyes and wheels, and finally faces and cars.",
        "• In a language model the layers build from letters to words to meaning.",
        H("Still the same recipe"),
        "Every neuron: weigh, add, bend. Training: a loss, the slopes of all knobs, a step downhill. Repeat, "
        "often for weeks on thousands of computers.",
        "Finding the slopes through many layers needs backpropagation, the topic of the next course.",
    )


@step("Summary: the whole recipe")
def page_summary(t):
    ax = map_axes(t, t.content(), title="The network with 4 hidden neurons")
    p = NET2.forward(GX)[1]
    wash(ax).set_data(p.reshape(GT.shape))
    contour_50(ax, p)
    ax.plot([], [], color=C_FIT, linewidth=2.5, label="50% border")
    for k in range(4):
        ax.plot(*clip_line(*NET2.W[k], NET2.b[k]), color=C_HID[k], linewidth=1.3, linestyle="--")
    plot_map_points(ax)
    plot_map_points(ax, test=True)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    t.write(
        H("A neural network in 8 steps"),
        "1.  Collect examples with answers, keep some for the test.",
        "2.  Rescale the inputs to small numbers around 0.",
        "3.  A neuron: weigh the inputs, add a bias, bend with an activation function.",
        "4.  Hidden neurons draw lines, the output neuron combines them.",
        "5.  Loss: the average penalty $-\\ln q$ over all examples.",
        "6.  Slopes of the loss for every knob (backpropagation).",
        "7.  Start random, step downhill again and again: gradient descent.",
        "8.  Check on the test data. Watch out for overfitting.",
        H("What comes next"),
        "How backpropagation finds all slopes in one backward pass, from scratch and in small steps:",
        Box("python backprop_tutor.py"),
    )


if __name__ == "__main__":
    course.run()
