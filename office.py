"""
The afternoons of nn.py the way the courses show them: sorted data, colours and plot helpers,
shared by nn_tutor.py and backprop_tutor.py.
"""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from tinynet import X_test, X_train, rescale, y_test, y_train
from tutorkit import style

# --- The data: the afternoons of nn.py, sorted by temperature only for readable plots ---
order = np.argsort(X_train[:, 0], kind="stable")
T, HU, Y = X_train[order, 0], X_train[order, 1], y_train[order]
X2 = rescale(X_train[order])             # what a network with two inputs sees
X1 = X2[:, :1]                           # what a network with one input sees: only the temperature
TT, HT, YT = X_test[:, 0], X_test[:, 1], y_test
X2T = rescale(X_test)
n, n_test = len(Y), len(YT)
JIT = np.random.default_rng(3).uniform(-0.03, 0.03, n)           # spread the dots a little, so they don't hide each other
JIT_T = np.random.default_rng(4).uniform(-0.03, 0.03, n_test)


def to_x(temperature):
    return (np.asarray(temperature, float) - 22) / 4


def to_temp(x):
    return 22 + 4 * np.asarray(x, float)


# --- Grids for curves over the temperature and maps over temperature and humidity ---
TLIM, HLIM = (14, 31), (10, 90)
TG = np.linspace(*TLIM, 400)             # temperatures for curves
XG = to_x(TG)[:, None]
GT, GH = np.meshgrid(np.linspace(*TLIM, 160), np.linspace(*HLIM, 160))
GX = rescale(np.column_stack([GT.ravel(), GH.ravel()]))

# --- Look ---
C_YES, C_NO = "#2f6db3", "#ef8a17"       # comfortable / not comfortable
C_NET, C_FIT, C_MEAN, C_WRONG = "#7b3fa0", "#c62828", "#444444", "#d64545"
C_HID = ["#138d75", "#c2185b", "#b8860b", "#546e7a"]            # hidden neurons 1 to 4
PROB = LinearSegmentedColormap.from_list("prob", [C_NO, "white", C_YES])
PLIM = (-0.08, 1.08)


# --- Plot building blocks ---
def temp_axes(t, rect, title=None, ylabel="chance of comfortable", ylim=PLIM, top=True, xlabel=True):
    """Temperature on the bottom axis, the neuron's input x on top."""
    if top:
        rect = [rect[0], rect[1], rect[2], rect[3] - 0.03]
    ax = style(t.axes(rect), title)
    ax.set(xlim=TLIM, ylim=ylim, ylabel=ylabel)
    ax.set_xticks(range(14, 32, 2 if rect[2] > 0.3 else 4))
    if xlabel:
        ax.set_xlabel("temperature (°C)")
    if top:
        top_axis = ax.secondary_xaxis("top", functions=(to_x, to_temp))
        top_axis.set_xticks(np.arange(-2, 2.5, 0.5 if rect[2] > 0.4 else 1))
        top_axis.tick_params(labelsize=9, colors="#777")
        ax.text(1.005, 1.0, "x", transform=ax.transAxes, fontsize=10, color="#777", va="bottom")
    return ax


def plot_votes(ax, alpha=1.0, s=34, label=True):
    for vote, color, name in ((1, C_YES, "comfortable"), (0, C_NO, "not comfortable")):
        k = Y == vote
        ax.scatter(T[k], Y[k] + JIT[k], s=s, color=color, edgecolor="white", linewidth=0.6, alpha=alpha, zorder=4,
                   label=f"{name} ($y = {vote}$)" if label else None)


def plot_test_votes(ax, alpha=1.0, s=40):
    for vote, color, name in ((1, C_YES, "test: comfortable"), (0, C_NO, "test: not comfortable")):
        k = YT == vote
        ax.scatter(TT[k], YT[k] + JIT_T[k], s=s, color=color, marker="s", edgecolor="black", linewidth=0.6,
                   alpha=alpha, zorder=4, label=name)


def map_axes(t, rect, title=None):
    ax = style(t.axes(rect), title)
    ax.set(xlim=TLIM, ylim=HLIM, xlabel="temperature (°C)", ylabel="relative humidity (%)")
    ax.set_xticks(range(14, 32, 2 if rect[2] > 0.3 else 4))
    return ax


def plot_map_points(ax, s=34, alpha=1.0, label=True, test=False):
    for vote, color, name in ((1, C_YES, "comfortable"), (0, C_NO, "not comfortable")):
        if test:
            k = YT == vote
            ax.scatter(TT[k], HT[k], s=s, color=color, marker="s", edgecolor="black", linewidth=0.7, alpha=alpha,
                       zorder=4, label=f"test: {name}" if label else None)
        else:
            k = Y == vote
            ax.scatter(T[k], HU[k], s=s, color=color, edgecolor="white", linewidth=0.6, alpha=alpha, zorder=4,
                       label=name if label else None)


def wash(ax, alpha=0.45, cmap=PROB):
    """A colour wash over the map, filled by an update function: orange = 0, white = 0.5, blue = 1."""
    return ax.imshow(np.full(GT.shape, 0.5), extent=(*TLIM, *HLIM), origin="lower", aspect="auto",
                     cmap=cmap, vmin=0, vmax=1, alpha=alpha, zorder=0, interpolation="bilinear")


def strip(ax, ylim, alpha=0.18):
    """A colour band along the temperature axis, filled by an update function."""
    return ax.imshow(np.full((1, len(TG)), 0.5), extent=(*TLIM, *ylim), origin="lower", aspect="auto",
                     cmap=PROB, vmin=0, vmax=1, alpha=alpha, zorder=0)


def clip_line(w1, w2, b, level=0.0):
    """End points of the line w1*x1 + w2*x2 + b = level inside the map, in °C and %."""
    A, B = w1 / 4, w2 / 20
    C = b - level - 22 * A - 50 * B      # A*temperature + B*humidity + C = 0
    points = set()
    if abs(B) > 1e-12:
        points |= {(u, -(A * u + C) / B) for u in TLIM if HLIM[0] <= -(A * u + C) / B <= HLIM[1]}
    if abs(A) > 1e-12:
        points |= {(-(B * h + C) / A, h) for h in HLIM if TLIM[0] <= -(B * h + C) / A <= TLIM[1]}
    if len(points) < 2:
        return [], []
    (ta, ha), (tb, hb) = min(points), max(points)
    return [ta, tb], [ha, hb]


def switch_temp(w, b):
    """The temperature where w*x + b = 0, or None."""
    return None if abs(w) < 1e-9 else float(to_temp(-b / w))
