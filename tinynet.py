"""
The afternoons of nn.py and a tiny neural network in plain numpy, shared by nn_tutor.py and backprop_tutor.py.

The network has one hidden layer of sigmoid neurons and one sigmoid output neuron:

    h = sigmoid(W x + b)        hidden layer: one row of W and one entry of b per hidden neuron
    p = sigmoid(v · h + c)      output neuron: the chance of 'comfortable'
"""
import numpy as np
from sklearn.model_selection import train_test_split


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# --- The data: identical to nn.py ---
rng = np.random.default_rng(7)
n_all = 200
temp = rng.uniform(15, 30, n_all).round(1)
humidity = rng.uniform(15, 85, n_all).round()
felt = temp + 0.1 * (humidity - 50)
chance = (sigmoid(2 * (temp - 18)) * sigmoid(2 * (26.5 - felt))
          * sigmoid(0.4 * (humidity - 22)) * sigmoid(0.4 * (76 - humidity)))
y = (rng.random(n_all) < chance).astype(int)
X = np.column_stack([temp, humidity])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

CENTER, SPREAD = np.array([22.0, 50.0]), np.array([4.0, 20.0])


def rescale(X):
    """Small numbers around 0: 22 °C -> 0, 26 °C -> 1, and 50 % -> 0, 70 % -> 1."""
    return (np.asarray(X, float) - CENTER) / SPREAD


class Net:
    """The knobs of a network with one hidden layer. The slopes of the loss have the same shape, so they are a Net too."""

    def __init__(self, W, b, v, c):
        self.W = np.array(W, float)      # hidden weights: one row per hidden neuron, one column per input
        self.b = np.array(b, float)      # hidden biases: one per hidden neuron
        self.v = np.array(v, float)      # output weights: one per hidden neuron
        self.c = float(c)                # output bias

    @classmethod
    def random(cls, inputs, hidden, seed, spread=1.0):
        rng = np.random.default_rng(seed)
        return cls(rng.normal(0, spread, (hidden, inputs)), rng.normal(0, spread, hidden),
                   rng.normal(0, spread, hidden), 0.0)

    def forward(self, X):
        H = sigmoid(X @ self.W.T + self.b)  # hidden layer, one row per afternoon
        p = sigmoid(H @ self.v + self.c)    # output, one chance per afternoon
        return H, p

    def loss(self, X, y):
        """Average log-loss, written so that it never has to take the log of a rounded 0."""
        z = sigmoid(X @ self.W.T + self.b) @ self.v + self.c
        return np.mean(np.logaddexp(0, z) - y * z)

    def backprop(self, X, y):
        H, p = self.forward(X)
        n = len(y)
        e = p - y                              # blame of the output neuron
        D = np.outer(e, self.v) * H * (1 - H)  # blame of the hidden neurons
        return Net(D.T @ X / n, D.mean(axis=0), H.T @ e / n, e.mean())

    def step(self, slopes, eta):
        """One step of gradient descent: every knob moves against its slope."""
        return Net(self.W - eta * slopes.W, self.b - eta * slopes.b,
                   self.v - eta * slopes.v, self.c - eta * slopes.c)

    def knobs(self):
        """All knobs in one flat list: W row by row, then b, v and c."""
        return np.concatenate([self.W.ravel(), self.b, self.v, [self.c]])

    @classmethod
    def from_knobs(cls, knobs, inputs, hidden):
        k = np.asarray(knobs, float)
        s = hidden * inputs
        return cls(k[:s].reshape(hidden, inputs), k[s:s + hidden], k[s + hidden:s + 2 * hidden], k[-1])

    def nudge_slopes(self, X, y, h=1e-5):
        """The slopes the slow way: nudge every knob a tiny bit up and down and watch the loss."""
        knobs, (hidden, inputs) = self.knobs(), self.W.shape
        slopes = np.zeros_like(knobs)
        for i in range(len(knobs)):
            up, down = knobs.copy(), knobs.copy()
            up[i] += h
            down[i] -= h
            slopes[i] = (Net.from_knobs(up, inputs, hidden).loss(X, y)
                         - Net.from_knobs(down, inputs, hidden).loss(X, y)) / (2 * h)
        return slopes


def train(net, X, y, eta, steps, every=0):
    """Gradient descent. With every > 0 also the list of knobs after every that many steps, starting with net."""
    history = [net]
    for k in range(1, steps + 1):
        net = net.step(net.backprop(X, y), eta)
        if every and k % every == 0:
            history.append(net)
    return (net, history) if every else net
