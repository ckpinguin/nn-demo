import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# --- 1. Data: office climate at 3 pm (temperature, humidity) -> did the staff find it comfortable? ---
rng = np.random.default_rng(7)           # fixed seed = reproducible
n = 200                                  # afternoons over a year
temp = rng.uniform(15, 30, n).round(1)   # °C
humidity = rng.uniform(15, 85, n).round()   # % relative humidity
felt = temp + 0.1 * (humidity - 50)      # humid air feels warmer
# the "true" comfort zone has four soft walls: not too cold, not too warm, not too dry, not too humid
chance = (sigmoid(2 * (temp - 18)) * sigmoid(2 * (26.5 - felt))
          * sigmoid(0.4 * (humidity - 22)) * sigmoid(0.4 * (76 - humidity)))
y = (rng.random(n) < chance).astype(int)  # 1 = comfortable, 0 = not
X = np.column_stack([temp, humidity])

# --- 2. Split into training and test data (stratify = same share of comfortable afternoons in both) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# --- 3. Rescale: neurons like small numbers around 0 ---
def rescale(X):
    return (X - [22, 50]) / [4, 20]      # 22 °C -> 0, 26 °C -> 1;  50 % -> 0, 70 % -> 1


S_train, S_test = rescale(X_train), rescale(X_test)

# --- 4. Fit the models: one neuron (a straight border) vs. a network with 4 hidden neurons ---
one_neuron = LogisticRegression(C=np.inf).fit(S_train, y_train)

# plain gradient descent on all training afternoons at once: 5000 steps of length 2, no extras
net = MLPClassifier(hidden_layer_sizes=(4,), activation="logistic", solver="sgd", batch_size=len(S_train),
                    learning_rate_init=2.0, momentum=0, alpha=0, max_iter=5000, tol=0, random_state=0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)   # we stop after 5000 steps on purpose
    net.fit(S_train, y_train)

W, v = net.coefs_[0].T, net.coefs_[1][:, 0]    # W: one row per hidden neuron
b, c = net.intercepts_[0], net.intercepts_[1][0]
for j in range(4):
    print(f"h{j + 1} = sigmoid({W[j, 0]:+6.2f} * x1 {W[j, 1]:+6.2f} * x2 {b[j]:+6.2f})")
print("p  = sigmoid(" + " ".join(f"{v[j]:+.2f} * h{j + 1}" for j in range(4)) + f" {c:+.2f})")
print(f"training log loss after {net.n_iter_} steps = {net.loss_:.3f}")

# --- 5. Evaluate on unseen data ---
for name, model in [("one neuron ", one_neuron), ("4 hidden   ", net)]:
    p_test = model.predict_proba(S_test)[:, 1]
    print(f"{name} accuracy = {accuracy_score(y_test, p_test >= 0.5):.3f}   log loss = {log_loss(y_test, p_test):.3f}")
print("confusion matrix of the network (rows: truth 0/1, columns: prediction 0/1)")
print(confusion_matrix(y_test, net.predict(S_test)))

# --- 6. Predict something new ---
new = [[24, 60]]
print(f"24 °C, 60 % -> {net.predict_proba(rescale(np.array(new)))[0, 1]:.0%} chance that it feels comfortable")

# --- 7. Plot ---
grid_t, grid_h = np.meshgrid(np.linspace(14, 31, 300), np.linspace(10, 90, 300))
grid = rescale(np.column_stack([grid_t.ravel(), grid_h.ravel()]))
grid_p = net.predict_proba(grid)[:, 1].reshape(grid_t.shape)

plt.figure(figsize=(9, 7.5))
plt.contourf(grid_t, grid_h, grid_p, levels=np.linspace(0, 1, 11), cmap="RdBu", alpha=0.35)
plt.colorbar(label="chance that it feels comfortable")
plt.contour(grid_t, grid_h, grid_p, levels=[0.5], colors="crimson", linewidths=2)
plt.plot([], [], color="crimson", linewidth=2, label="network (50% border)")

# every hidden neuron switches along a straight line: W[j, 0] * x1 + W[j, 1] * x2 + b[j] = 0
for j, color in enumerate(["teal", "mediumvioletred", "goldenrod", "slategray"]):
    z = (grid @ W[j] + b[j]).reshape(grid_t.shape)
    plt.contour(grid_t, grid_h, z, levels=[0], colors=color, linewidths=1.2, linestyles="--")
    plt.plot([], [], color=color, linewidth=1.2, linestyle="--", label=f"hidden neuron {j + 1}")
line_z = (grid @ one_neuron.coef_[0] + one_neuron.intercept_[0]).reshape(grid_t.shape)
plt.contour(grid_t, grid_h, line_z, levels=[0], colors="black", linewidths=1.5, linestyles=":")
plt.plot([], [], color="black", linewidth=1.5, linestyle=":", label="one neuron (50% line)")

for label, color, name in [(1, "steelblue", "comfortable"), (0, "darkorange", "not comfortable")]:
    plt.scatter(*X_train[y_train == label].T, color=color, edgecolor="white", label=f"training: {name}")
    plt.scatter(*X_test[y_test == label].T, color=color, marker="s", edgecolor="black", label=f"test: {name}")

plt.xlim(14, 31)
plt.ylim(10, 90)
plt.xlabel("temperature (°C)")
plt.ylabel("relative humidity (%)")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8, ncols=3, frameon=False)
plt.title("Neural Network Demo")
plt.tight_layout()
plt.show()
