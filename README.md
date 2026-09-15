# Neural Network Demo

Two interactive step-by-step courses for beginners that show how a neural network works and how it learns, plus a
short scikit-learn script. The example throughout: an office where the staff vote at 3 pm whether the temperature and
humidity feel comfortable.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.14 and the versions pinned in `requirements.txt`.

## Run

```
python nn.py                 # the whole thing in ~100 lines with scikit-learn, prints results and shows a map
python nn_tutor.py           # course 1: neural networks, step by step (38 steps)
python backprop_tutor.py     # course 2: backpropagation, step by step (31 steps)
```

Take the courses in this order. Use the Back / Next buttons or the arrow keys, and play with the sliders and buttons.
Hover over symbols in the text for an explanation.

Options for both courses:

```
python nn_tutor.py --step 20        # open at step 20
python nn_tutor.py --save pages     # export every step as a PNG into the folder pages
```

## What the courses cover

**nn_tutor.py:** one neuron (weight, bias, sigmoid), hidden neurons working together, loss, slopes and gradient
descent, learning rate and random starts, two inputs, overfitting, the test on unseen data, ReLU and deeper networks.

**backprop_tutor.py:** slopes and the chain rule, one neuron backwards, a small network backwards with real numbers,
all examples at once with matrices and the actual numpy code, gradient checks, vanishing slopes in deep networks.

No maths beyond +, −, × and ÷ is assumed.

## Files

| File | Contents |
| --- | --- |
| `nn.py` | Standalone demo with scikit-learn's `MLPClassifier` |
| `nn_tutor.py` | Course 1 |
| `backprop_tutor.py` | Course 2 |
| `tinynet.py` | The data and a tiny neural network in plain numpy, used by both courses |
| `office.py` | Data as the courses show it, colours and plot helpers |
| `tutorkit.py` | The course window: pages, text panel, sliders, buttons, glossary popups |
