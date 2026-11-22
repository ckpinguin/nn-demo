"""The automated check of both courses and nn.py. Run it from the project folder with: python -m unittest

The courses draw with matplotlib. The tests choose the Agg backend here, before anything imports pyplot,
so that they never open a window and also run on a machine without a display.
"""
import matplotlib

matplotlib.use("Agg")
