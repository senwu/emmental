import matplotlib.pyplot as plt
import numpy as np


def visualize_dataset(dataset):
    """ Visualizes all labelsets for a given dataset. """
    X = np.array([x.numpy() for x in dataset.X_dict["data"]])
    labelsets = dataset.Y_dict
    for label_name, labels in labelsets.items():
        print(f"Vizualizing {label_name} from {dataset.name}")
        Y = labels.numpy()
        plot_xy(X, Y)


def plot_xy(X, Y, gt=None, c=None):
    Y1_mask = Y == 1
    Y2_mask = Y == 2
    plt.scatter(X[Y1_mask, 0], X[Y1_mask, 1], label="Y=1")
    plt.scatter(X[Y2_mask, 0], X[Y2_mask, 1], label="Y=2")
    plt.legend()
    set_and_show_plot()


def set_and_show_plot(xlim=(-1, 1), ylim=(-1, 1)):
    # assume that all data lies within these (x, y) bounds
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.axes().set_aspect("equal")
    plt.show()
