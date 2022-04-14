import numpy as np
from matplotlib.colors import ListedColormap
from pylab import *
from skimage import measure


def confusion_matrix(y_true, y_pred):
    m = 2 * y_true + y_pred
    TP = np.sum(m == 3)
    TN = np.sum(m == 0)
    FP = np.sum(m == 1)
    FN = np.sum(m == 2)
    N = len(m.ravel())
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN, "N": N}


def dice(y_true, y_pred):
    inter = np.sum(y_true * y_pred)
    union = np.sum(y_true + y_pred)
    return 2 * inter / union if union > 0 else None


def n_positive(y):
    y[y > 0] = 1
    return np.sum(y) / len(y.ravel())


def n_components(y):
    return np.max(measure.label(y))


def color_code(y_true, y_pred):
    m = 2 * y_true + y_pred

    aa = m.astype(int)
    plt.rcParams["figure.facecolor"] = "white"
    cmap = cmap = ListedColormap(["black", "green", "red", "white"])
    fig = plt.figure(figsize=(25, 25))
    plt.imshow(aa, cmap=cmap, vmin=np.min(aa.ravel()), vmax=np.max(aa.ravel()))
    cbar = plt.colorbar(fraction=0.026, pad=0.04)
    cbar.set_ticks(np.unique(aa))
    cbar.set_ticklabels(["TN", "FP", "FN", "TP"])
    plt.close()
    return fig


def precision(cm):
    return cm["TP"] / (cm["TP"] + cm["FP"])


def recall(cm):
    return cm["TP"] / (cm["TP"] + cm["FN"])
