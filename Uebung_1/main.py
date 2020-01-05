
import sys
import math
import numpy as np
from matplotlib import pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy as unp
from scipy import constants as consts
from scipy.optimize import curve_fit
from scipy.stats import chi2

from mpl_toolkits.mplot3d import Axes3D


def vector_field(x, y, z):
    e1 = -(x ** 2 + y ** 2) * y
    e2 = (x ** 2 + y ** 2) * x
    e3 = x * z

    return e1, e2, e3


def main(argv: list) -> int:

    fig = plt.figure(figsize=(8, 8))

    ax = fig.gca(projection="3d")

    # grid limits and steps for arrows
    # control arrow density and positions here
    x, y, z = np.meshgrid(
        np.arange(-1, 1, 2 / 8),  # x
        np.arange(-1, 1, 2 / 8),  # y
        np.arange(-1, 1, 2 / 8)   # z
    )

    # directional data for the arrows
    u, v, w = vector_field(x, y, z)

    ax.quiver(x, y, z, u, v, w, length=0.1, normalize=False)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

    return 0


if __name__ == "__main__":
    main(sys.argv)
