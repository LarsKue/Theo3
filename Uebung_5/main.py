
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


def scalar_field(x, y, z):
    l = 1
    a = 1
    return l * (np.log((x - a) ** 2 + y ** 2) - np.log((x + a) ** 2 + y ** 2))


def potential(x, y):
    q = 1
    a = 1
    return q * (1 / np.sqrt((x - a) ** 2 + (y - a) ** 2) + 1 / np.sqrt((x + a) ** 2 + (y + a) ** 2) - 1 / np.sqrt((x - a) ** 2 + (y + a) ** 2) - 1 / np.sqrt((x + a) ** 2 + (y - a) ** 2))


def main(argv: list) -> int:
    fig = plt.figure(figsize=(8, 8))

    ax = fig.gca()

    x = y = np.linspace(-5, 5, 888)
    x, y = np.meshgrid(x, y)
    # phi = scalar_field(x, y, 0)
    phi = potential(x, y)

    levels = np.linspace(0.05, 10, 50)
    levels = sorted(np.append(-levels, levels))

    ax.contour(x, y, phi, levels=levels)

    plt.show()

    fig = plt.figure(figsize=(8, 8))

    ax = fig.gca(projection="3d")

    x = y = np.linspace(-5, 5, 333)

    x, y = np.meshgrid(x, y)

    phi = scalar_field(x, y, 0)

    cs = ax.contour3D(x, y, phi, levels=levels)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("phi")

    plt.show()



    return 0


if __name__ == "__main__":
    main(sys.argv)
