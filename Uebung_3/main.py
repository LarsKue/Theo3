
import sys
import math
import numpy as np
from matplotlib import pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy as unp
from scipy import constants as consts
from scipy.optimize import curve_fit
from scipy.stats import chi2


def potential(r, q, alpha):
    return q * (1 + alpha * r / 2) * np.exp(-1 * alpha * r) / r


def main(argv: list) -> int:

    qs = range(-5, 5 + 1)
    qs = [1, 2, 3]
    alphas = range(-5, 5 + 1)
    alphas = range(0, 3)

    x = np.linspace(0.001, 10, 10000)

    for q in qs:
        for alpha in alphas:
            plt.plot(x, potential(x, q, alpha), label=r"$q={}, \alpha={}$".format(q, alpha))

    plt.xscale("log")
    plt.yscale("log")

    plt.ylabel(r"\phi")
    plt.xlabel("r")
    plt.legend()
    plt.show()

    return 0


if __name__ == "__main__":
    main(sys.argv)
