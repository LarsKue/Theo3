
import sys

import numpy as np
from matplotlib import pyplot as plt

from scipy.integrate import quad

from Uebung_6 import fourier


def a(k, I_0):
    return 0


# def b(k, I_0, T):
#     return 8 * I_0 / (k * np.pi * T ** 2) * ((-1) ** (k + 1) * (np.pi * T + np.pi ** 2 + 1 / (k ** 2)) - 1 / (k ** 2))

# def b(k, I_0, T):
#     return 8 * I_0 / (k ** 3 * np.pi * T ** 2) * (-k ** 2 * np.pi * np.cos(k * np.pi) * (T + np.pi) + 2 * (np.cos(k * np.pi) - 1))


def b(k, I_0):
    return -16 * I_0 / (k ** 3 * np.pi ** 3) * (-1 + np.cos(k * np.pi))


def f(t, n,  I_0, T):
    s = sum([a(k, I_0) * np.cos(k * t) + b(k, I_0) * np.sin(k * np.pi * t / T) for k in range(1, n)])
    return a(0, I_0) / 2 + s


def I_p(t, I_0, T):
    return 4 * I_0 / (T ** 2) * ((T + t) * t * np.heaviside(-t, 1) + (T - t) * t * np.heaviside(t, 1))


def main(argv: list) -> int:
    # for n in range(1, 101, 20):

    n = 80
    t = np.linspace(-2 * np.pi, 2 * np.pi, 4000)

    for T in range(1, 20, 2):
        plt.plot(t, I_p(t, 1, T), color="blue")
        plt.plot(t, f(t, n, 1, T), color="orange")
        plt.show()

    # t = np.linspace(-1, 1, 4000)
    # plt.plot(t, I_p(t, 1, 1))
    #
    # plt.plot(t, fourier.fourier_series(I_p, 300, 1, 1)(t))
    # plt.show()

    return 0


if __name__ == "__main__":
    main(sys.argv)
