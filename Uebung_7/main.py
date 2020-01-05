
import sys

import numpy as np
import math
from matplotlib import pyplot as plt
from scipy import special

from scipy.interpolate import griddata

from mpl_toolkits.mplot3d import Axes3D

from scipy.integrate import dblquad

from typing import Callable, Iterable


def flatten(l: Iterable):
    """ Flattens any Iterable """
    for item in l:
        if isinstance(item, Iterable):
            yield from flatten(item)
            continue
        yield item


def density(r, theta, phi):
    # a = e = 1
    return np.sin(theta) ** 2 * r ** 2 * np.exp(-r) / (64 * np.pi)


# def q(l: int, m: int) -> Callable:
#     def qq(r, theta, phi):
#         return np.sqrt(4 * np.pi / (2 * l + 1)) * np.conjugate(special.sph_harm(m, l, theta, phi)) * 1 / (64 * np.pi) \
#                * np.sin(theta) ** 2 * (-0.5 * r ** 2 * special.gammainc(l + 3, r) + r * special.gammainc(l + 4, r)
#                                        - 0.5 * special.gammainc(l + 5, r)) * np.exp(-r)
#     return qq

def test(l, m):
    def y(theta, phi):
        return np.conjugate(special.sph_harm(m, l, theta, phi)) * (4 * np.sqrt(3) * special.sph_harm(0, 0, theta, phi) / 3 - np.sqrt(16 * np.pi / 5) * special.sph_harm(0, 2, theta, phi))
    return dblquad(y, 0, 2 * np.pi, lambda x: 0, lambda x: np.pi)


def ctos(x, y, z):
    """ Cartesian to Spherical Coordinates
        :return: r, theta, phi
                where r is in [0, inf), theta in [0, pi] and phi in [0, 2*pi)
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return r, np.arccos(z / r), np.arctan(y / x)


def stoc(r, theta, phi):
    """ Cartesian to Spherical Coordinates
        :return: x, y, z
        r must be in [0, inf), theta in [0, pi] and phi in [0, 2*pi).
    """
    return r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)





def main(argv: list) -> int:

    for l in range(0, 100, 2):
        m = 0
        rv = test(l, m)[0]
        if abs(rv) > 1e-10:
            print(f"l = {l}, m = {m}:", rv)

    # x = y = z = np.linspace(-5, 5, 10)
    # x, y, z = np.meshgrid(x, y, z)
    #
    # plt.figure(figsize=(9, 9))
    #
    # ax = plt.axes(projection="3d")
    #
    # ax.scatter(x, y, z, c=list(flatten(density(*ctos(x, y, z)))))
    #
    # plt.show()
    #
    # for l in range(2 + 1):
    #     for m in range(-l, l + 1):
    #         plt.figure(figsize=(9, 9))
    #         ax = plt.axes(projection="3d")
    #
    #         c = list(flatten(np.abs(q(l, m)(*ctos(x, y, z)))))
    #
    #         print(c)
    #
    #         ax.scatter(x, y, z, c=c)
    #
    #         plt.title("$| q_{" + str(l) + "," + str(m) + "} |$")
    #         plt.show()



    return 0


if __name__ == "__main__":
    main(sys.argv)
