
import sys

import numpy as np
from matplotlib import pyplot as plt


def alpha(i: int) -> float:
    return - (i + 1) / i


def beta(i: int) -> float:
    return 1


def gamma(i: int) -> float:
    return 1 / alpha(i)


def gen_L(N: int) -> np.ndarray:
    result = np.zeros((N, N))
    for i in range(N - 1):
        result[i + 1][i] = 1
        result[i][i] = alpha(i + 1)
    result[N - 1][N - 1] = alpha(N)

    return result


def gen_R(N: int) -> np.ndarray:
    result = np.identity(N)
    for i in range(N - 1):
        result[i][i + 1] = gamma(i + 1)

    return result


def delta_phi(x):
    # return -4 * pi * rho(x)
    return x - 1


def analyt(x, length):
    c = 1 / 2 * length - 1 / 6 * length ** 2
    return 1 / 6 * x ** 3 - 1 / 2 * x ** 2 + c * x


@np.vectorize
def deviation(x, y):
    return abs(x - y) / max(abs(x), abs(y))


def main(argv: list) -> int:
    for N in [10, 50, 200, 500, 1000, 3000]:
        length = 1
        h = length / N

        L = gen_L(N)
        R = gen_R(N)

        x = np.linspace(0, length, N)
        xlit = np.linspace(0, length, 10000)

        dp = delta_phi(x)

        try:
            LI = np.linalg.inv(L)
            RI = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            print("Not Invertible.")
            return -1

        phi = h ** 2 * np.dot(np.dot(RI, LI), dp)
        phi_lit = analyt(xlit, length)

        # for i in range(N):
        #     print(f"phi: {phi[i]:8.4f}    lit: {phi_lit[i]:8.4f}    dev: {deviation(phi[i], phi_lit[i]):.4f}")

        plt.plot(x, phi, label="Näherung")
        plt.plot(xlit, phi_lit, label="Analytische Lösung")
        plt.xlabel("x")
        plt.ylabel(r"$\phi$")
        plt.title(f"N = {N}")
        plt.legend()

        plt.savefig(f"{N}.png")
        # idx = int(0.01 * N)
        # plt.plot(x[idx:-idx], deviation(phi, phi_lit)[idx:-idx])

        plt.show()

    return 0


if __name__ == "__main__":
    main(sys.argv)