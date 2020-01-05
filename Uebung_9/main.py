
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
    # generates the Matrix L
    result = np.zeros((N, N))
    for i in range(N - 1):
        result[i + 1][i] = 1
        result[i][i] = alpha(i + 1)
    result[N - 1][N - 1] = alpha(N)

    return result


def gen_R(N: int) -> np.ndarray:
    # generates the Matrix R (also called U in the paper)
    result = np.identity(N)
    for i in range(N - 1):
        result[i][i + 1] = gamma(i + 1)

    return result


def delta_phi(x):
    # this is the right hand side of the poisson-equation
    # return -4 * pi * rho(x)
    return x - 1


def analyt(x, length):
    # this is the analytical solution of the poisson-equation
    c = 1 / 2 * length - 1 / 6 * length ** 2
    return 1 / 6 * x ** 3 - 1 / 2 * x ** 2 + c * x


@np.vectorize
def deviation(x, y):
    return abs(x - y) / max(abs(x), abs(y))


def main(argv: list) -> int:
    length = 1
    xlit = np.linspace(0, length, 10000)
    phi_lit = analyt(xlit, length)
    # calculate the approximation for different N
    for N in [10, 50, 200, 500, 1000, 3000]:
        h = length / N

        L = gen_L(N)
        R = gen_R(N)

        x = np.linspace(0, length, N)

        dp = delta_phi(x)

        try:
            # invert the L and R matrices, if possible
            LI = np.linalg.inv(L)
            RI = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            # if they aren't invertible, the poisson equation may not be solved using this algorithm
            print("Not Invertible.")
            return -1

        # calculate the approximate solution for phi
        phi = h ** 2 * np.dot(np.dot(RI, LI), dp)

        # write solutions, analytical solutions and deviations to a file
        with open(f"{N}.txt", "w+") as f:
            phi_lit_comp = analyt(x, length)
            for i in range(N):
                f.write(f"phi: {phi[i]:8.4f}    lit: {phi_lit_comp[i]:8.4f}    dev: {deviation(phi[i], phi_lit_comp[i]):.4f}\n")

        # plt.figure(figsize=(10, 8))
        # plt.plot(x, deviation(phi, phi_lit_comp), color="green")
        # plt.xlabel("x")
        # plt.ylabel("Abweichung [%]")
        # plt.title(f"N = {N}")
        # plt.savefig("Deviation500.png")
        # plt.show()
        # return 0

        # plot solution and analytical solution and save as image
        plt.figure(figsize=(10, 8))
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