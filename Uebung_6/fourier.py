
import numpy as np

from scipy.integrate import quad

from typing import Callable


def a(f: Callable, k: int, *args, **kwargs):
    def g(t):
        return f(t, *args, **kwargs) * np.cos(k * t)
    return 1 / np.pi * quad(g, -np.pi, np.pi)[0]


def b(f: Callable, k: int, *args, **kwargs):
    def g(t):
        return f(t, *args, **kwargs) * np.sin(k * t)
    return 1 / np.pi * quad(g, -np.pi, np.pi)[0]


def fourier_series(f: Callable, n: int, *args, **kwargs) -> Callable:
    def fs(t):
        s = sum([a(f, k, *args, **kwargs) * np.cos(k * t) + b(f, k, *args, **kwargs) * np.sin(k * t) for k in range(1, n + 1)])
        return a(f, 0, *args, **kwargs) / 2 + s
    return fs
