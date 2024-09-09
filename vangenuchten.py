import numpy as np


def S(h, a, n, **kwargs):
    """
    Saturation as a function of pressure head in the VG model.

    h has dimensions L
    a has dimensions L-1 (units consistent with h)
    n and the result are dimensionless
    """
    h = np.asarray(h)
    a = np.asarray(a)
    n = np.asarray(n)

    m = (n - 1.0) / n

    return (1 + np.absolute(a * h) ** n) ** (-m)


def mc(h, a, n, mcr, mcs, **kwargs):
    """
    Volumetric water content as a function of pressure head in the VG model.

    h has dimensions L
    a has dimensions L-1 (units consistent with h)
    n, mcr, mcs, and the result are dimensionless
    """
    return mcr + (mcs - mcr) * S(h, a, n)


def Kr(S, n, l=0.5, **kwargs):
    """
    Relative hydraulic conductivity as a function of saturation in the VG model.

    S, n, l, and the result are dimensionless
    """
    assert np.all(S <= 1.0), 'Sauration must be <= 1.'

    S = np.asarray(S)
    m = (n - 1.0) / n

    return S**l * (1 - (1 - S ** (1.0 / m)) ** m) ** 2


def dmcdh(h, a, n, mcr, mcs, **kwargs):
    """
    Derivative of the water content over pressure head, as a function of pressure head in the VG model.

    h has dimensions L
    a has dimensions L-1 (units consistent with h)
    n, mcr, nad mcs are dimensionless
    result has dimensions L-1 (units consistent with h)
    """
    return (mcs - mcr) * (n - 1.0) * a**n * h ** (n - 1.0) / (1 + np.absolute(a * h) ** n) ** ((2 * n - 1) / n)
