"""
kernels.py
----------
Kernel functions for the frequency-domain relaxation spectrum problem.

The Maxwell kernel maps the log-CRS H(s) to the complex modulus G*(w):

    G'(w)  = integral  [ws^2 / (1 + ws^2)] * exp(H(s)) d ln s
    G''(w) = integral  [ws   / (1 + ws^2)] * exp(H(s)) d ln s

getKernMat   : build the prestored weighted kernel matrix (2n x ns)
kernel_prestore : evaluate K * exp(H) [+ G0], given the prestored matrix
kernelD      : Jacobian approximation dK/dH (2n x ns)
"""

import numpy as np


def getKernMat(s: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Build the prestored weighted kernel matrix.

    Generates a (2n x ns) matrix:

        [ ws^2 / (1 + ws^2) ] * hs
        [ ws   / (1 + ws^2) ] * hs

    where hs are the trapezoidal quadrature weights in log(s) space.
    Multiplying by exp(H) then gives the predicted [G' | G''].

    Parameters
    ----------
    s : (ns,) array of relaxation times
    w : (n,)  array of angular frequencies

    Returns
    -------
    kernMat : (2n x ns) array
    """
    ns = len(s)
    hs = np.zeros(ns)

    # trapezoidal weights in log(s) space
    hs[0]      = 0.5 * np.log(s[1] / s[0])
    hs[-1]     = 0.5 * np.log(s[-1] / s[-2])
    hs[1:-1]   = 0.5 * (np.log(s[2:]) - np.log(s[:-2]))

    S, W = np.meshgrid(s, w)
    ws   = S * W
    ws2  = ws ** 2

    return np.vstack((ws2 / (1 + ws2), ws / (1 + ws2))) * hs


def kernel_prestore(
    H: np.ndarray,
    kernMat: np.ndarray,
    G0: float = 0.0,
) -> np.ndarray:
    """
    Evaluate K * exp(H) + G0, using the prestored kernel matrix.

    Parameters
    ----------
    H       : (ns,) log-CRS
    kernMat : (2n x ns) prestored kernel matrix from getKernMat
    G0      : plateau modulus (added to G' only; default 0.0)

    Returns
    -------
    Kh : (2n,) predicted [G' | G'']
    """
    n   = kernMat.shape[0] // 2
    Kh  = np.dot(kernMat, np.exp(H))

    if G0 != 0.0:
        G0v       = np.zeros(2 * n)
        G0v[:n]   = G0
        Kh        = Kh + G0v

    return Kh


def kernelD(H: np.ndarray, kernMat: np.ndarray) -> np.ndarray:
    """
    Jacobian approximation dK/dH, shape (2n x ns).

    Uses the identity dK_i/dH_j ≈ K_ij * exp(H_j), i.e. element-wise
    scaling of kernMat by exp(H).

    Parameters
    ----------
    H       : (ns,) log-CRS
    kernMat : (2n x ns) prestored kernel matrix

    Returns
    -------
    DK : (2n x ns) Jacobian matrix
    """
    n      = kernMat.shape[0] // 2
    ns     = kernMat.shape[1]
    Hsuper = np.dot(np.ones((2 * n, 1)), np.exp(H).reshape(1, ns))
    return kernMat * Hsuper
