"""
continuous.py
-------------
Pure numerical routines for extracting the continuous relaxation spectrum
H(s) from G*(w) data via Tikhonov regularization with Bayesian lambda
selection.

Public entry point
------------------
    getContSpec(w, Gexp, wexp, config) -> ContinuousResult

All other functions are internal helpers.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.optimize import least_squares

from .kernels import getKernMat, kernel_prestore, kernelD
from .config import ReSpectConfig


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ContinuousResult:
    s:       np.ndarray              # relaxation time grid (ns,)
    H:       np.ndarray              # log-CRS H(s)          (ns,)
    G_fit:   np.ndarray              # predicted [G'|G''] (2n,)
    G0:      float        = 0.0     # plateau modulus (0 if plateau=False)
    lamC:    float        = 0.0     # optimal lambda used
    lam:     Optional[np.ndarray] = None   # lambda array from lcurve
    rho:     Optional[np.ndarray] = None   # residual norms
    eta:     Optional[np.ndarray] = None   # curvature norms
    logP:    Optional[np.ndarray] = None   # log posterior
    dH:      Optional[np.ndarray] = None   # error band on H (ns,); None if lamC pre-set


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _getAmatrix(ns: int) -> np.ndarray:
    """
    Symmetric matrix A = L^T L for error analysis.
    L is the (ns-2 x ns) second-difference operator.
    """
    nl = ns - 2
    L  = (np.diag(np.ones(ns - 1), 1)
          + np.diag(np.ones(ns - 1), -1)
          + np.diag(-2.0 * np.ones(ns)))
    L  = L[1:nl + 1, :]
    return np.dot(L.T, L)


def _getBmatrix(
    H: np.ndarray,
    kernMat: np.ndarray,
    Gexp: np.ndarray,
    wexp: np.ndarray,
    G0: float = 0.0,
) -> np.ndarray:
    """
    B matrix required for the Bayesian log-evidence calculation.
    B = Jr^T Jr + diag(r^T Jr), where Jr is the weighted Jacobian.
    """
    n  = kernMat.shape[0] // 2
    ns = kernMat.shape[1]

    Kmatrix = np.dot((wexp / Gexp).reshape(2 * n, 1), np.ones((1, ns)))
    Jr      = -kernelD(H, kernMat) * Kmatrix

    r = wexp * (1.0 - kernel_prestore(H, kernMat, G0) / Gexp)

    return np.dot(Jr.T, Jr) + np.diag(np.dot(r.T, Jr))


def _residualLM(
    H: np.ndarray,
    lam: float,
    Gexp: np.ndarray,
    wexp: np.ndarray,
    kernMat: np.ndarray,
) -> np.ndarray:
    """
    Residual vector for the Levenberg-Marquardt solve in getH.

    Layout: [ weighted data misfit (2n) | curvature penalty (nl) ]

    If len(H) > ns, the last element of H is G0 (plateau mode).
    """
    n  = kernMat.shape[0] // 2
    ns = kernMat.shape[1]
    nl = ns - 2
    r  = np.zeros(2 * n + nl)

    if len(H) > ns:          # plateau mode: last entry is G0
        G0       = H[-1]
        H        = H[:-1]
        r[:2*n]  = wexp * (1.0 - kernel_prestore(H, kernMat, G0) / Gexp)
    else:
        r[:2*n]  = wexp * (1.0 - kernel_prestore(H, kernMat) / Gexp)

    r[2*n:] = np.sqrt(lam) * np.diff(H, n=2)
    return r


def _jacobianLM(
    H: np.ndarray,
    lam: float,
    Gexp: np.ndarray,
    wexp: np.ndarray,
    kernMat: np.ndarray,
) -> np.ndarray:
    """
    Jacobian of _residualLM with respect to H (and optionally G0).
    Returns a (2n+nl x ns) matrix, or (2n+nl x ns+1) in plateau mode.
    """
    n  = kernMat.shape[0] // 2
    ns = kernMat.shape[1]
    nl = ns - 2

    L = (np.diag(np.ones(ns - 1), 1)
         + np.diag(np.ones(ns - 1), -1)
         + np.diag(-2.0 * np.ones(ns)))
    L = L[1:nl + 1, :]

    Kmatrix = np.dot((wexp / Gexp).reshape(2 * n, 1), np.ones((1, ns)))

    if len(H) > ns:          # plateau mode
        G0  = H[-1]
        H   = H[:-1]
        Jr  = np.zeros((2 * n + nl, ns + 1))
        Jr[:2*n, :ns]    = -kernelD(H, kernMat) * Kmatrix
        Jr[:n,   ns]     = -wexp[:n] / Gexp[:n]   # dR/dG0 for G' rows only
        Jr[2*n:, :ns]    = np.sqrt(lam) * L
        Jr[2*n:,  ns]    = 0.0
    else:
        Jr = np.zeros((2 * n + nl, ns))
        Jr[:2*n, :ns]    = -kernelD(H, kernMat) * Kmatrix
        Jr[2*n:, :ns]    = np.sqrt(lam) * L

    return Jr


# ---------------------------------------------------------------------------
# Core solvers
# ---------------------------------------------------------------------------

def _getH(
    lam: float,
    Gexp: np.ndarray,
    wexp: np.ndarray,
    H: np.ndarray,
    kernMat: np.ndarray,
    G0: Optional[float] = None,
):
    """
    Solve the regularized least-squares problem for a given lambda:

        min_H  || w * (1 - K[H]/Gexp) ||^2  +  lambda * ||L H||^2

    Uses scipy least_squares (Trust-Region) with analytic Jacobian.

    Parameters
    ----------
    G0 : float or None
        If provided, G0 is appended to H and jointly optimized (plateau mode).

    Returns
    -------
    H_opt : (ns,) optimal H
    G0_opt : float  (only returned when G0 is not None)
    """
    if G0 is not None:
        Hplus   = np.append(H, G0)
        res     = least_squares(
            _residualLM, Hplus, jac=_jacobianLM,
            args=(lam, Gexp, wexp, kernMat)
        )
        return res.x[:-1], res.x[-1]
    else:
        res = least_squares(
            _residualLM, H, jac=_jacobianLM,
            args=(lam, Gexp, wexp, kernMat)
        )
        return res.x


def _InitializeH(
    Gexp: np.ndarray,
    wexp: np.ndarray,
    s: np.ndarray,
    kernMat: np.ndarray,
    G0: Optional[float] = None,
):
    """
    Generate an initial guess for H by solving with a large lambda.
    A single guess suffices because lcurve sweeps from high to low lambda.

    Returns H [, G0] matching the calling convention of _getH.
    """
    H   = -5.0 * np.ones(len(s)) + np.sin(np.pi * s)
    lam = 1e0

    if G0 is not None:
        return _getH(lam, Gexp, wexp, H, kernMat, G0)
    else:
        return _getH(lam, Gexp, wexp, H, kernMat)


def _lcurve(
    Gexp: np.ndarray,
    wexp: np.ndarray,
    Hgs: np.ndarray,
    kernMat: np.ndarray,
    config: ReSpectConfig,
    G0: Optional[float] = None,
):
    """
    Sweep lambda from lam_max down to lam_min, computing at each step:
      - rho : || weighted residual ||
      - eta : || second difference of H ||
      - logP: log Bayesian evidence  log p(lambda)

    Early termination when logP drops more than 18 units below its peak
    (the tail contributes negligibly to the posterior mean).

    Also computes dH: the posterior error band on H, defined as the
    standard deviation of H over all lambda with plam > 0.1.

    Returns
    -------
    lamM    : float       - posterior mean lambda
    lam     : (nkept,)   - lambda values explored
    rho     : (nkept,)
    eta     : (nkept,)
    logP    : (nkept,)   - normalised (max = 0)
    dH      : (ns,)      - error band on H
    """
    ns      = len(Hgs)
    npoints = int(config.lam_density
                  * (np.log10(config.lam_max) - np.log10(config.lam_min)))
    hlam    = (config.lam_max / config.lam_min) ** (1.0 / (npoints - 1))
    lam_arr = config.lam_min * hlam ** np.arange(npoints)

    eta     = np.zeros(npoints)
    rho     = np.zeros(npoints)
    logP    = np.zeros(npoints)
    H       = Hgs.copy()

    logPmax  = -np.inf
    Hlambda  = np.zeros((ns, npoints))

    Amat       = _getAmatrix(ns)
    _, LogDetN = np.linalg.slogdet(Amat)

    # sweep from high lambda to low (cuts compute time: warm-starting works well)
    i_stop = 0
    for i in reversed(range(npoints)):
        lamb = lam_arr[i]

        if G0 is not None:
            H, G0_i  = _getH(lamb, Gexp, wexp, H, kernMat, G0)
            rho[i]   = np.linalg.norm(
                wexp * (1.0 - kernel_prestore(H, kernMat, G0_i) / Gexp)
            )
            Bmat     = _getBmatrix(H, kernMat, Gexp, wexp, G0_i)
        else:
            H        = _getH(lamb, Gexp, wexp, H, kernMat)
            rho[i]   = np.linalg.norm(
                wexp * (1.0 - kernel_prestore(H, kernMat) / Gexp)
            )
            Bmat     = _getBmatrix(H, kernMat, Gexp, wexp)

        eta[i]        = np.linalg.norm(np.diff(H, n=2))
        Hlambda[:, i] = H

        _, LogDetC = np.linalg.slogdet(lamb * Amat + Bmat)
        V          = rho[i] ** 2 + lamb * eta[i] ** 2

        # log evidence with prior exp(-lambda)
        logP[i] = (-V
                   + 0.5 * (LogDetN + ns * np.log(lamb) - LogDetC)
                   - lamb)

        if logP[i] > logPmax:
            logPmax = logP[i]
        elif logP[i] < logPmax - 18:
            i_stop = i
            break

    # truncate to significant lambda range
    lam_arr  = lam_arr[i_stop:]
    logP     = logP[i_stop:]
    eta      = eta[i_stop:]
    rho      = rho[i_stop:]
    Hlambda  = Hlambda[:, i_stop:]
    logP     = logP - np.max(logP)   # normalise so max = 0

    # posterior mean lambda
    plam = np.exp(logP)
    plam = plam / np.sum(plam)
    lamM = np.exp(np.sum(plam * np.log(lam_arr)))

    # smoothness nudge
    if config.SmFacLam > 0:
        lamM = np.exp(
            np.log(lamM)
            + config.SmFacLam * (np.max(np.log(lam_arr)) - np.log(lamM))
        )
    elif config.SmFacLam < 0:
        lamM = np.exp(
            np.log(lamM)
            + config.SmFacLam * (np.log(lamM) - np.min(np.log(lam_arr)))
        )

    # error band: std of H over lambda with plam > 0.1
    Hm   = np.zeros(ns)
    Hm2  = np.zeros(ns)
    cnt  = 0
    for i in range(len(lam_arr)):
        if plam[i] > 0.1:
            Hm  += Hlambda[:, i]
            Hm2 += Hlambda[:, i] ** 2
            cnt += 1
    Hm = Hm / cnt
    dH = np.sqrt(np.maximum(Hm2 / cnt - Hm ** 2, 0.0))

    return lamM, lam_arr, rho, eta, logP, dH


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def getContSpec(
    w: np.ndarray,
    Gexp: np.ndarray,
    wexp: np.ndarray,
    config: ReSpectConfig,
) -> ContinuousResult:
    """
    Extract the continuous relaxation spectrum H(s) from G*(w) data.

    Parameters
    ----------
    w     : (n,)  angular frequencies
    Gexp  : (2n,) [G' | G''] concatenated
    wexp  : (2n,) per-datapoint weights
    config: ReSpectConfig

    Returns
    -------
    ContinuousResult
    """
    n   = len(w)
    ns  = config.ns

    # build relaxation time grid s
    wmin = w[0];  wmax = w[-1]
    fc   = config.freq_end_int
    if fc == 1:
        smin = np.exp(-np.pi / 2) / wmax;  smax = np.exp(np.pi / 2) / wmin
    elif fc == 2:
        smin = 1.0 / wmax;                  smax = 1.0 / wmin
    else:
        smin = np.exp(np.pi / 2) / wmax;   smax = np.exp(-np.pi / 2) / wmin

    hs      = (smax / smin) ** (1.0 / (ns - 1))
    s       = smin * hs ** np.arange(ns)
    kernMat = getKernMat(s, w)

    # initial guess
    G0_guess = float(np.min(Gexp)) if config.plateau else None
    if config.plateau:
        Hgs, G0 = _InitializeH(Gexp, wexp, s, kernMat, G0_guess)
    else:
        Hgs = _InitializeH(Gexp, wexp, s, kernMat)
        G0  = 0.0

    # lambda selection
    lam_arr = rho = eta = logP = dH = None

    if config.lam_C is None:
        if config.plateau:
            lamC, lam_arr, rho, eta, logP, dH = _lcurve(
                Gexp, wexp, Hgs, kernMat, config, G0
            )
        else:
            lamC, lam_arr, rho, eta, logP, dH = _lcurve(
                Gexp, wexp, Hgs, kernMat, config
            )
    else:
        lamC = config.lam_C

    # final spectrum
    if config.plateau:
        H, G0 = _getH(lamC, Gexp, wexp, Hgs, kernMat, G0)
    else:
        H  = _getH(lamC, Gexp, wexp, Hgs, kernMat)
        G0 = 0.0

    # compute G_fit for storage
    G_fit = kernel_prestore(H, kernMat, G0)

    return ContinuousResult(
        s=s,
        H=H,
        G_fit=G_fit,
        G0=G0,
        lamC=lamC,
        lam=lam_arr,
        rho=rho,
        eta=eta,
        logP=logP,
        dH=dH,
    )
