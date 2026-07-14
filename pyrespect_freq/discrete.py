"""
discrete.py
-----------
Pure numerical routines for extracting the discrete relaxation spectrum
(Maxwell modes g_i, tau_i) from the continuous spectrum H(s).

Public entry point
------------------
    getDiscSpec(w, Gexp, wexp, cont, config) -> DiscreteResult
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import nnls, minimize, least_squares

from .continuous import ContinuousResult
from .config import ReSpectConfig


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DiscreteResult:
    g:      np.ndarray            # mode weights g_i
    tau:    np.ndarray            # relaxation times tau_i
    N:      int                   # number of modes
    G_fit:  np.ndarray            = None  # predicted [G'|G''] (2n,)
    G0:     float                 = 0.0  # plateau modulus (0 if plateau=False)
    error:  float                 = 0.0  # residual error of best fit

    # stored for save(which="full")
    wtBase:  Optional[np.ndarray] = None
    AICbst:  Optional[np.ndarray] = None
    nzNbst:  Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nnLLS(
    w: np.ndarray,
    tau: np.ndarray,
    Gexp: np.ndarray,
    wexp: np.ndarray,
    isPlateau: bool,
):
    """
    Solve the weighted non-negative least-squares problem for mode weights g.

    Builds the Maxwell kernel K (2n x N_modes), optionally appending a
    column for G0 when isPlateau=True, then calls scipy.optimize.nnls.

    Returns
    -------
    g       : (N_modes,) or (N_modes+1,) with G0 appended if isPlateau
    error   : scalar weighted residual
    condKp  : condition number of the weighted kernel
    """
    n    = len(Gexp) // 2
    ntau = len(tau)

    S, W = np.meshgrid(tau, w)
    ws   = S * W
    ws2  = ws ** 2
    K    = np.vstack((ws2 / (1 + ws2), ws / (1 + ws2)))   # (2n x ntau)

    if isPlateau:
        K = np.hstack((K, np.ones((len(Gexp), 1))))
        K[n:, ntau] = 0.0          # G'' has no G0 contribution

    Kp     = np.dot(np.diag(wexp / Gexp), K)
    condKp = np.linalg.cond(Kp)
    g      = nnls(Kp, wexp, maxiter=100000)[0]

    GstM  = np.dot(K, g)
    error = np.sum((wexp * (GstM / Gexp - 1.0)) ** 2)

    return g, error, condKp


def _MaxwellModes(
    z: np.ndarray,
    w: np.ndarray,
    Gexp: np.ndarray,
    wexp: np.ndarray,
    isPlateau: bool,
):
    """
    Given log(tau) positions z, solve for weights g via NNLS and prune
    negligible or out-of-window modes.

    Returns g, tau, error, condKp.
    """
    tau = np.exp(z)

    g, error, condKp = _nnLLS(w, tau, Gexp, wexp, isPlateau)

    # remove runaway modes far outside the frequency window
    cond  = max(w) * min(tau) < 0.02 or min(w) * max(tau) > 50.0
    izero = np.where(np.atleast_1d(cond))
    tau   = np.delete(tau, izero);
    g     = np.delete(g, izero)


    # prune negligible weights
    g_ref = g[:-1] if isPlateau else g
    izero = np.where(g_ref / np.max(g_ref) < 1e-8)
    tau   = np.delete(tau, izero)
    g     = np.delete(g,   izero)

    return g, tau, error, condKp


def _GetWeights(
    H: np.ndarray,
    w: np.ndarray,
    s: np.ndarray,
    wb: float,
) -> np.ndarray:
    """
    Compute placement weights for the discrete modes.

    Each mode's weight reflects its fractional contribution to G*(w),
    mixed with a uniform baseline wb to avoid over-concentration.

    Returns wt : (ns,) weight distribution over log(s).
    """
    ns = len(s)
    n  = len(w)

    hs = np.zeros(ns)
    hs[0]    = 0.5 * np.log(s[1] / s[0])
    hs[-1]   = 0.5 * np.log(s[-1] / s[-2])
    hs[1:-1] = 0.5 * (np.log(s[2:]) - np.log(s[:-2]))

    S, W  = np.meshgrid(s, w)
    ws    = S * W
    ws2   = ws ** 2
    kern  = np.vstack((ws2 / (1 + ws2), ws / (1 + ws2)))

    wij = np.dot(kern, np.diag(hs * np.exp(H)))  # (2n x ns)
    K   = np.dot(kern, hs * np.exp(H))            # (2n,)

    for i in range(n):
        wij[i, :] = wij[i, :] / K[i]

    wt = np.sum(wij, axis=0)                      # (ns,)
    wt = wt / np.trapezoid(wt, np.log(s))
    wt = (1.0 - wb) * wt + (wb * np.mean(wt)) * np.ones(ns)

    return wt


def _GridDensity(
    x: np.ndarray,
    px: np.ndarray,
    N: int,
):
    """
    Distribute N points according to the density px(x) using
    equal-mass intervals (quantile placement).

    Returns z : (N,) points, hz : (N,) interval widths.
    """
    npts = 100
    xi   = np.linspace(x.min(), x.max(), npts)
    fint = interp1d(x, px, kind="cubic")
    pint = fint(xi)
    ci   = cumulative_trapezoid(pint, xi, initial=0)
    pint = pint / ci[-1]
    ci   = ci   / ci[-1]

    alfa    = 1.0 / (N - 1)
    zij     = np.zeros(N + 1)
    z       = np.zeros(N)
    z[0]    = x.min()
    z[-1]   = x.max()

    fint2      = interp1d(ci, xi, kind="cubic")
    beta       = np.arange(0.5, N - 0.5) * alfa
    zij[0]     = z[0]
    zij[-1]    = z[-1]
    zij[1:N]   = fint2(beta)
    hz         = np.diff(zij)

    beta2      = np.arange(1, N - 1) * alfa
    z[1:-1]    = fint2(beta2)

    return z, hz


def _mergeModes(
    g: np.ndarray,
    tau: np.ndarray,
    imode: int,
):
    """
    Merge modes imode and imode+1 into a single mode by minimising
    the integrated squared relative error between the two-mode and
    one-mode representations.
    """
    def _normKern(w, gn, taun, g1, tau1, g2, tau2):
        wt    = w * taun
        Gnp   = gn * wt**2 / (1 + wt**2)
        Gnpp  = gn * wt    / (1 + wt**2)
        wt    = w * tau1
        Gop   = g1 * wt**2 / (1 + wt**2)
        Gopp  = g1 * wt    / (1 + wt**2)
        wt    = w * tau2
        Gop  += g2 * wt**2 / (1 + wt**2)
        Gopp += g2 * wt    / (1 + wt**2)
        return (Gnp / Gop - 1.0)**2 + (Gnpp / Gopp - 1.0)**2

    from scipy.integrate import quad

    def _cost(par):
        gn, taun = par
        g1, g2   = g[imode], g[imode + 1]
        t1, t2   = tau[imode], tau[imode + 1]
        wmin     = min(1.0/t1, 1.0/t2) / 10.0
        wmax     = max(1.0/t1, 1.0/t2) * 10.0
        return quad(_normKern, wmin, wmax, args=(gn, taun, g1, t1, g2, t2))[0]

    res       = minimize(_cost, [g[imode] + g[imode+1],
                                 0.5*(tau[imode] + tau[imode+1])])
    newtau    = np.delete(tau, imode + 1)
    newtau[imode] = res.x[1]
    newg      = np.delete(g, imode + 1)
    newg[imode]   = res.x[0]

    return newg, newtau


def _FineTuneSolution(
    tau: np.ndarray,
    w: np.ndarray,
    Gexp: np.ndarray,
    wexp: np.ndarray,
    isPlateau: bool,
):
    """
    NLLS refinement of tau positions, warm-started from the current tau.
    Falls back to the original tau if the refinement does not improve the fit.

    Returns (success, g, tau).
    """
    def _res_wG(tau, w, Gexp, wexp, isPlateau):

        g, _, _ = _nnLLS(w, tau, Gexp, wexp, isPlateau)

        S, W = np.meshgrid(tau, w)
        ws   = S * W;  ws2 = ws**2
        K    = np.vstack((ws2 / (1 + ws2), ws / (1 + ws2)))

        if isPlateau:
            Gmodel        = np.dot(K, g[:-1])
            n             = len(Gexp) // 2
            Gmodel[:n]   += g[-1]
        else:
            Gmodel = np.dot(K, g)

        return wexp * (Gmodel / Gexp - 1.0)

    initError = np.linalg.norm(_res_wG(tau, w, Gexp, wexp, isPlateau))
    success   = False

    try:
        res = least_squares(
            _res_wG, tau,
            bounds=(0.02 / max(w), 50 / min(w)),
            args=(w, Gexp, wexp, isPlateau),
        )
        tau     = res.x
        success = True
    except Exception:
        pass

    g, tau, _, _ = _MaxwellModes(np.log(tau), w, Gexp, wexp, isPlateau)
    finalError   = np.linalg.norm(_res_wG(tau, w, Gexp, wexp, isPlateau))

    if finalError > initError:
        success = False

    return success, g, tau


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def getDiscSpec(
    w: np.ndarray,
    Gexp: np.ndarray,
    wexp: np.ndarray,
    cont: ContinuousResult,
    config: ReSpectConfig,
) -> DiscreteResult:
    """
    Extract the discrete relaxation spectrum from the continuous result.

    Scans over (wb, N) pairs, selects the combination minimising the AIC:

        AIC = 2N + 2 * Cerror * error(N)

    then fine-tunes tau positions via NLLS and merges modes that are
    too close (tau_{i+1}/tau_i < min_tau_spacing).

    Parameters
    ----------
    w, Gexp, wexp : experimental data (from GetExpData)
    cont          : ContinuousResult from getContSpec
    config        : ReSpectConfig

    Returns
    -------
    DiscreteResult
    """
    s   = cont.s
    H   = cont.H
    ns  = len(s)
    n   = len(w)

    # estimate Cerror from continuous fit residual
    from .kernels import getKernMat, kernel_prestore
    kernMat = getKernMat(s, w)
    Gc      = kernel_prestore(H, kernMat, cont.G0)
    Cerror  = 1.0 / np.std(wexp * (Gc / Gexp - 1.0))

    # range of N scanned
    Nmax = min(np.floor(3.0 * np.log10(max(w) / min(w))), n / 4)
    if config.max_num_modes is not None:
        Nmax = min(Nmax, config.max_num_modes)
    Nmin = max(np.floor(0.5 * np.log10(max(w) / min(w))), 3)
    Nv   = np.arange(Nmin, Nmax + 1).astype(int)
    npts = len(Nv)

    wtBase  = config.delta_base_weight_dist * np.arange(
        1, int(1.0 / config.delta_base_weight_dist)
    )
    AICbst  = np.zeros(len(wtBase))
    Nbst    = np.zeros(len(wtBase))
    nzNbst  = np.zeros(len(wtBase))


    for ib, wb in enumerate(wtBase):
        wt    = _GetWeights(H, w, s, wb)
        ev    = np.zeros(npts)
        nzNv  = np.zeros(npts)

        for i, N in enumerate(Nv):
            z, _          = _GridDensity(np.log(s), wt, N)
            g, tau, ev[i], _ = _MaxwellModes(z, w, Gexp, wexp, config.plateau)
            nzNv[i]       = len(g)

        AIC        = 2.0 * Nv + 2.0 * Cerror * ev
        AICbst[ib] = AIC.min()
        Nbst[ib]   = Nv[AIC.argmin()]
        nzNbst[ib] = nzNv[AIC.argmin()]

    Nopt  = int(Nbst[AICbst.argmin()])
    wbopt = wtBase[AICbst.argmin()]

    # recompute best solution
    wt               = _GetWeights(H, w, s, wbopt)
    z, _             = _GridDensity(np.log(s), wt, Nopt)
    g, tau, error, cKp = _MaxwellModes(z, w, Gexp, wexp, config.plateau)

    succ, gf, tauf   = _FineTuneSolution(tau, w, Gexp, wexp, config.plateau)
    if succ:
        g, tau = gf.copy(), tauf.copy()        

    # sort and merge close modes
    indx       = np.argsort(tau)
    tau        = tau[indx]
    if config.plateau:
        g[:-1] = g[indx]
    else:
        g      = g[indx]

    tauSpacing = tau[1:] / tau[:-1]
    itry       = 0
    while min(tauSpacing) < config.min_tau_spacing and itry < 3:
        imode        = np.argmin(tauSpacing)
        g, tau       = _mergeModes(g, tau, imode)
        succ, g, tau = _FineTuneSolution(tau, w, Gexp, wexp, config.plateau)
        if succ:
            # fine-tune after merge succeeded: revert to the pre-merge
            # fine-tuned solution (gf, tauf), matching legacy behaviour
            g, tau = gf.copy(), tauf.copy()
        tauSpacing = tau[1:] / tau[:-1]
        itry      += 1

    G0 = 0.0
    if config.plateau:
        G0 = g[-1]
        g  = g[:-1]

    # compute G_fit for storage
    S, W  = np.meshgrid(tau, w)
    ws    = S * W;  ws2 = ws ** 2
    K     = np.vstack((ws2 / (1 + ws2), ws / (1 + ws2)))
    G_fit = np.dot(K, g)
    if config.plateau:
        G_fit[:n] += G0

    return DiscreteResult(
        g=g,
        tau=tau,
        N=len(g),
        G_fit=G_fit,
        G0=G0,
        error=error,
        wtBase=wtBase,
        AICbst=AICbst,
        nzNbst=nzNbst,
    )