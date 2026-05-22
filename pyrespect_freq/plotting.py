"""
plotting.py — Plotting routines for pyReSpect-freq.

All plot functions are pure in the sense that they only read from
result dataclasses and return matplotlib Figure objects. No computation
happens here. Figures can optionally be saved to disk.

Public API
----------
plot(which, toFile, path, w, Gexp, wexp, cont_result, disc_result)
    Dispatcher: produces all requested figures and returns them as a list.

All other functions are private to this module.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .config import ReSpectError
from .continuous import ContinuousResult
from .discrete import DiscreteResult


# Valid 'which' tokens
_VALID_WHICH = {"base", "full"}
_NEEDS_CONT  = {"base", "full"}
_NEEDS_DISC  = {"base", "full"}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot(
    which:       Union[str, list[str]],
    toFile:      bool,
    path:        str,
    w:           np.ndarray,
    Gexp:        np.ndarray,
    cont_result: Optional[ContinuousResult],
    disc_result: Optional[DiscreteResult],
) -> list[Figure]:
    """Produce plots of the fitted spectra.

    Parameters
    ----------
    which : str or list of str
        Which plots to produce. Valid values:

        - ``"base"`` : two-panel figure —
                       left:  exp(H(s)) with ±2.5 dH error band and
                              discrete modes g_i overlaid;
                       right: G*(w) data vs continuous and discrete fits.
        - ``"full"`` : above + three-panel diagnostic figure —
                       log p(λ) vs λ, ρ-η L-curve, AIC scan.
                       Panels requiring L-curve data are silently replaced
                       with informative text when lam_C was pre-specified.

    toFile : bool
        If True, save each figure as a PDF in *path*.
    path : str
        Output directory for saved figures.
    w : np.ndarray, shape (n,)
        Angular frequencies.
    Gexp : np.ndarray, shape (2n,)
        Concatenated experimental data [G' | G''].
    cont_result : ContinuousResult or None
    disc_result : DiscreteResult or None

    Returns
    -------
    figs : list of Figure
        All figures produced, in the order requested.

    Raises
    ------
    ReSpectError
        If a requested plot requires a result that is not available,
        or if an invalid 'which' token is supplied.
    """
    tokens = _parse_which(which)
    _validate_which(tokens, cont_result, disc_result)

    if toFile:
        os.makedirs(path, exist_ok=True)

    figs: list[Figure] = []

    for token in tokens:

        if token == "base":
            fig = _plot_base(w, Gexp, cont_result, disc_result)
            figs.append(fig)
            if toFile:
                _save_fig(fig, path, "Gfit.pdf")
            else:
                plt.show()

        elif token == "full":
            fig1 = _plot_base(w, Gexp, cont_result, disc_result)
            figs.append(fig1)
            if toFile:
                _save_fig(fig1, path, "Gfit.pdf")
            else:
                plt.show()

            fig2 = _plot_diagnostics(cont_result, disc_result)
            figs.append(fig2)
            if toFile:
                _save_fig(fig2, path, "diagnostics.pdf")
            else:
                plt.show()

    return figs


# ---------------------------------------------------------------------------
# Private: main figure
# ---------------------------------------------------------------------------

def _plot_base(
    w:           np.ndarray,
    Gexp:        np.ndarray,
    cont_result: ContinuousResult,
    disc_result: DiscreteResult,
) -> Figure:
    """Two-panel figure.

    Left panel
        Continuous spectrum exp(H(s)) with ±2.5 dH error band (when
        available), overlaid with discrete mode weights g_i vs τ_i.

    Right panel
        Experimental G*(w) data (G' circles, G'' squares) against
        continuous (solid) and discrete (dashed) model fits, on a
        log-log axis.
    """
    n   = len(w)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ---- Left: exp(H(s)) with error band and discrete modes ----
    ax = axes[0]
    ax.loglog(cont_result.s, np.exp(cont_result.H), label=r"CRS")

    if cont_result.dH is not None:
        ax.loglog(
            cont_result.s,
            np.exp(cont_result.H + 2.5 * cont_result.dH),
            c="gray", alpha=0.5,
        )
        ax.loglog(
            cont_result.s,
            np.exp(cont_result.H - 2.5 * cont_result.dH),
            c="gray", alpha=0.5,
        )

    ax.loglog(
        disc_result.tau, disc_result.g,
        "o-", label="DRS",
    )
    ax.set_xlabel(r"$s; \tau_i$")
    ax.set_ylabel(r"$h(s),\; g_i$")
    ax.legend(fontsize=10)

    # ---- Right: G*(w) data vs fits ----
    ax = axes[1]
    ax.loglog(w, Gexp[:n], "x",  c="gray")
    ax.loglog(w, Gexp[n:], "+",  c="gray")
    ax.loglog(w, cont_result.G_fit[:n], "-",  c="C0", label=r"CRS")
    ax.loglog(w, cont_result.G_fit[n:], "-",  c="C0")
    ax.loglog(w, disc_result.G_fit[:n], "--", c="C1", label=r"DRS")
    ax.loglog(w, disc_result.G_fit[n:], "--", c="C1")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$G^*(\omega)$")
    ax.legend(fontsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Private: diagnostic figure
# ---------------------------------------------------------------------------

def _plot_diagnostics(
    cont_result: ContinuousResult,
    disc_result: DiscreteResult,
) -> Figure:
    """Three-panel diagnostic figure: log p(λ) | ρ-η L-curve | AIC scan.

    The left and middle panels are replaced by informative text when
    lam_C was pre-specified (cont_result.lam is None). The AIC panel
    is always shown.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ---- Left: log p(λ) ----
    ax = axes[0]
    if cont_result.lam is not None:
        ax.plot(cont_result.lam, cont_result.logP, "o-")
        ax.axvline(
            cont_result.lamC, color="gray",
            label=r"$\lambda_M = {:.2e}$".format(cont_result.lamC),
        )
        ax.set_xscale("log")
        ax.set_ylim(-20, 1)
        ax.legend(loc="upper left", fontsize=10)
    else:
        ax.text(
            0.5, 0.5,
            "L-curve not computed\n(lam_C pre-specified)",
            ha="center", va="center", transform=ax.transAxes,
        )
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\log\, p(\lambda)$")

    # ---- Middle: ρ-η L-curve ----
    ax = axes[1]
    if cont_result.lam is not None:
        ax.plot(cont_result.rho, cont_result.eta)
        # interpolate optimal point in log-log space for accuracy
        rho_opt = np.exp(np.interp(
            np.log(cont_result.lamC),
            np.log(cont_result.lam),
            np.log(cont_result.rho),
        ))
        eta_opt = np.exp(np.interp(
            np.log(cont_result.lamC),
            np.log(cont_result.lam),
            np.log(cont_result.eta),
        ))
        ax.plot(
            rho_opt, eta_opt, "o", c="C1",
            label=r"$\lambda^* = {:.2e}$".format(cont_result.lamC),
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=10)
    else:
        ax.text(
            0.5, 0.5,
            "L-curve not computed\n(lam_C pre-specified)",
            ha="center", va="center", transform=ax.transAxes,
        )
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\eta$")

    # ---- Right: AIC scan ----
    ax  = axes[2]
    ax2 = ax.twinx()

    color_aic = "C2"
    color_n   = "C1"

    ax.plot(
        disc_result.wtBase, disc_result.AICbst,
        color=color_aic, label="AIC",
    )
    ax.set_xlabel(r"$w_b$")
    ax.set_ylabel("AIC", color=color_aic)
    ax.set_yscale("log")
    ax.tick_params(axis="y", labelcolor=color_aic)

    ax2.plot(
        disc_result.wtBase, disc_result.nzNbst,
        color=color_n, linestyle="--", label=r"$N_\mathrm{bst}$",
    )
    ax2.set_ylabel(r"$N_\mathrm{bst}$", color=color_n)
    ax2.tick_params(axis="y", labelcolor=color_n)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Private: helpers
# ---------------------------------------------------------------------------

def _save_fig(fig: Figure, path: str, fname: str) -> None:
    fig.savefig(os.path.join(path, fname))
    plt.close(fig)


def _parse_which(which: Union[str, list[str]]) -> list[str]:
    tokens  = [which] if isinstance(which, str) else list(which)
    invalid = set(tokens) - _VALID_WHICH
    if invalid:
        raise ReSpectError(
            f"Invalid 'which' value(s): {invalid}. "
            f"Must be one or more of {_VALID_WHICH}."
        )
    return tokens


def _validate_which(
    tokens:      list[str],
    cont_result: Optional[ContinuousResult],
    disc_result: Optional[DiscreteResult],
) -> None:
    for token in tokens:
        if token in _NEEDS_CONT and cont_result is None:
            raise ReSpectError(
                f"'{token}' requires a continuous result. Run fit() first."
            )
        if token in _NEEDS_DISC and disc_result is None:
            raise ReSpectError(
                f"'{token}' requires a discrete result. Run fit() first."
            )
