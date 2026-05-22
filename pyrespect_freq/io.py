"""
io.py — Data loading and file output for pyReSpect-freq.

All file I/O is confined here. No scientific computation happens in
this module.

Functions
---------
load_data(source, resample, n_resample)
    Load G*(w) data from a file path or a tuple of arrays.

save(path, which, w, cont_result, disc_result)
    Write results to files in the specified output directory.

Private
-------
_load_from_file(fname)
    Read a 3- or 5-column text file.
_resample_geometric(w, Gp, Gpp, n)
    Resample G', G'' onto a geometric frequency grid.
_parse_which(which)
    Normalise and validate the 'which' argument.
_validate_which(tokens, cont_result, disc_result)
    Check that required results exist for the requested outputs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .config import ReSpectError
from .continuous import ContinuousResult
from .discrete import DiscreteResult


# Valid 'which' tokens
_VALID_WHICH  = {"base", "full"}
_NEEDS_CONT   = {"base", "full"}
_NEEDS_DISC   = {"base", "full"}


# ---------------------------------------------------------------------------
# Public: load_data
# ---------------------------------------------------------------------------

def load_data(
    source:     Union[str, tuple],
    resample:   bool = True,
    n_resample: int  = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load experimental G*(w) data.

    Accepts a file path or a tuple of arrays. Optionally resamples
    3-column / length-3-tuple data onto a geometric frequency grid.

    Parameters
    ----------
    source : str or tuple
        One of:

        - A path to a data file with 3 columns ``[w, G', G'']`` or
          5 columns ``[w, G', G'', w_{G'}, w_{G''}]``.
        - A tuple ``(w, Gp, Gpp)`` of 1-D numpy arrays.
        - A tuple ``(w, Gp, Gpp, wGp, wGpp)`` of 1-D numpy arrays,
          where the last two are per-datapoint weights for G' and G''.

        5-column files and length-5 tuples are treated as pre-processed
        (no resampling is applied regardless of the *resample* flag).

    resample : bool, optional
        If True (default), resample 3-column / length-3 tuple data onto
        a geometric grid of *n_resample* points. Ignored for 5-column
        / length-5 data.
    n_resample : int, optional
        Number of output points when resampling. Default: 100.

    Returns
    -------
    w    : np.ndarray, shape (n,)
        Angular frequencies.
    Gexp : np.ndarray, shape (2n,)
        Concatenated ``[G' | G'']``.
    wexp : np.ndarray, shape (2n,)
        Concatenated per-datapoint weights ``[w_{G'} | w_{G''}]``.

    Raises
    ------
    ReSpectError
        If the source cannot be read or is incorrectly formatted.
    """
    if isinstance(source, (str, Path)):
        w, Gp, Gpp, wGp, wGpp, is_preprocessed = _load_from_file(str(source))

    elif isinstance(source, tuple):
        if len(source) == 3:
            w, Gp, Gpp = (np.asarray(a, dtype=float) for a in source)
            _check_shapes(w, Gp, Gpp)
            wGp  = np.ones(len(w))
            wGpp = np.ones(len(w))
            is_preprocessed = False

        elif len(source) == 5:
            w, Gp, Gpp, wGp, wGpp = (np.asarray(a, dtype=float) for a in source)
            _check_shapes(w, Gp, Gpp, wGp, wGpp)
            is_preprocessed = True

        else:
            raise ReSpectError(
                "Tuple source must have length 3 (w, Gp, Gpp) or "
                "5 (w, Gp, Gpp, wGp, wGpp)."
            )
    else:
        raise ReSpectError(
            "source must be a file path (str) or a tuple of arrays."
        )

    # Resample unless data is pre-processed (5-col / length-5 tuple)
    if resample and not is_preprocessed:
        w, Gp, Gpp = _resample_geometric(w, Gp, Gpp, n_resample)
        wGp  = np.ones(len(w))
        wGpp = np.ones(len(w))

    Gexp = np.append(Gp,  Gpp)
    wexp = np.append(wGp, wGpp)

    return w, Gexp, wexp


# ---------------------------------------------------------------------------
# Public: save
# ---------------------------------------------------------------------------

def save(
    path:        str,
    which:       Union[str, list[str]],
    w:           np.ndarray,
    cont_result: Optional[ContinuousResult] = None,
    disc_result: Optional[DiscreteResult]   = None,
) -> None:
    """Write results to files in the specified output directory.

    Parameters
    ----------
    path : str
        Output directory. Created if it does not exist.
    which : str or list of str
        Which outputs to write. Valid values:

        - ``"base"`` : crs.dat, drs.dat, Gfit.dat.
        - ``"full"`` : above + rho-eta.dat, logPlam.dat, aic.dat.
          Diagnostic files requiring L-curve data are silently skipped
          when lam_C was pre-specified.

    w : np.ndarray, shape (n,)
        Angular frequencies (needed to write Gfit.dat).
    cont_result : ContinuousResult or None
    disc_result : DiscreteResult or None

    Raises
    ------
    ReSpectError
        If a requested output requires a result that has not been
        computed, or an invalid 'which' token is supplied.
    """
    tokens = _parse_which(which)
    _validate_which(tokens, cont_result, disc_result)
    os.makedirs(path, exist_ok=True)

    for token in tokens:
        if token == "base":
            _write_base(path, w, cont_result, disc_result)
        elif token == "full":
            _write_base(path, w, cont_result, disc_result)
            _write_full(path, cont_result, disc_result)


# ---------------------------------------------------------------------------
# Private: file loading
# ---------------------------------------------------------------------------

def _load_from_file(
    fname: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, bool]:
    """Read a 3- or 5-column G*(w) data file.

    Returns
    -------
    w, Gp, Gpp, wGp, wGpp : np.ndarray
    is_preprocessed : bool
        True for 5-column files; resampling is skipped.
    """
    try:
        data = np.loadtxt(fname)
    except OSError:
        raise ReSpectError(
            f"Could not read data file '{fname}'. "
            "Check that the path is correct and the file is properly "
            "formatted."
        )

    if data.ndim != 2 or data.shape[1] not in (3, 5):
        raise ReSpectError(
            f"Data file '{fname}' must have 3 columns [w, G', G''] "
            "or 5 columns [w, G', G'', w_G', w_G'']."
        )

    # remove duplicate frequencies
    w, idx = np.unique(data[:, 0], return_index=True)
    Gp     = data[idx, 1]
    Gpp    = data[idx, 2]

    if data.shape[1] == 5:
        wGp  = data[idx, 3]
        wGpp = data[idx, 4]
        return w, Gp, Gpp, wGp, wGpp, True
    else:
        return w, Gp, Gpp, np.ones(len(w)), np.ones(len(w)), False


def _resample_geometric(
    w:   np.ndarray,
    Gp:  np.ndarray,
    Gpp: np.ndarray,
    n:   int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample G', G'' onto n geometrically-spaced frequency points."""
    from scipy.interpolate import interp1d

    fp   = interp1d(w, Gp,  fill_value="extrapolate")
    fpp  = interp1d(w, Gpp, fill_value="extrapolate")
    w_new   = np.geomspace(w.min(), w.max(), n)
    return w_new, fp(w_new), fpp(w_new)


def _check_shapes(*arrays: np.ndarray) -> None:
    """Raise ReSpectError if arrays are not all 1-D and the same length."""
    shapes = [a.shape for a in arrays]
    if any(a.ndim != 1 for a in arrays):
        raise ReSpectError("All input arrays must be 1-D.")
    if len(set(s[0] for s in shapes)) > 1:
        raise ReSpectError(
            f"All input arrays must have the same length; "
            f"got shapes {shapes}."
        )


# ---------------------------------------------------------------------------
# Private: write helpers
# ---------------------------------------------------------------------------

def _write_base(
    path:        str,
    w:           np.ndarray,
    cont_result: ContinuousResult,
    disc_result: DiscreteResult,
) -> None:
    """Write crs.dat, drs.dat, and Gfit.dat."""
    n = len(w)

    # crs.dat: [s, exp(H(s))] with optional G0 header
    hdr_c = f"G0 = {cont_result.G0:.6e}" if cont_result.G0 else ""
    np.savetxt(
        os.path.join(path, "crs.dat"),
        np.c_[cont_result.s, np.exp(cont_result.H)],
        fmt="%e",
        header=hdr_c,
    )

    # drs.dat: [g_i, tau_i] with optional G0 header
    hdr_d = f"G0 = {disc_result.G0:.6e}" if disc_result.G0 else ""
    np.savetxt(
        os.path.join(path, "drs.dat"),
        np.c_[disc_result.g, disc_result.tau],
        fmt="%e",
        header=hdr_d,
    )

    # Gfit.dat: [w, Gp_cont, Gpp_cont, Gp_disc, Gpp_disc]
    np.savetxt(
        os.path.join(path, "Gfit.dat"),
        np.c_[
            w,
            cont_result.G_fit[:n], cont_result.G_fit[n:],
            disc_result.G_fit[:n], disc_result.G_fit[n:],
        ],
        fmt="%e",
        header="w  Gp_cont  Gpp_cont  Gp_disc  Gpp_disc",
    )


def _write_full(
    path:        str,
    cont_result: ContinuousResult,
    disc_result: DiscreteResult,
) -> None:
    """Write rho-eta.dat, logPlam.dat, aic.dat."""

    # L-curve files only available when lam_C was auto-determined
    if cont_result.lam is not None:
        np.savetxt(
            os.path.join(path, "rho-eta.dat"),
            np.c_[cont_result.lam, cont_result.rho, cont_result.eta],
            fmt="%e",
            header="lambda  rho  eta",
        )
        np.savetxt(
            os.path.join(path, "logPlam.dat"),
            np.c_[cont_result.lam, cont_result.logP],
            fmt="%e",
            header="lambda  logP",
        )

    # AIC scan always available
    if disc_result.wtBase is not None:
        np.savetxt(
            os.path.join(path, "aic.dat"),
            np.c_[disc_result.wtBase, disc_result.nzNbst, disc_result.AICbst],
            fmt="%f\t%i\t%e",
            header="baseDistWt  Nbst  AIC",
        )


# ---------------------------------------------------------------------------
# Private: validation helpers
# ---------------------------------------------------------------------------

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
