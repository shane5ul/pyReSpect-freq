"""
solver.py — ReSpect: the primary user-facing solver class.

    solver = ReSpect()
    solver.fit("Gst.dat")

    solver.continuous.H        # continuous spectrum H(s)
    solver.continuous.G_fit    # predicted [G'|G''] from CRS
    solver.discrete.tau        # discrete relaxation times
    solver.discrete.G_fit      # predicted [G'|G''] from DRS

    solver.plot(which="base")
    solver.save(which="full", path="output/")

Method chaining is supported:

    ReSpect().fit("Gst.dat").save(which="base", path="output/")
"""

from __future__ import annotations

from typing import Optional, Union
import numpy as np
import os
from pathlib import Path

from .config import ReSpectConfig, ReSpectError
from .io import load_data, save
from .plotting import plot as _plot
from .continuous import getContSpec, ContinuousResult
from .discrete import getDiscSpec, DiscreteResult


class ReSpect:
    """
    Solver for extracting continuous and discrete relaxation spectra
    from frequency-domain G*(w) data.

    Parameters
    ----------
    config : ReSpectConfig, optional
        Solver configuration. Defaults to ReSpectConfig() if not supplied.
    """

    def __init__(self, config: Optional[ReSpectConfig] = None) -> None:
        self.config:     ReSpectConfig              = config or ReSpectConfig()
        self.continuous: Optional[ContinuousResult] = None
        self.discrete:   Optional[DiscreteResult]   = None
        self._w:    Optional[np.ndarray]            = None
        self._Gexp: Optional[np.ndarray]            = None
        self._wexp: Optional[np.ndarray]            = None

    # ------------------------------------------------------------------
    # Alternative constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_toml(cls, path: str) -> ReSpect:
        """Construct a ReSpect solver from a TOML configuration file."""
        return cls(ReSpectConfig.from_toml(path))

    @classmethod
    def from_yaml(cls, path: str) -> ReSpect:
        """Construct a ReSpect solver from a YAML configuration file."""
        return cls(ReSpectConfig.from_yaml(path))

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        source: Union[str, tuple],
    ) -> ReSpect:
        """
        Load G*(w) data and compute continuous then discrete spectra.

        Parameters
        ----------
        source : str or tuple
            One of:

            - ``"Gst.dat"`` : path to a 3-column ``[w, G', G'']`` or
              5-column ``[w, G', G'', w_{G'}, w_{G''}]`` data file.
            - ``(w, Gp, Gpp)`` : tuple of 1-D arrays.
            - ``(w, Gp, Gpp, wGp, wGpp)`` : tuple with per-datapoint weights.

            5-column files and length-5 tuples are treated as
            pre-processed (resampling is skipped).

        Returns
        -------
        self — supports method chaining.
        """
        # --- Pre-flight validation ---
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            if not source_path.exists():
                raise ReSpectError(
                    f"Could not read data file '{source}'. "
                    "Check that the path is correct and the file is properly formatted."
                )

        
        self._w, self._Gexp, self._wexp = load_data(
            source,
            resample=self.config.resample,
            n_resample=self.config.n_resample,
        )

        self.continuous = getContSpec(
            self._w, self._Gexp, self._wexp, self.config
        )

        self.discrete = getDiscSpec(
            self._w, self._Gexp, self._wexp, self.continuous, self.config
        )

        return self

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------

    def save(
        self,
        which: Union[str, list[str]] = "base",
        path:  str                   = "./",
    ) -> ReSpect:
        """
        Write result files to *path*.

        Parameters
        ----------
        which : "base" or "full"
            ``"base"`` writes crs.dat, drs.dat, Gfit.dat.
            ``"full"`` additionally writes rho-eta.dat, logPlam.dat, aic.dat.
        path : str
            Output directory (created if it does not exist).

        Returns
        -------
        self — supports method chaining.
        """
        self._check_fitted("save")
        save(
            path=path,
            which=which,
            w=self._w,
            cont_result=self.continuous,
            disc_result=self.discrete,
        )
        return self

    # ------------------------------------------------------------------
    # plot
    # ------------------------------------------------------------------

    def plot(
        self,
        which:  Union[str, list[str]] = "base",
        toFile: bool                  = False,
        path:   str                   = "./",
    ) -> ReSpect:
        """
        Plot spectra and diagnostics.

        Parameters
        ----------
        which : "base" or "full"
            ``"base"`` produces a two-panel figure: exp(H(s)) with error
            band and discrete modes (left), G*(w) data vs fits (right).
            ``"full"`` additionally produces a three-panel diagnostic
            figure: log p(λ), ρ-η L-curve, AIC scan.
        toFile : bool
            If True, save figures as PDFs to *path* instead of
            displaying interactively.
        path : str
            Output directory for PDFs (used only when toFile=True).

        Returns
        -------
        self — supports method chaining.
        """
        self._check_fitted("plot")
        _plot(
            which=which,
            toFile=toFile,
            path=path,
            w=self._w,
            Gexp=self._Gexp,
            cont_result=self.continuous,
            disc_result=self.discrete,
        )
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self, caller: str) -> None:
        if self.continuous is None or self.discrete is None:
            raise ReSpectError(
                f"ReSpect.{caller}() called before fit(). "
                "Run solver.fit(source) first."
            )
