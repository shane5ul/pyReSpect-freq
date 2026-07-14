# pyReSpect-freq-2.0

A rewrite of the classic python library for extracting **continuous** and **discrete** relaxation spectra from frequency sweep data. The core algorithms are the same, the interface (both developer and user) is modernized.

The legacy codebase has been [archived](https://github.com/shane5ul/pyReSpect-freq-legacy).

pyReSpect solves the inverse problem: given measurements of
$G'(\omega)$ and $G''(\omega)$, recover $H(s)$ such that

$$G'(\omega)  = \int \frac{\omega^2 s^2}{1+\omega^2 s^2} e^{H(s)} d\ln s, \qquad
  G''(\omega) = \int \frac{\omega s}{1+\omega^2 s^2} e^{H(s)} d\ln s$$

The CRS $H(s)$ is found via Tikhonov regularization with a Bayesian criterion for selecting the regularization parameter $\lambda$. The DRS (Maxwell modes $g_i$, $\tau_i$) is then extracted from the CRS using an AIC-minimization strategy.

## References

- Shanbhag, S., "Relaxation spectra using nonlinear Tikhonov regularization with a Bayesian criterion", *Rheologica Acta*, **2020**, 59, 509.
  [doi:10.1007/s00397-020-01212-w](https://doi.org/10.1007/s00397-020-01212-w)
- Shanbhag, S., "pyReSpect: A Computer Program to Extract Discrete and Continuous
  Spectra from Stress Relaxation Experiments", *Macromolecular Theory and Simulations*, **2019**, 1900005.
  [doi:10.1002/mats.201900005](https://doi.org/10.1002/mats.201900005)
- Takeh, A. and Shanbhag, S., "A computer program to extract the continuous and   discrete relaxation spectra from dynamic viscoelastic measurements", *Appl.
  Rheol.* **2013**, 23, 24628.

---

## Installation

### Requirements

- Python >= 3.12
- numpy >= 2.4
- scipy >= 1.17
- matplotlib

### From GitHub

```bash
pip install git+https://github.com/shane5ul/pyReSpect-freq.git
```

### For development

Clone the repository and install in editable mode from the repo root:

```bash
git clone https://github.com/shane5ul/pyReSpect-freq.git
cd pyReSpect-freq
pip install -e .
```

**Dependencies**: Python ≥ 3.12, NumPy, SciPy, Matplotlib.

---
## Features

- Easy installation
- Library functions can be imported and called from other programs
- Clean object-oriented API with method chaining
- Simplified user-interface: 
  - configuration (old `inp.dat`) and data (old `Gt.dat`) can be supplied both programmatically or via files
  - TOML configuration file support
- It separates computation from I/O and plotting.
- Continuous spectrum via Tikhonov regularization with Bayesian $\lambda$ selection
- Discrete Maxwell modes via AIC minimization and NLLS fine-tuning
- Optional plateau modulus $G_0$ for viscoelastic solids

---

## Quick start

```python
from pyrespect_freq import ReSpect

solver = ReSpect()
solver.fit("Gst.dat")
solver.save(which="full", path="output/")
solver.plot(which="base")
```

Accessing results directly:

```python
solver.continuous.s      # relaxation time grid  (ns,)
solver.continuous.H      # log-CRS H(s)          (ns,)  →  h(s) = exp(H(s))
solver.continuous.G_fit  # predicted [G'|G'']    (2n,)
solver.continuous.G0     # plateau modulus (0 if plateau=False)
solver.continuous.lamC   # optimal λ used

solver.discrete.g        # mode weights g_i       (N,)
solver.discrete.tau      # relaxation times τ_i   (N,)
solver.discrete.G_fit    # predicted [G'|G'']     (2n,)
solver.discrete.G0       # plateau modulus
```

---

## Input data

`fit()` accepts either a **file path** or a **tuple of arrays**.

### File format

A plain-text file with whitespace-separated columns. Two formats are supported:

| Columns | Meaning | Resampled? |
|---------|---------|------------|
| 3: `w  G'  G''` | raw frequency-sweep data | yes (onto 100-point geometric grid) |
| 5: `w  G'  G''  wt_G'  wt_G''` | pre-processed data with per-point weights | no |

Duplicate frequency values are removed automatically. Lines beginning with `#`
are ignored.

### Tuple input

```python
import numpy as np

w   = np.logspace(-2, 2, 50)
Gp  = ...   # G'(w)
Gpp = ...   # G''(w)

solver.fit((w, Gp, Gpp))                 # 3-tuple: resampled
solver.fit((w, Gp, Gpp, wt_Gp, wt_Gpp))  # 5-tuple: 
```

---

## Configuration

All parameters are collected in `ReSpectConfig`. Every parameter has a sensible
default, so `ReSpectConfig()` works out of the box.

```python
from pyrespect_freq import ReSpect, ReSpectConfig

config = ReSpectConfig(
    ns       = 100,       # CRS grid points
    plateau  = False,     # set True for viscoelastic solids (fits G0)
    freq_end = "lenient", # s-axis window: "lenient" | "neutral" | "strict"
)
solver = ReSpect(config).fit("Gst.dat")
```

### Configuration from file

**TOML** (`inp.toml`):

```toml
[spectrum]
ns       = 100
plateau  = true

[io]
resample   = true
n_resample = 200
```

```python
solver = ReSpect.from_toml("inp.toml").fit("Gst.dat")
```

### Full parameter reference

#### Continuous spectrum

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ns` | int | 100 | Number of grid points for $s$ (the CRS axis). Typical range: 50–200. |
| `plateau` | bool | False | Fit a non-zero plateau modulus $G_0$ (for viscoelastic solids). |
| `freq_end` | str | `"lenient"` | How far $s$ extends beyond the frequency window. |

#### Regularization ($\lambda$ selection)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lam_min` | float | 1e-10 | Lower bound of the $\lambda$ search grid. |
| `lam_max` | float | 1e3 | Upper bound of the $\lambda$ search grid. |
| `lam_C` | float \| None | None | Pin $\lambda$ to this value instead of using the Bayesian L-curve. When set, `dH` (the error band) is not computed. |
| `lam_density` | int | 2 | $\lambda$ grid points per decade. Increase for a finer search. |
| `SmFacLam` | float | 0.0 | Smoothness nudge in $[-1, 1]$. Positive values push $\lambda$ toward `lam_max` (smoother spectrum); negative values push toward `lam_min` (rougher). |

#### Discrete spectrum

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_num_modes` | int \| None | None | Cap on the number of Maxwell modes scanned. `None` = auto. |
| `delta_base_weight_dist` | float | 0.2 | Step size for the AIC scan over the base weight parameter $w_b \in (0,1)$. Smaller → finer scan, higher cost. |
| `min_tau_spacing` | float | 1.25 | Minimum allowed ratio $\tau_{i+1}/\tau_i$. Pairs closer than this are merged. Must be > 1. |

#### Input / output

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resample` | bool | True | Resample 3-column input onto a geometric frequency grid. No effect on 5-column (pre-processed) data. |
| `n_resample` | int | 100 | Number of points in the resampled grid. |

---

## Output files

`solver.save(which, path)` writes results to `path/`. Two levels of output are
available.

### `which="base"` — main results

| File | Columns | Description |
|------|---------|-------------|
| `crs.dat` | `s  h(s)` | Continuous spectrum $h(s) = e^{H(s)}$. Header stores $G_0$ when `plateau=True`. |
| `drs.dat` | `g_i  tau_i` | Discrete Maxwell modes. Header stores $G_0$ when `plateau=True`. |
| `Gfit.dat` | `w  G'_cont  G''_cont  G'_disc  G''_disc` | Model fits from both CRS and DRS vs frequency. |

### `which="full"` — diagnostics (in addition to base)

| File | Columns | Description |
|------|---------|-------------|
| `logPlam.dat` | `lambda  logP` | Log Bayesian evidence $\log p(\lambda)$ vs $\lambda$. Peak locates $\lambda_M$. |
| `rho-eta.dat` | `lambda  rho  eta` | L-curve data: $\rho(\lambda)$ (data misfit) and $\eta(\lambda)$ (curvature penalty). |
| `aic.dat` | `wb  N_bst  AIC` | AIC scan results: optimal mode count $N$ and AIC value for each $w_b$. |

These three diagnostic files are silently skipped when `lam_C` is pre-specified
(the L-curve is not computed in that case).

---

## Plotting

```python
solver.plot(which="base")              # interactive display
solver.plot(which="full", toFile=True, path="output/")  # save PDFs
```

`which="base"` produces a two-panel figure:

- **Left**: $h(s) = e^{H(s)}$ with $\pm 2.5\,\Delta H$ error band, overlaid with
  discrete mode weights $g_i$ vs $\tau_i$.
- **Right**: experimental $G^*(\omega)$ data against continuous and discrete fits
  on a log-log axis.

`which="full"` adds a three-panel diagnostic figure: $\log p(\lambda)$ vs
$\lambda$, the $\rho$-$\eta$ L-curve with $\lambda^*$ marked, and the AIC scan.

---

The package is structured so that **all scientific computation is separated from I/O and plotting**. `continuous.py`, `discrete.py`, and `kernels.py` are pure numerical modules; they read no files and produce no figures. This makes them straightforward to call from other programs or notebooks.

---

## History

The code descends from the Matlab program
[ReSpect](https://www.mathworks.com/matlabcentral/fileexchange/40458-respect).

- **May 2026** — Major refactoring: packaged as an installable library with a
  clean `ReSpect` API; computation separated from I/O and plotting; TOML/YAML
  configuration; `ReSpectConfig` dataclass with validation.
- **March–April 2019** — Plateau modulus $G_0$ inference; Bayesian formulation
  for uncertainty in the CRS; more efficient L-curve sweep (3–4× speedup).
- **January 2019** — NLLS refinement of $\tau_i$ positions in the DRS;
  significantly improved discrete fit quality.
- **August 2018** — AIC-based mode selection and mode-merging algorithm
  incorporated from pyReSpect-time.
