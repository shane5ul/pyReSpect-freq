# pyReSpect-freq

Extract the continuous and discrete relaxation spectra from complex modulus G*(w). Papers which describe the method are:

+ Shanbhag, S., "Relaxation spectra using nonlinear Tikhonov regularization with a Bayesian criterion", Rheologica Acta, **2020**,  59, 509 [doi: 10.1007/s00397-020-01212-w].
+ Shanbhag, S., "pyReSpect: A Computer Program to Extract Discrete and Continuous Spectra from Stress Relaxation Experiments", Macromolecular Theory and Simulations, **2019**, 1900005 [doi: 10.1002/mats.201900005].
+ Takeh, A. and Shanbhag, S. "A computer program to extract the continuous and discrete relaxation spectra from dynamic viscoelastic measurements", Appl. Rheol. **2013**, 23, 24628.

## Files

### Code Files

This repository contains two python modules `contSpec.py` `discSpec.py`. They extract the continuous and discrete relaxation spectra from a stress relaxation data. (w versus G*(w) experiment or simulation).

It containts a third module `common.py` which contains utilities required by both `contSpec.py` and `discSpec.py`.

In addition to the python modules, jupyter notebooks `interactContSpec.ipynb`. This provisionally allows the user to experiment with parameter settings interactively.

### Input Files

The user is expected to supply two files:

+ `inp.dat` is used to control parameters and settings
+ `Gst.dat` which contains three columns of data `w`, `G'` and, `G"` [07/2023: optionally 2 more columns specifying weights]

### Output Files

Graphical and onscreen output can be suppressed by appropriate flags in `inp.dat`. Text files and PDFs of graphs containing output from the code are stored in a directory `output/`. 

These include a fit of the data, the spectra, and other files relevant to the continuous or discrete spectra calculation. 

Contents of the files are described below. The main output files are:

1. `H.dat`: This file contains 2 or 3 columns. 
    * The first two columns are $\tau$ and $H(\tau)$. The third column, when it is present, provides errorbars $\Delta H(\tau)$. These are plotted as `H.pdf`.
    * $H(\tau)$ is the natural logarithm of the CRS; exponentiate to get the CRS $h(\tau) = \exp(H(\tau))$.
    * When `plateau` is set to true (viscoelastic solids), then the header of this file stores the plateau modulus `G0`.
2. `dmodes.dat`: This file contains 2 columns describing the DRS
    * The fist column is $g_i$ and the second column is $\tau_i$.
    * When `plateau` is set to true (viscoelastic solids), then the header of this file stores the plateau modulus `G0`.
    * `dmodes.pdf` plots the DRS and compares it with the CRS $h(\tau) = e^{H(\tau)}$.
3. `Gfit.dat`: This file contains three columns.
    * the first column is frequency $\omega$, the second and third columns are  $G'$ and $G''$ corresponding to the **CRS**.
    * `Gfit.pdf` compares the fits with the interpolated experimental data.
4. `Gfitd.dat`: This file contains three columns.
    * the first column is frequency $\omega$, the second and third columns are  $G'$ and $G''$ corresponding to the **DRS**.
    * `Gfitd.pdf` compares the fits with the interpolated experimental data. It also shows the fit of the CRS.

The remaining output files may be helpful for diagnostics:

1. `logPlam.dat`: This file contains two columns, which store information used to figure out the optimal $\lambda$ for CRS extraction.
    * The first column is the set of $\lambda$ explored, and the second column is the posterior probability that $\lambda$, $p(\lambda)$.
    * `logP.pdf` plots this data and identifies the location of the optimal $\lambda$ near the peak of $p(\lambda)$. It uses 
2. `rho-eta.dat`: This file contains three columns $\lambda$, $\rho(\lambda)$, and $\eta(\lambda)$
    * $\lambda$ is the regularization parameter in Tikhonov regularization. $\rho$ measures the accuracy of the data fit, and $\eta$ measures the curvature (penalty).
    * The file `rho-eta.pdf` plots $\eta$ versus $\rho$ and marks the  optimal $\lambda^{*}$
3. `aic.dat`: This file has three columns that contain information used to figure out the optimal number of discrete modes.
    * the three columns in order are: baseDistWt, Noptimal, AIC
    * they are plotted in `AIC.pdf` which also shows the optimal choice.

### Test Files

A bunch of test files are supplied in the folder `tests/`.

## Usage

Once `inp.dat` and `Gst.dat` are furnished, running the code is simple.

To get the continuous spectrum:

`python3 contSpec.py`

The **continuous spectrum must be extracted before the discrete spectrum** is computed. The discrete spectrum can then be calculated by

`python3 discSpec.py`

### Interactive Mode

The interactive mode offers a "GUI" for exploring parameter settings. To launch use `jupyter notebook interactContSpec.ipynb`.

### Pre-requisites

The numbers in parenthesis show the latest versions this program has been tested on.

python3 (3.8)
numpy (1.24)
scipy (1.10)

For interactive mode:

jupyter-core (4.6.3)
jupyter-notebook (6.0.3)
ipywidgets (8.1.1)

## History

The code is based on the Matlab program [ReSpect](https://www.mathworks.com/matlabcentral/fileexchange/40458-respect), which extract the continuous and discrete relaxation spectra from frequency data, G*(w).

### Major Upgrade: March-Apr 2019
+ added ability to infer plateau modulus G0; modified all python routines and reorganized inp.dat
+ use a Bayesian formulation to infer uncertainty in the continuous spectrum
+ currently keeping old method to determine critical lambda, but using a far more efficient method (3-4x savings in compute time)
+ made discSpec.py compliant with G0
+ lots of bug-fixes
+ organized tests/ folder better.


### Major Upgrade: Jan 2019

+ moved all common imports and definitions to common; made completely visible
+ in discSpec(): added a NLLS routine to optimize tau; use previous optima as initial guesses for final tau; this greatly improved the quality of fits.

### Major Upgrade: August 2018

+ Incorporating changes from pyReSpect-time into frequency calculation
+ discrete spectrum only "magic" mode from now on, since it works so well
+ Major improvements in discrete spectrum (AIC/mergemodes)

