# pyReSpect-freq

Extract continuous and discrete relaxation spectra from complex modulus G*(w)

## Files

### Code Files

This repository contains two python modules `contSpec.py` `discSpec.py`. They extract the continuous and discrete relaxation spectra from a stress relaxation data. (w versus G*(w) experiment or simulation).

It containts a third module `common.py` which contains utilities required by both `contSpec.py` and `discSpec.py`.

In addition to the python modules, jupyter notebooks `interactContSpec.ipynb`. This provisionally allows the user to experiment with parameter settings interactively.

### Input Files

The user is expected to supply two files:

+ `inp.dat` is used to control parameters and settings
+ `Gst.dat` which contains three columns of data `w`, `G'` and, `G"`

### Output Files

Text files containting output from the code are stored in a directory `output/`. These include a fit of the data, the spectra, and other files relevant to the continuous or discrete spectra calculation. 

Graphical and onscreen output can be suppressed by appropriate flags in `inp.dat`.

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

The numbers in parenthesis show the version this has been tested on. 

python3 (3.5.2)
numpy (1.14.2)
scipy (1.0.1)

For interactive mode:

jupyter (4.3)
ipywidgets (6.0.0)

## History

The code is based on the Matlab program [ReSpect](https://www.mathworks.com/matlabcentral/fileexchange/40458-respect), which extract the continuous and discrete relaxation spectra from frequency data, G*(w).

### Major Upgrage: March-Apr 2019
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

