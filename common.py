#
# July 2023: allowing for optional 2 additional columns to specify error in G' and G"
#
# last modified: Feb 2019
#

#
# Global Imports and Plot Settings
#

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz, quad
from scipy.optimize import nnls, minimize, least_squares

import time
import os

#
#
# plotting preferences: change this block to suit your taste
#

plt.style.use('ggplot')		

#~ try:
	#~ import seaborn as sns
#~ except ImportError:
	#~ plt.style.use('ggplot')		
#~ else:
	#~ plt.style.use('seaborn-ticks')
	#~ sns.set_color_codes()
	#~ sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

from matplotlib import rcParams
rcParams['axes.labelsize'] = 28 
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20 
rcParams['legend.fontsize'] = 20
rcParams['lines.linewidth'] = 2

#
# Functions common to both discrete and continuous spectra
#

def readInput(fname='inp.dat'):
	"""Reads data from the input file (default = 'inp.dat')
	   and populates the parameter dictionary par"""
	
	par  = {}

	# read the input file
	for line in open(fname):

		li=line.strip()

		# if not empty or comment line; currently list or float
		if len(li) > 0 and not li.startswith("#" or " "):

			li = line.rstrip('\n').split(':')
			key = li[0].strip()
			tmp = li[1].strip()

			val = eval(tmp)

			par[key] = val		
			
	# create output directory if none exists
	if not os.path.exists("output"):
		os.makedirs("output")

			
	return par
	
def GetExpData(fname):

	"""Function: GetExpData(input)
	   Reads in the experimental data from the input file
	   Input:  fname = name of file that contains G(t) in 2 columns [t Gt]
	   Output: A n*1 vector "t", and a n*1 vector Gt"""
	   

	try:
		data = np.loadtxt(fname)
		cols = data.shape[1]		# number of columns in data file

		# if only 3 columns, then set weights to 1.0
		if cols == 3:
			wo   = data[:,0]
			Gpo  = data[:,1]
			Gppo = data[:,2]
		else:
			wo   = data[:,0]
			Gpo  = data[:,1]
			Gppo = data[:,2]
			wG1  = data[:,3]
			wG2  = data[:,4]

	except OSError:
		print('*Error*: Gst data file is either not in the correct path, or incorrectly formatted')
		quit()

	#
	# any repeated "time" values
	#	
	wo, indices = np.unique(wo, return_index = True)
	Gpo         = Gpo[indices]
	Gppo        = Gppo[indices]
	if cols > 3:
		wG1 = wG1[indices]
		wG2 = wG2[indices]


	#
	# if three columns are provided then assume that the data is preprocessed
	# and it does not need any interpolation; 
	#
	# Sanitize the input by spacing it out. Using linear interpolation
	#
	if cols == 3:
		fp  =  interp1d(wo, Gpo, fill_value="extrapolate")
		fpp =  interp1d(wo, Gppo, fill_value="extrapolate")

		w   =  np.geomspace(np.min(wo), np.max(wo), 100)		
		Gp  =  fp(w)
		Gpp =  fpp(w)

		Gst =  np.append(Gp, Gpp)  # % Gp followed by Gpp (2n*1)

		return w, Gst, np.ones(len(Gst))
	else:
		return wo, np.append(Gpo, Gppo), np.append(wG1, wG2)


def getKernMat(s, w):
	"""furnish kerMat() which helps faster kernel evaluation, given s, w
	   Generates a 2n*ns matrix [(ws^2/1+ws^2) | (ws/1+ws)]'*hs, which can be 
	   multiplied with exp(H) to get predicted G*"""
	   
	ns          = len(s)
	hsv         = np.zeros(ns);

	hsv[0]      = 0.5 * np.log(s[1]/s[0])
	hsv[ns-1]   = 0.5 * np.log(s[ns-1]/s[ns-2])
	hsv[1:ns-1] = 0.5 * (np.log(s[2:ns]) - np.log(s[0:ns-2]))

	S, W        = np.meshgrid(s, w);
	ws          = S*W
	ws2         = ws**2
	
	return np.vstack((ws2/(1+ws2), ws/(1+ws2))) *hsv
		
def kernel_prestore(H, kernMat, *argv):
	"""
	     turbocharging kernel function evaluation by prestoring kernel matrix
		 Date    : 8/17/2018
		 Function: kernel_prestore(input) returns K*h, where h = exp(H)
		 
		 Same as kernel, except prestoring hs, S, and W to improve speed 3x.
		
		 outputs the 2n*1 dimensional vector K(H)(w) which is comparable to G* = [G'|G"]'
		 3/11/2019: returning Kh + G0
		 		
		 Input: H = substituted CRS,
		        kernMat = 2n*ns matrix [(ws^2/1+ws^2) | (ws/1+ws)]'*hs
		        	
	"""
	if len(argv) > 0:
		n = int(kernMat.shape[0]/2);
		G0v = np.zeros(2*n)
		G0v[:n] = argv[0]
	else:
		G0v = 0.
	
	return np.dot(kernMat, np.exp(H)) + G0v

