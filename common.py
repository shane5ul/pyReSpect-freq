import numpy as np
from scipy.interpolate import interp1d

#
# last modified: August 2018
#

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

	#
	# Check Consistency of Input Parameters
	#
	
	#~ if par['BaseDistWt'] < 0. or par['BaseDistWt'] > 1. :
		#~ raise ValueError('The input parameter BaseDistWt needs to be between 0 and 1')
  
	#~ elif par['condWt'] < 0. or par['condWt'] > 1.:
		#~ raise ValueError('The input parameter condWt needs to be between 0 and 1')
	#~ else:
		#~ print('Input parameters look OK ...')			
			
			
	return par
	
def GetExpData(fname):

	"""Function: GetExpData(input)
	   Reads in the experimental data from the input file
	   Input:  fname = name of file that contains G(t) in 2 columns [t Gt]
	   Output: A n*1 vector "t", and a n*1 vector Gt"""
	   
	   
	wo, Gpo, Gppo = np.loadtxt(fname, unpack=True)  # 3 columns,  w - Gp - Gpp

	#
	# any repeated "time" values
	#
	
	wo, indices = np.unique(wo, return_index = True)
	Gpo         = Gpo[indices]
	Gppo        = Gppo[indices]

	#
	# Sanitize the input by spacing it out. Using linear interpolation
	#	
	fp  =  interp1d(wo, Gpo, fill_value="extrapolate")
	fpp =  interp1d(wo, Gppo, fill_value="extrapolate")

	w   =  np.logspace(np.log10(np.min(wo)), np.log10(np.max(wo)), max(len(wo),200))		
	Gp  =  fp(w)
	Gpp =  fpp(w)

	Gst =  np.append(Gp, Gpp)  # % Gp followed by Gpp (2n*1)

	return w, Gst

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
		
def kernel_prestore(H, kernMat):
	"""
	     turbocharging kernel function evaluation by prestoring kernel matrix
		
		 Date    : 8/17/2018
		 Function: kernel_prestore(input)
		 
		 Same as kernel, except prestoring hs, S, and W to improve speed 3x.
		
		 outputs the 2n*1 dimensional vector K(H)(w) which is comparable to G* = [G'|G"]'
		
		 Input: H = substituted CRS,
		        kernMat = 2n*ns matrix [(ws^2/1+ws^2) | (ws/1+ws)]'*hs
		        	
	"""
	return np.dot(kernMat, np.exp(H))	

