import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.optimize import least_squares

import common
import os
import time

plt.style.use('ggplot')		

#~ from matplotlib import rcParams
#~ rcParams['axes.labelsize'] = 28 
#~ rcParams['xtick.labelsize'] = 20
#~ rcParams['ytick.labelsize'] = 20 
#~ rcParams['legend.fontsize'] = 20
#~ rcParams['lines.linewidth'] = 2

# June 19, 2018: Trying to merge Gstar/contSpec.m and python/Gt/contspec.py into a python version
#                for Gstar

def InitializeH(Gexp, s, kernMat):
	"""%
	% Function: InitializeH(input)
	%
	% Input:  Gexp    = 2n*1 vector [G';G"],
	%         s       = relaxation modes,
	%         kernMat = matrix for faster kernel evaluation
	%
	% Output: H = guessed H
	%"""
	
	# To guess spectrum, pick a negative Hgs and a large value of lambda to get a
	# solution that is most determined by the regularization, then use that as
	# the next guess. 

	H    = -5.0 * np.ones(len(s)) + np.sin(np.pi * s)

	lam  = 1e0	
	Hlam = getH(lam, Gexp, H, kernMat)

	#
	# Successively improve the initial guess until you have are reasonably good
	# guess for low lambda
	#
	
	lam  = 1e-3;
	H    = getH(lam, Gexp, Hlam, kernMat)
	
	return H

def lcurve(Gexp, Hgs, kernMat, par):
	"""% 
	% Function: lcurve(input)
	%
	% Input: Gexp = 2n*1 vector [Gt],
	%        Hgs  = guessed H,
	%        kernMat = matrix for faster kernel evaluation
	%
	% Output: lamC and 3 vectors of size npoints*1 contains a range of lambda, rho
	% and eta. "Elbow"  = lamC is estimated using a heuristic.
	%
	%"""

	# take a coarse mesh: 2 lambda's per decade (auto)
	lam_max   = par['lam_max']
	lam_min   = par['lam_min']
	SmoothFac = par['SmFacLam']	
	
	npoints  = int(2 * (np.log10(lam_max) - np.log10(lam_min)))

	hlam    = (lam_max/lam_min)**(1./(npoints-1.))	
	lam     = lam_min * hlam**np.arange(npoints)

	eta     = np.zeros(npoints)
	rho     = np.zeros(npoints)
	H       = Hgs.copy()
	
	#
	# This step can be "parfor"ed in the future perhaps?
	#
	
	for i in range(len(lam)):
		lamb    = lam[i]
		H       = getH(lamb, Gexp, H, kernMat)
		rho[i]  = np.linalg.norm((1 - common.kernel_prestore(H, kernMat)/Gexp))
		eta[i]  = np.linalg.norm(np.diff(H, n=2))

	#
	# 8/1/2018: Making newer strategy more accurate and robust: dividing by minimum rho/eta
	# which is not as sensitive to lam_min, lam_max. This makes lamC robust to range of lam explored
	#
	
	#er = rho/np.amin(rho) + eta/np.amin(eta);
	er    = rho/np.amin(rho) + eta/(np.sqrt(np.amax(eta)*np.amin(eta)));
	#
	# Since rho v/s lambda is smooth, we can interpolate the coarse mesh to find minimum
	#
	
	lami = np.logspace(np.log10(min(lam)), np.log10(max(lam)), 1000)
	erri = np.exp(interp1d(np.log(lam), np.log(er), kind='cubic')(np.log(lami)))

	ermin = np.amin(erri)
	eridx = np.argmin(erri)
	lamC  = lami[eridx]	

	#
	# Dialling in the Smoothness Factor
	#

	if SmoothFac > 0:
		lamC = np.exp(np.log(lamC) + SmoothFac*(np.log(lam_max) - np.log(lamC)));
	elif SmoothFac < 0:
		lamC = np.exp(log(np.lamC) + SmoothFac*(np.log(lamC) - np.log(lam_min)));

	return lamC, lam, rho, eta

def getH(lam, Gexp, H, kernMat):

	"""% Purpose: Given a lambda, this function finds the H_lambda(s) that minimizes V(lambda)
	%
	%          V(lambda) := 1/n * ||Gexp - kernel(H)||^2 +  lambda/nl * ||L H||^2
	%
	% Input  : lambda  = regularization parameter ,
	%          Gexp    = experimental data,
	%          H       = guessed H,
    %          kernMat = matrix for faster kernel evaluation
	%
	% Output : H_lam
	%          Default uses Trust-Region Method
	%"""

	res_lsq = least_squares(residualLM, H, jac=jacobianLM, args=(lam, Gexp, kernMat))
	return res_lsq.x

def residualLM(H, lam, Gexp, kernMat):
	"""
	%
	% HELPER FUNCTION: Gets Residuals r
	%"""

	n   = int(kernMat.shape[0]/2);
	ns  = kernMat.shape[1];
	
	nl  = ns - 2;
	r   = np.zeros(2*n + nl);

	# 
	# Get the residual vector first
	# r = vector of size (n+nl,1); does DIFF work the same?
	#

	r[0:2*n]      = (1. - common.kernel_prestore(H,kernMat)/Gexp)/np.sqrt(n)  # the Gt and
	r[2*n:2*n+nl] = np.sqrt(lam) * np.diff(H, n=2)/np.sqrt(nl)  # second derivative
	
	return r
	
def jacobianLM(H, lam, Gexp, kernMat):
	"""
	%
	% HELPER FUNCTION: Gets Jacobian J
	%"""
	
	n   = int(kernMat.shape[0]/2);
	ns  = kernMat.shape[1];
	nl  = ns - 2;

	Jr  = np.zeros((2*n + nl,ns))

	#
	# L is a nl*ns tridiagonal matrix with 1
	# -2 and 1 on its diagonal.
	#
	
	L  = np.diag(np.ones(ns-1), 1) + np.diag(np.ones(ns-1),-1) + np.diag(-2. * np.ones(ns))	     
	L  = L[1:nl+1,:]
	 	
	#
	# Furnish the Jacobian Jr - (2n+nl)*ns matrix
	# Kmatrix is 2*n * ns matrix
	#
	Kmatrix             = np.dot((1./Gexp).reshape(2*n,1), np.ones((1,ns)))/np.sqrt(n);
	Jr[0:2*n, 0:ns]     = -kernelD(H, kernMat) * Kmatrix;
	Jr[2*n:2*n+nl,0:ns] = np.sqrt(lam) * L/np.sqrt(nl);

	return	Jr

def kernelD(H, kernMat):
	"""%
	% Function: kernelD(input)
	%
	% outputs the 2n*ns dimensional vector DK(H)(w)
	% approximates dK/dHj
	%
	% Input: H       = substituted CRS,
	%	     kernMat = matrix for faster kernel evaluation
	% Output: DK     = Jacobian of H
	%	
	%"""

	n   = int(kernMat.shape[0]/2);
	ns  = kernMat.shape[1];
		
	Hsuper  = np.dot(np.ones((2*n,1)), np.exp(H).reshape(1, ns))       
	DK      = kernMat * Hsuper
	
	return DK

# Furnish Globals that you will need for interactive plot
def guiFurnishGlobals(par):

	from matplotlib import rcParams

	w, Gexp = common.GetExpData(par['GexpFile'])

	if par['verbose']:
		print('(*) Initial Set up...', end="")

	# Set up some internal variables
	n    = len(w)
	ns   = par['ns']    # discretization of 'tau'

	wmin = w[0];
	wmax = w[n-1];

	# determine frequency window
	if par['FreqEnd'] == 1:
		smin = np.exp(-np.pi/2)/wmax; smax = np.exp(np.pi/2)/wmin		
	elif par['FreqEnd'] == 2:
		smin = 1./wmax; smax = 1./wmin				
	elif par['FreqEnd'] == 3:
		smin = np.exp(+np.pi/2)/wmax; smax = np.exp(-np.pi/2)/wmin

	hs   = (smax/smin)**(1./(ns-1))
	s    = smin * hs**np.arange(ns)

	kernMat = common.getKernMat(s, w)

	# toggle flags to prevent printing

	par['verbose'] = False
	par['plotting'] = False

	# load lamda, rho, eta
	lam, rho, eta = np.loadtxt('output/rho-eta.dat', unpack=True)

	# plot settings
	rcParams['axes.labelsize'] = 14 
	rcParams['xtick.labelsize'] = 12
	rcParams['ytick.labelsize'] = 12 
	rcParams['legend.fontsize'] = 12
	rcParams['lines.linewidth'] = 2

	plt.clf()

	return s, w, kernMat, Gexp, par, lam, rho, eta

def getContSpec(par):
	
	# read input
	if par['verbose']:
		print('\n(*) Start\n(*) Loading Data File: {}...'.format(par['GexpFile']))

	w, Gexp = common.GetExpData(par['GexpFile'])

	if par['verbose']:
		print('(*) Initial Set up...', end="")
  
	# Set up some internal variables
	n    = len(w)
	ns   = par['ns']    # discretization of 'tau'

	wmin = w[0];
	wmax = w[n-1];

	# determine frequency window
	if par['FreqEnd'] == 1:
		smin = np.exp(-np.pi/2)/wmax; smax = np.exp(np.pi/2)/wmin		
	elif par['FreqEnd'] == 2:
		smin = 1./wmax; smax = 1./wmin				
	elif par['FreqEnd'] == 3:
		smin = np.exp(+np.pi/2)/wmax; smax = np.exp(-np.pi/2)/wmin
		
	hs   = (smax/smin)**(1./(ns-1))
	s    = smin * hs**np.arange(ns)

	kernMat = common.getKernMat(s, w)
		
	tic  = time.time()
	Hgs  = InitializeH(Gexp, s, kernMat)
	
	if par['verbose']:
		te   = time.time() - tic
		print('\t({0:.1f} seconds)\n(*) Building the L-curve ...'.format(te), end="")	
		tic  = time.time()

	#
	# Find Optimum Lambda with 'lcurve'
	#
	
	if par['lamC'] == 0:
		lamC, lam, rho, eta = lcurve(Gexp, Hgs, kernMat, par)
	else:
		lamC = par['lamC']

	if par['verbose']:
		te = time.time() - tic
		print('{0:0.3e} ({1:.1f} seconds)\n(*) Extracting the continuous spectrum, ...'.format(lamC, te), end="")
		tic  = time.time()

	#
	# Get the spectrum	
	#
	
	H  = getH(lamC, Gexp, Hgs, kernMat);

	#
	# Print some datafiles
	#
	if par['verbose']:
		te = time.time() - tic
		print('done ({0:.1f} seconds)\n(*) Writing and Printing, ...'.format(te), end="")

		# create output directory if none exists
		if not os.path.exists("output"):
			os.makedirs("output")

		np.savetxt('output/H.dat', np.c_[s, H], fmt='%e')
		
		K   = common.kernel_prestore(H, kernMat);	
		np.savetxt('output/Gfit.dat', np.c_[w, K[:n], K[n:]], fmt='%e')

	#
	# Graphing
	#
	
	if par['plotting']:

		plt.clf()
		plt.semilogx(s, H,'o-')
		plt.xlabel('s')
		plt.ylabel('H(s)')
		plt.savefig('output/H.pdf')

		plt.clf()
		K = common.kernel_prestore(H, kernMat)
		plt.loglog(w, Gexp[:n],'o')
		plt.loglog(w, K[:n], 'k-')
		plt.loglog(w, Gexp[n:],'o')
		plt.loglog(w, K[n:], 'k-')
		plt.xlabel(r'$\omega$')
		plt.ylabel(r'$G^{*}$')
		plt.savefig('output/Gfit.pdf')

		# if lam not explicitly specified then print rho-eta.pdf
		try:
			lam
		except NameError:
		  print("lamC prespecified, so not printing rho-eta.pdf/dat")
		else:		
			np.savetxt('output/rho-eta.dat', np.c_[lam, rho, eta], fmt='%e')

			plt.clf()
			plt.plot(rho, eta)
			rhost = np.interp(lamC, lam, rho)
			etast = np.interp(lamC, lam, eta)
			plt.scatter(rhost, etast)
			plt.xscale('log')
			plt.yscale('log')
			
			
			plt.xlabel(r'$\rho$')
			plt.ylabel(r'$\eta$')
			plt.savefig('output/rho-eta.pdf')

	if par['verbose']:
		print('done\n(*) End\n')
		
	return H, lamC


#	 
# Main Driver: This part is not run when contSpec.py is imported as a module
#              For example as part of GUI
#
if __name__ == '__main__':
	#
	# Read input parameters from file "inp.dat"
	#
	par = common.readInput('inp.dat')
	H, lamC = getContSpec(par)
