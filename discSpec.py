#
# Function: discSpec(par)
#
# Uses the continuous relaxation spectrum extracted using contSpec()
# to determine an approximate discrete approximation.
#
# Input: Communicated by the datastructure "par"
#
#        fGstFile = name of file that contains G*(w) in 3 columns [w Gp Gpp]
#                   default: 'Gst.dat' is assumed.
#        verbose  = true, then prints onscreen messages, and prints datafiles
#        plotting = true, then plots to stdio.
#
#
# Output: Nopt    = optimum number of discrete modes
#         [g tau] = spectrum
#         error   = error norm of the discrete fit
#        
#         dmodes.dat : Prints the [g tau] for the particular Nopt
#         aic.dat    : [N error aic]
#         Gfitd.dat  : The discrete G* for Nopt [w Gp Gpp] 
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz, quad
from scipy.optimize import minimize


import os
import time
from common import *
np.set_printoptions(precision=2)

plt.style.use('ggplot')

def MaxwellModes(z, w, Gexp):
	"""
	%
	% Function: MaxwellModes(input)
	%
	% Solves the linear least squares problem to obtain the DRS
	%
	% Input: z = points distributed according to the density,
	%        t    = n*1 vector contains times,
	%        Gexp = 2n*1 vector contains Gp and Gpp
	%
	% Output: g, tau = spectrum  (array)
	%         error = relative error between the input data and the G(t) inferred from the DRS
	%         condKp = condition number
	%
	"""

	N      = len(z)
	tau    = np.exp(z)
	n      = len(w)

	#
	# Prune small -ve weights g(i)
	#
	g, error, condKp = nnLLS(w, tau, Gexp)

	izero = np.where(g < 1e-8)
	tau   = np.delete(tau, izero)
	g     = np.delete(g, izero)

	return g, tau, error, condKp

def nnLLS(w, tau, Gexp):
	"""
	#
	# Helper subfunction which does the actual LLS problem
	# helps MaxwellModes
	#
	"""
	from scipy.optimize import nnls
	
	n       = len(Gexp)
	S, W    = np.meshgrid(tau, w)
	ws      = S*W
	ws2     = ws**2
	K       = np.vstack((ws2/(1+ws2), ws/(1+ws2)))   # 2n * nmodes
		
	#
	# gets (Gt/GtE - 1)^2, instead of  (Gt -  GtE)^2
	#

	Kp      = np.dot(np.diag((1./Gexp)), K)
	condKp  = np.linalg.cond(Kp)
	g       = nnls(Kp, np.ones(len(Gexp)))[0]
		
	GtM   	= np.dot(K, g)
	error 	= np.sum((GtM/Gexp - 1.)**2)

	return g, error, condKp

def GetWeights(H, w, s, wb):

	"""
	%
	% Function: GetWeights(input)
	%
	% Finds the weight of "each" mode by taking a weighted average of its contribution
	% to Gp and Gpp, mixed with an even distribution given by `wb`
	%
	% Input: H = CRS (ns * 1)
	%        w = n*1 vector contains times
	%        s = relaxation modes (ns * 1)
	%       wb = weightBaseDist
	%
	% Output: wt = weight of each mode
	%
	"""
  
	ns         = len(s)
	n          = len(w)

	hs         = np.zeros(ns)
	wt         = hs
	
	hs[0]      = 0.5 * np.log(s[1]/s[0])
	hs[ns-1]   = 0.5 * np.log(s[ns-1]/s[ns-2])
	hs[1:ns-1] = 0.5 * (np.log(s[2:ns]) - np.log(s[0:ns-2]))

	S, W       = np.meshgrid(s, w);
	ws         = S*W
	ws2        = ws**2

	kern       = np.vstack((ws2/(1+ws2), ws/(1+ws2)))
	wij        = np.dot(kern, np.diag(hs * np.exp(H)))  # 2n * ns
	K          = np.dot(kern, hs * np.exp(H)) # 2n * 1, comparable with Gexp

	for i in np.arange(n):
		wij[i,:] = wij[i,:]/K[i]

	for j in np.arange(ns):
		wt[j] = np.sum(wij[:,j])

	wt  = wt/np.trapz(wt, np.log(s))
	wt  = (1. - wb) * wt + (wb * np.mean(wt)) * np.ones(len(wt))

	return wt

def GridDensity(x, px, N):

	"""#
	#  PROGRAM: GridDensity(input)
	#
	#	Takes in a PDF or density function, and spits out a bunch of points in
	#       accordance with the PDF
	#
	#  Input:
	#       x  = vector of points. It need *not* be equispaced,
	#       px = vector of same size as x: probability distribution or
	#            density function. It need not be normalized but has to be positive.
	#  	    N  = Number of points >= 3. The end points of "x" are included
	#  	     necessarily,
	# 
	#  Output:
	#       z  = Points distributed according to the density
	#       hz = width of the "intervals" - useful to apportion domain to points
	#            if you are doing quadrature with the results, for example.
	#
	#  (c) Sachin Shanbhag, November 11, 2015
	#"""

	npts = 100;                              # can potentially change
	xi   = np.linspace(min(x),max(x),npts)   # reinterpolate on equi-spaced axis
	fint = interp1d(x,px,'cubic')	         # smoothen using cubic splines
	pint = fint(xi)        					 # interpolation
	ci   = cumtrapz(pint, xi, initial=0)                
	pint = pint/ci[npts-1]
	ci   = ci/ci[npts-1]                     # normalize ci

	alfa = 1./(N-1)                          # alfa/2 + (N-1)*alfa + alfa/2
	zij  = np.zeros(N+1)                     # quadrature interval end marker
	z    = np.zeros(N)                       # quadrature point

	z[0]    = min(x);  
	z[N-1]  = max(x); 

	#
	# ci(Z_j,j+1) = (j - 0.5) * alfa
	#
	beta       = np.arange(0.5, N-0.5) * alfa
	zij[0]     = z[0]
	zij[N]     = z[N-1]
	fint       = interp1d(ci, xi, 'cubic')
	zij[1:N]   = fint(beta)
	h          = np.diff(zij)

	#
	# Quadrature points are not the centroids, but rather the center of masses
	# of the quadrature intervals
	#

	beta     = np.arange(1, N-1) * alfa
	z[1:N-1] = fint(beta)

	return z, h

def ReadData(par, fNameH = 'output/H.dat'):
	"""
		Read experimental data, and the continuous spectrum
	"""

	# Read experimental data
	w, Gexp = GetExpData(par['GexpFile'])

	# Read continuous spectrum
	s, H    = np.loadtxt(fNameH, unpack=True)

	return w, Gexp, s, H
	
def initializeDiscSpec(par):
	
	# read input; initialize parameters
	if par['verbose']:
		print('\n(*) Start\n(*) Loading Data Files: ... {}...'.format(par['GexpFile']))

	w, Gexp, s, H = ReadData(par)
	
	n    = len(w);
	ns   = len(s);
	
	# range of N scanned
	Nmax  = min(np.floor(3.0 * np.log10(max(w)/min(w))),n/4); # maximum N
	Nmin  = max(np.floor(0.5 * np.log10(max(w)/min(w))),3);   # minimum N
	Nv    = np.arange(Nmin, Nmax + 1).astype(int)

	# Estimate Error Weight from Continuous Curve Fit
	kernMat = getKernMat(s, w)
	Gc      = kernel_prestore(H, kernMat);
	Cerror  = 1./(np.std(Gc/Gexp - 1.))  #	Cerror = 1.?
	
	return w, Gexp, s, H, Nv, Gc, Cerror

def getDiscSpecMagic(par):
	"""Assume magic parameter is always turned on."""
	
	w, Gexp, s, H, Nv, Gc, Cerror = initializeDiscSpec(par)

	n    = len(w);
	ns   = len(s);
	npts = len(Nv)

	# range of wtBaseDist scanned
	wtBase = par['deltaBaseWeightDist'] * np.arange(1, int(1./par['deltaBaseWeightDist']))
	AICbst = np.zeros(len(wtBase))
	Nbst   = np.zeros(len(wtBase))  # nominal number of modes
	nzNbst = np.zeros(len(wtBase))  # actual number of nonzero modes
	
		
	# main loop over wtBaseDist
	for ib, wb in enumerate(wtBase):
		
		# Find the distribution of nodes you need
		wt  = GetWeights(H, w, s, wb)

		# Scan the range of number of Maxwell modes N = (Nmin, Nmax) 
		ev    = np.zeros(npts)
		nzNv  = np.zeros(npts)  # number of nonzero modes 

		for i, N in enumerate(Nv):
			z, hz  = GridDensity(np.log(s), wt, N)         # select "tau" Points
			g, tau, ev[i], _ = MaxwellModes(z, w, Gexp)    # get g_i
			nzNv[i]                 = len(g)

		# store the best solution for this particular wb
		AIC        = 2. * Nv + 2. * Cerror * ev
		
		AICbst[ib] = min(AIC)
		Nbst[ib]   = Nv[np.argmin(AIC)]
		nzNbst[ib] = nzNv[np.argmin(AIC)]
		
	# global best settings of wb and Nopt; note this is nominal Nopt (!= len(g) due to NNLS)
	Nopt  = int(Nbst[np.argmin(AICbst)])
	wbopt = wtBase[np.argmin(AICbst)]

	#
	# Recompute the best data-set stats
	#
	wt                 = GetWeights(H, w, s, wbopt)
	z, hz              = GridDensity(np.log(s), wt, Nopt)   # Select "tau" Points
	g, tau, error, cKp = MaxwellModes(z, w, Gexp)   		# Get g_i, taui

	#
	# Check if modes are close enough to merge
	#
	tauSpacing = tau[1:]/tau[:-1]
	while min(tauSpacing) < par['minTauSpacing']:
		imode      = np.argmin(tauSpacing)      # merge modes imode and imode + 1
		g, tau     = mergeModes_magic(g, tau, imode)
		tauSpacing = tau[1:]/tau[:-1]

	if par['verbose']:
		print('\n(*) Number of optimum nodes = {0:d}\n'.format(len(g)))

	#
	# Some Plotting
	#
	if par['plotting']:

		plt.clf()
		plt.plot(wtBase, AICbst, label='AIC')
		plt.plot(wtBase, nzNbst, label='Nbst')
		plt.scatter(wbopt, len(g), color='k')
		plt.scatter(wbopt, np.min(AICbst), color='k')
		plt.yscale('log')
		plt.xlabel('baseDistWt')
		plt.legend()
		plt.tight_layout()
		plt.savefig('output/AIC.pdf')		


		plt.clf()
		plt.loglog(tau,g,'o-', label='disc')
		plt.loglog(s, np.exp(H), label='cont')
		plt.xlabel('tau')
		plt.ylabel('g')
		plt.legend(loc='lower right')
		plt.tight_layout()
		plt.savefig('output/dmodes.pdf')			


		plt.clf()
		S, W    = np.meshgrid(tau, w)
		ws      = S*W
		ws2     = ws**2
		K       = np.vstack((ws2/(1+ws2), ws/(1+ws2)))   # 2n * nmodes
		GstM   	= np.dot(K, g)
		
		plt.loglog(w, Gexp[:n],'o')
		plt.loglog(w, Gexp[n:],'o')

		plt.loglog(w, GstM[n:], 'g-')		
		plt.loglog(w, GstM[:n], 'g-', label='disc')

		# continuous curve	
		plt.loglog(w, Gc[:n], 'k--', label='cont')
		plt.loglog(w, Gc[n:], 'k--')
		

		plt.xlabel(r'$\omega$')
		plt.ylabel(r'$G^{*}$')
		plt.legend()
		plt.savefig('output/Gfitd.pdf')
  
	#
	# Some Printing
	#
	if par['verbose']:
		
		print('(*) Condition number of matrix equation: {0:e}\n'.format(cKp))

		print('\n\t\tModes\n\t\t-----\n\n')
		print('i \t    g(i) \t    tau(i)\n')
		print('---------------------------------------\n')
		
		for i in range(len(g)):
			print('{0:3d} \t {1:.5e} \t {2:.5e}'.format(i+1,g[i],tau[i]))
		print("\n")

		np.savetxt('output/dmodes.dat', np.c_[g, tau], fmt='%e')
		np.savetxt('output/aic.dat', np.c_[wtBase, nzNbst, AICbst], fmt='%f\t%i\t%e')


		S, W    = np.meshgrid(tau, w)
		ws      = S*W
		ws2     = ws**2
		K       = np.vstack((ws2/(1+ws2), ws/(1+ws2)))   # 2n * nmodes
		GstM   	= np.dot(K, g)		
		
		np.savetxt('output/Gfitd.dat', np.c_[w, GstM[:n], GstM[n:]], fmt='%e')

	return Nopt, g, tau, error

def normKern_magic(w, gn, taun, g1, tau1, g2, tau2):
	"""helper function: for costFcn and mergeModes
	   used only when magic = True"""
	wt   = w*taun
	Gnp  = gn * (wt**2)/(1. + wt**2)
	Gnpp = gn *  wt/(1. + wt**2)

	wt   = w*tau1
	Gop  = g1 * (wt**2)/(1. + wt**2)
	Gopp = g1 *  wt/(1. + wt**2)
	
	wt    = w*tau2
	Gop  += g2 * (wt**2)/(1. + wt**2)
	Gopp += g2 *  wt/(1. + wt**2)
	
	return (Gnp/Gop - 1.)**2 + (Gnpp/Gopp - 1.)**2

def costFcn_magic(par, g, tau, imode):
	""""helper function for mergeModes; establishes cost function to minimize
		   used only when magic = True"""

	gn   = par[0]
	taun = par[1]

	g1   = g[imode]
	g2   = g[imode+1]
	tau1 = tau[imode]
	tau2 = tau[imode+1]

	wmin = min(1./tau1, 1./tau2)/10.
	wmax = max(1./tau1, 1./tau2)*10.

	return quad(normKern_magic, wmin, wmax, args=(gn, taun, g1, tau1, g2, tau2))[0]

def mergeModes_magic(g, tau, imode):
	"""merge modes imode and imode+1 into a single mode
	   return gp and taup corresponding to this new mode
	   used only when magic = True"""

	from scipy.integrate import quad
	from scipy.optimize import minimize	




	iniGuess = [g[imode] + g[imode+1], 0.5*(tau[imode] + tau[imode+1])]
	res = minimize(costFcn_magic, iniGuess, args=(g, tau, imode))

	newtau        = np.delete(tau, imode+1)
	newtau[imode] = res.x[1]

	newg          = np.delete(g, imode+1)
	newg[imode]   = res.x[0]
		
	return newg, newtau
	

#############################
#
# M A I N  P R O G R A M
#
#############################

if __name__ == '__main__':
	#
	# Read input parameters from file "inp.dat"
	#
	par = readInput('inp.dat')
	_ = getDiscSpecMagic(par)
