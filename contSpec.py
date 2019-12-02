#
# Help to find continuous spectrum: 
#
# March 2019 major update:
# (i)   added plateau modulus G0 (also in pyReSpect-time) calculation
# (ii)  following Hansen Bayesian interpretation of Tikhonov to extract p(lambda): 
# (iii) simplifying lcurve (starting from high lambda to low) - super cost savings!
# (iv)  still showing lamdaC from prior method (although abridged lambda scale) comparison
#

from common import *

# HELPER FUNCTIONS

def InitializeH(Gexp, s, kernMat,  *argv):
	"""
	 Function: InitializeH(input)
	
	 Input:  Gexp    = 2n*1 vector [G';G"],
	         s       = relaxation modes,
	         kernMat = matrix for faster kernel evaluation
		     G0      = optional; if plateau is nonzero	
	
	 Output: H = guessed H
			  G0 = optional guess if *argv is nonempty	
	"""
	
	#
	# To guess spectrum, pick a negative Hgs and a large value of lambda to get a
	# solution that is most determined by the regularization
	# March 2019; a single guess is good enough now, because going from large lambda to small
	#             lambda in lcurve.

	H    = -5.0 * np.ones(len(s)) + np.sin(np.pi * s)
	lam  = 1e0
	
	if len(argv) > 0:
		G0       = argv[0]
		Hlam, G0 = getH(lam, Gexp, H, kernMat, G0)		
		return Hlam, G0
	else:
		Hlam     = getH(lam, Gexp, H, kernMat)
		return Hlam


def getAmatrix(ns):
	"""Generate symmetric matrix A = L' * L required for error analysis:
	   helper function for lcurve in error determination"""
	# L is a ns*ns tridiagonal matrix with 1 -2 and 1 on its diagonal;
	nl = ns - 2
	L  = np.diag(np.ones(ns-1), 1) + np.diag(np.ones(ns-1),-1) + np.diag(-2. * np.ones(ns))
	L  = L[1:nl+1,:]
			
	return np.dot(L.T, L)
	
def getBmatrix(H, kernMat, Gexp, *argv):
	"""get the Bmatrix required for error analysis; helper for lcurve()
	   not explicitly accounting for G0 in Jr because otherwise I get underflow problems"""
	n   = int(len(Gexp)/2);
	ns  = len(H);
	nl  = ns - 2;
	r   = np.zeros(n);   	  # vector of size (n);

	# furnish relevant portion of Jacobian and residual
	Kmatrix = np.dot((1./Gexp).reshape(2*n,1), np.ones((1,ns)));
	Jr      = -kernelD(H, kernMat) * Kmatrix;    

	# if plateau then unfurl G0
	if len(argv) > 0:
		G0 = argv[0]
		r  = (1. - kernel_prestore(H, kernMat, G0)/Gexp)
	else:
		r = (1. - kernel_prestore(H, kernMat)/Gexp)
	
	B = np.dot(Jr.T, Jr) + np.diag(np.dot(r.T, Jr))

	return B

def oldLamC(par, lam, rho, eta):

	#
	# 8/1/2018: Making newer strategy more accurate and robust: dividing by minimum rho/eta
	# which is not as sensitive to lam_min, lam_max. This makes lamC robust to range of lam explored
	#
	#er = rho/np.amin(rho) + eta/np.amin(eta);
	er    = rho/np.amin(rho) + eta/(np.sqrt(np.amax(eta)*np.amin(eta)));

	#
	# Since rho v/s lambda is smooth, we can interpolate the coarse mesh to find minimum
	#
	# change 3/20/2019: Scipy 0.17 has a bug with extrapolation: so making lami tad smaller 
	lami = np.logspace(np.log10(min(lam)+1e-15), np.log10(max(lam)-1e-15), 1000)
	erri = np.exp(interp1d(np.log(lam), np.log(er), kind='cubic', bounds_error=False,
	                   fill_value=(np.log(er[0]), np.log(er[-1])))(np.log(lami)))


	ermin = np.amin(erri)
	eridx = np.argmin(erri)	
	lamC  = lami[eridx]
		
	#
	# 2/2: Copying 12/18 edit from pyReSpect-time;
	#      for rough data have cutoff at rho = rho_cutoff?
	#
	rhoF  = interp1d(lam, rho, bounds_error=False, fill_value=(rho[0], rho[-1]))

	if  rhoF(lamC) <= par['rho_cutoff']:
		try:
			eridx = (np.abs(rhoF(lami) - par['rho_cutoff'])).argmin()
			if lami[eridx] > lamC:
				lamC = lami[eridx]				
		except:
			pass

	return lamC

def lcurve(Gexp, Hgs, kernMat, par, *argv):
	"""
	 Function: lcurve(input)
	
	 Input: Gexp    = 2n*1 vector [Gt],
	        Hgs     = guessed H,
	        kernMat = matrix for faster kernel evaluation
	        par     = parameter dictionary
	        G0      = optionally
	        
	
	 Output: lamC and 3 vectors of size npoints*1 contains a range of lambda, rho
	         and eta. "Elbow"  = lamC is estimated using a *NEW* heuristic AND by Hansen method

	
		March 2019: starting from large lambda to small cuts calculation time by a lot
				also gives an error estimate 
	"""

	if par['plateau']:
		G0 = argv[0]

	npoints = int(par['lamDensity'] * (np.log10(par['lam_max']) - np.log10(par['lam_min'])))
	hlam    = (par['lam_max']/par['lam_min'])**(1./(npoints-1.))	
	lam     = par['lam_min'] * hlam**np.arange(npoints)
	eta     = np.zeros(npoints)
	rho     = np.zeros(npoints)
	logP    = np.zeros(npoints)
	H       = Hgs.copy()
	n       = len(Gexp)
	ns      = len(H)
	nl      = ns - 2
	logPmax = -np.inf					# so nothing surprises me!
	Hlambda = np.zeros((ns, npoints))

	# Error Analysis: Furnish A_matrix
	Amat       = getAmatrix(len(H))
	_, LogDetN = np.linalg.slogdet(Amat)
			
	#
	# This is the costliest step
	#
	for i in reversed(range(len(lam))):
		
		lamb    = lam[i]
		
		if par['plateau']:
			H, G0   = getH(lamb, Gexp, H, kernMat, G0)			
			rho[i]  = np.linalg.norm((1. - kernel_prestore(H, kernMat, G0)/Gexp))
			Bmat    = getBmatrix(H, kernMat, Gexp, G0)			
		else:
			H       = getH(lamb, Gexp, H, kernMat)
			rho[i]  = np.linalg.norm((1. - kernel_prestore(H,kernMat)/Gexp))
			Bmat    = getBmatrix(H, kernMat, Gexp)

		eta[i]       = np.linalg.norm(np.diff(H, n=2))
		Hlambda[:,i] = H

		_, LogDetC = np.linalg.slogdet(lamb*Amat + Bmat)
		V          =  rho[i]**2 + lamb * eta[i]**2		
					
		# this assumes a prior exp(-lam)
		logP[i]    = -V + 0.5 * (LogDetN + ns*np.log(lamb) - LogDetC) - lamb
		
		if(logP[i] > logPmax):
			logPmax = logP[i]
		elif(logP[i] < logPmax - 18):
			break		

	# truncate all to significant lambda
	lam  = lam[i:]
	logP = logP[i:]
	eta  = eta[i:]
	rho  = rho[i:]
	logP = logP - max(logP)

	Hlambda = Hlambda[:,i:]

	#
	# currently using both schemes to get optimal lamC
	# new lamM works better with actual experimental data  
	#
	lamC = oldLamC(par, lam, rho, eta)
	plam = np.exp(logP); plam = plam/np.sum(plam)
	lamM = np.exp(np.sum(plam*np.log(lam)))

	#
	# Dialling in the Smoothness Factor
	#
	if par['SmFacLam'] > 0:
		lamM = np.exp(np.log(lamM) + par['SmFacLam']*(max(np.log(lam)) - np.log(lamM)));
	elif par['SmFacLam'] < 0:
		lamM = np.exp(np.log(lamM) + par['SmFacLam']*(np.log(lamM) - min(np.log(lam))));

	#
	# printing this here for now because storing lamC for sometime only
	#
	if par['plotting']:
		plt.clf()
		plt.axvline(x=lamC, c='k', label=r'$\lambda_c$')
		plt.axvline(x=lamM, c='gray', label=r'$\lambda_m$')
		plt.ylim(-20,1)
		plt.plot(lam, logP, 'o-')
		plt.xscale('log')
		plt.xlabel(r'$\lambda$')
		plt.ylabel(r'$\log\,p(\lambda)$')
		plt.legend(loc='upper left')
		plt.tight_layout()
		plt.savefig('output/logP.pdf')

	return lamM, lam, rho, eta, logP, Hlambda

def getH(lam, Gexp, H, kernMat, *argv):
	"""
	 minimize_H  V(lambda) := ||Gexp - kernel(H)||^2 +  lambda * ||L H||^2

	 Input  : lambda  = regularization parameter ,
	          Gexp    = experimental data,
	          H       = guessed H,
  		      kernMat = matrix for faster kernel evaluation
  		      G0      = optional
	
	 Output : H_lam, [G0]
	          Default uses Trust-Region Method with Jacobian supplied by jacobianLM
	"""
	# send Hplus = [H, G0], on return unpack H and G0
	if len(argv) > 0:
		Hplus= np.append(H, argv[0])
		res_lsq = least_squares(residualLM, Hplus, jac=jacobianLM, args=(lam, Gexp, kernMat))
		return res_lsq.x[:-1], res_lsq.x[-1]
		
	# send normal H, and collect optimized H back
	else:
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

	# if plateau then unfurl G0
	if len(H) > ns:
		G0       = H[-1]
		H        = H[:-1]
		r[0:2*n] = (1. - kernel_prestore(H, kernMat, G0)/Gexp)
	else:
		r[0:2*n] = (1. - kernel_prestore(H,kernMat)/Gexp)
	
	# the curvature constraint is not affected by G0	
	r[2*n:2*n+nl] = np.sqrt(lam) * np.diff(H, n=2)  # second derivative
	
	return r
	
def jacobianLM(H, lam, Gexp, kernMat):
	"""
	
	 HELPER FUNCTION: Gets Jacobian J
		returns a (n+nl * ns) matrix Jr; (ns + 1) if G0 is also supplied.
	
		Jr_(i, j) = dr_i/dH_j
	
		It uses kernelD, which approximates dK_i/dH_j, where K is the kernel	
	
	"""
	
	n   = int(kernMat.shape[0]/2);
	ns  = kernMat.shape[1];
	nl  = ns - 2;

	# L is a nl*ns tridiagonal matrix with 1 -2 and 1 on its diagonal.
	L  = np.diag(np.ones(ns-1), 1) + np.diag(np.ones(ns-1),-1) + np.diag(-2. * np.ones(ns))	     
	L  = L[1:nl+1,:]


	Jr  = np.zeros((2*n + nl,ns))	
	 	
	#
	# Furnish the Jacobian Jr - (2n+nl)*ns matrix
	# Kmatrix is 2*n * ns matrix
	#
	Kmatrix = np.dot((1./Gexp).reshape(2*n,1), np.ones((1,ns)));

	if len(H) > ns:

		G0     = H[-1]
		H      = H[:-1]

		Jr  = np.zeros((2*n + nl,ns+1))	
		
		Jr[0:2*n, 0:ns]   = -kernelD(H, kernMat) * Kmatrix;
		Jr[0:n, ns]       = -1./Gexp[:n]						# nonzero dr_i/dG0 only for G'


		Jr[2*n:2*n+nl,0:ns] = np.sqrt(lam) * L;
		Jr[2*n:2*n+nl, ns]  = np.zeros(nl)						# column for dr_i/dG0 = 0
		
	else:

		Jr  = np.zeros((2*n + nl,ns))	

		Jr[0:2*n, 0:ns]     = -kernelD(H, kernMat) * Kmatrix;
		Jr[2*n:2*n+nl,0:ns] = np.sqrt(lam) * L;
    
	return	Jr

def kernelD(H, kernMat):
	"""
	 Function: kernelD(input)
	
	 outputs the 2n*ns dimensional vector DK(H)(w)
	 It approximates dK_i/dH_j = K * e(H_j):
	
	 Input: H       = substituted CRS,
	        kernMat = matrix for faster kernel evaluation
	 Output: DK     = Jacobian of H
		
	"""

	n   = int(kernMat.shape[0]/2);
	ns  = kernMat.shape[1];
		
	Hsuper  = np.dot(np.ones((2*n,1)), np.exp(H).reshape(1, ns))       
	DK      = kernMat * Hsuper
	
	return DK

# Furnish Globals that you will need for interactive plot

def getContSpec(par):
	
	# read input
	if par['verbose']:
		print('\n(*) Start\n(*) Loading Data File: {}...'.format(par['GexpFile']))

	w, Gexp = GetExpData(par['GexpFile'])

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

	kernMat = getKernMat(s, w)
		
	tic  = time.time()
	
	# get an initial guess for Hgs, G0
	if par['plateau']:
		Hgs, G0  = InitializeH(Gexp, s, kernMat, np.min(Gexp))		
	else:
		Hgs      = InitializeH(Gexp, s, kernMat)	
	
	if par['verbose']:
		te   = time.time() - tic
		print('\t({0:.1f} seconds)\n(*) Building the L-curve ...'.format(te), end="")	
		tic  = time.time()

	# Find Optimum Lambda with 'lcurve'
	if par['lamC'] == 0:
		if par['plateau']:
			lamC, lam, rho, eta, logP, Hlam = lcurve(Gexp, Hgs, kernMat, par, G0)
		else:
			lamC, lam, rho, eta, logP, Hlam = lcurve(Gexp, Hgs, kernMat, par)
	else:
		lamC = par['lamC']

	if par['verbose']:
		te = time.time() - tic
		print('({0:.1f} seconds)\n(*) Extracting CRS, ...\n\t... lamC = {1:0.3e}; '.
		      format(te, lamC), end="")
		tic = time.time()

	# Get the best spectrum	
	if par['plateau']:
		H, G0  = getH(lamC, Gexp, Hgs, kernMat, G0);
		print('G0 = {0:0.3e} ...'.format(G0), end="")
	else:
		H  = getH(lamC, Gexp, Hgs, kernMat);

	#----------------------
	# Print some datafiles
	#----------------------
	if par['verbose']:
		te = time.time() - tic
		print('done ({0:.1f} seconds)\n(*) Writing and Printing, ...'.format(te), end="")

		#~ # Save inferred H(s) and Gw
		#~ if par['plateau']:
			#~ K   = kernel_prestore(H, kernMat, G0);	
			#~ np.savetxt('output/H.dat', np.c_[s, H], fmt='%e', header='G0 = {0:0.3e}'.format(G0))
		#~ else:
			#~ K   = kernel_prestore(H, kernMat);
			#~ np.savetxt('output/H.dat', np.c_[s, H], fmt='%e')

		# Save inferred H(s) and Gw
		if par['lamC'] != 0:
			if par['plateau']:
				K   = kernel_prestore(H, kernMat, G0);	
				np.savetxt('output/H.dat', np.c_[s, H], fmt='%e', header='G0 = {0:0.3e}'.format(G0))
			else:
				K   = kernel_prestore(H, kernMat);
				np.savetxt('output/H.dat', np.c_[s, H], fmt='%e')
			
			np.savetxt('output/Gfit.dat', np.c_[w, K[:n], K[n:]], fmt='%e')

		# print Hlam, rho-eta, and logP if lcurve has been visited
		if par['lamC'] == 0:
			
			if os.path.exists("output/Hlam.dat"):
				os.remove("output/Hlam.dat")
				
			fHlam = open('output/Hlam.dat','ab')
			for i, lamb in enumerate(lam):
				np.savetxt(fHlam, Hlam[:,i])	
			fHlam.close()	

			# print logP
			np.savetxt('output/logPlam.dat', np.c_[lam, logP])
			
			# print rho-eta
			np.savetxt('output/rho-eta.dat', np.c_[lam, rho, eta], fmt='%e')

	#------------
	# Graphing
	#------------
	
	if par['plotting']:

		# plot spectrum "H.pdf" with errorbars
		plt.clf()

		plt.semilogx(s,H,'o-')
		plt.xlabel(r'$s$')
		plt.ylabel(r'$H(s)$')

		# error bounds are only available if lcurve has been implemented
		if par['lamC'] == 0:
			plam = np.exp(logP); plam = plam/np.sum(plam)			
			Hm   = np.zeros(len(s))
			Hm2  = np.zeros(len(s))
			cnt  = 0
			for i in range(len(lam)):	
				#~ Hm   += plam[i]*Hlam[:,i]
				#~ Hm2  += plam[i]*Hlam[:,i]**2
				# count all spectra within a threshold
				if plam[i] > 0.1:
					Hm   += Hlam[:,i]
					Hm2  += Hlam[:,i]**2
					cnt  += 1

			Hm = Hm/cnt
			dH = np.sqrt(Hm2/cnt - Hm**2)

			plt.semilogx(s,H+2.5*dH, c='gray', alpha=0.5)
			plt.semilogx(s,H-2.5*dH, c='gray', alpha=0.5)

			# save errorbar
			if par['verbose']:			
				if par['plateau']:
					K   = kernel_prestore(Hm, kernMat, G0);	
					np.savetxt('output/H.dat', np.c_[s, H, dH], fmt='%e', header='G0 = {0:0.3e}'.format(G0))
				else:
					K   = kernel_prestore(Hm, kernMat);	
					np.savetxt('output/H.dat', np.c_[s, H, dH], fmt='%e')

				np.savetxt('output/Gfit.dat', np.c_[w, K[:n], K[n:]], fmt='%e')
			
		plt.tight_layout()
		plt.savefig('output/H.pdf')





		plt.clf()
		
		if par['plateau']:
			K   = kernel_prestore(H, kernMat, G0);	
		else:
			K   = kernel_prestore(H, kernMat);
			
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
			#~ np.savetxt('output/rho-eta.dat', np.c_[lam, rho, eta], fmt='%e')

			plt.clf()
			plt.plot(rho, eta)
			rhost = np.interp(lamC, lam, rho)
			etast = np.interp(lamC, lam, eta)
			plt.plot(rhost, etast, 'o', c='k')
			plt.xscale('log')
			plt.yscale('log')
			
			
			plt.xlabel(r'$\rho$')
			plt.ylabel(r'$\eta$')
			plt.savefig('output/rho-eta.pdf')

	if par['verbose']:
		print('done\n(*) End\n')
		
	return H, lamC

def guiFurnishGlobals(par):

	from matplotlib import rcParams

	w, Gexp = GetExpData(par['GexpFile'])

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

	kernMat = getKernMat(s, w)

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


#	 
# Main Driver: This part is not run when contSpec.py is imported as a module
#              For example as part of GUI
#
if __name__ == '__main__':
	#
	# Read input parameters from file "inp.dat"
	#
	par = readInput('inp.dat')
	H, lamC = getContSpec(par)
