import numpy as np
import hopkins_pdf
import pylab as pl

Tvals = np.concatenate([[0],np.logspace(-3,0,5)])
sigmas = np.linspace(0.25,4,5)

keys = hopkins_pdf.moments_theoretical_hopkins(0, 1, 1, meanrho=1).keys()

theoretical_moments = {T:{s:hopkins_pdf.moments_theoretical_hopkins(0, s, T, meanrho=1) for s in sigmas} for T in Tvals}

grid = {k:np.array([[theoretical_moments[T][s][k] for T in Tvals] for s in sigmas]) for k in keys}

pl.figure(1)
pl.clf()
for ii,k in enumerate(keys):
    pl.subplot(2,4,ii)
    pl.imshow(grid[k])
    pl.title(k)
    pl.colorbar()

rho = np.logspace(-16,11,50000,base=np.e)

real_moments = {T:{s:hopkins_pdf.moments(rho, s, T, meanrho=1) for s in sigmas} for T in Tvals}

real_grid = {k:np.array([[real_moments[T][s][k] for T in Tvals] for s in sigmas]) for k in keys}

pl.figure(2)
pl.clf()
for ii,k in enumerate(keys):
    pl.subplot(2,4,ii)
    pl.imshow(grid[k])
    pl.title(k)
    pl.colorbar()

#for T in [0,0.001,0.01,0.1,1]:
#    for sigma in [0.5,1,2,4]:
#        m = moments(rho,sigma,T)
#        print "T=%6g S=%6g delta-sigma-V = %g" % (T, sigma, m['S_rho,V'] - m['S_rho,Vb'])

#        for k in m:
#                print sigma,k,round(moments_theoretical_hopkins(rho,sigma,T)[k]-moments(rho,sigma,T)[k],5)

