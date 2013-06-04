import numpy as np
import hopkins_pdf
import pylab as pl

pl.rc('font',size=14)

Tvals = np.concatenate([[0],np.logspace(-3,0,5)])
sigmas = np.array([0.25,0.5,1.0,2.0,4.0])
meanrhos = np.logspace(0,5,6)

keys = hopkins_pdf.moments_theoretical_hopkins(0, 1, 1, meanrho=1).keys()

theoretical_moments = {r:{T:{s:hopkins_pdf.moments_theoretical_hopkins(0, s, T, meanrho=r) for s in sigmas} for T in Tvals} for r in meanrhos}

grid = {k:np.array([[[theoretical_moments[r][T][s][k] for T in Tvals] for s in sigmas] for r in meanrhos]) for k in keys}

pl.figure(1)
pl.clf()
for ii,k in enumerate(keys):
    pl.subplot(2,4,ii)
    pl.imshow(grid[k][:,:,0])
    pl.title(k)
    pl.colorbar()

rho = np.logspace(-16,11,50000,base=np.e)

real_moments = {r:{T:{s:hopkins_pdf.moments(rho, s, T, meanrho=r) for s in sigmas} for T in Tvals} for r in meanrhos}

real_grid = {k:np.array([[[real_moments[r][T][s][k] for T in Tvals] for s in sigmas] for r in meanrhos]) for k in keys}

pl.figure(2)
pl.clf()
for ii,k in enumerate(keys):
    pl.subplot(2,4,ii)
    pl.imshow(grid[k][:,:,0])
    pl.title(k)
    pl.colorbar()

pl.figure(3)
pl.clf()
for ii,k in enumerate(['<rho>_V','<rho>_M','<ln rho>_V','<ln rho>_M']):
    pl.subplot(2,2,ii)
    if 'ln' in k:
        pl.plot(np.log10(meanrhos), grid[k][:,1,1]/np.log(10), linewidth=2, alpha=0.5)
        pl.ylabel(k)
    else:
        pl.plot(np.log10(meanrhos), np.log10(grid[k][:,1,1]), linewidth=2, alpha=0.5)
        pl.ylabel("log10({})".format(k))
    pl.plot(pl.gca().get_xlim(),pl.gca().get_xlim(),'k--',alpha=0.5)
    pl.xlabel("$\log_{10}(\\rho_0$)")

pl.show()



#for T in [0,0.001,0.01,0.1,1]:
#    for sigma in [0.5,1,2,4]:
#        m = moments(rho,sigma,T)
#        print "T=%6g S=%6g delta-sigma-V = %g" % (T, sigma, m['S_rho,V'] - m['S_rho,Vb'])

#        for k in m:
#                print sigma,k,round(moments_theoretical_hopkins(rho,sigma,T)[k]-moments(rho,sigma,T)[k],5)

