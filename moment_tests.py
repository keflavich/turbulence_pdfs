import numpy as np
import hopkins_pdf
import pylab as pl

pl.rc('font',size=14)

Tvals = np.concatenate([[0],np.logspace(-3,0,5)])
Tvals = np.linspace(0,1.5,7)
sigmas = np.array([0.25,0.5,1.0,2.0,4.0])
meanrhos = np.logspace(0,5,6)

keys = hopkins_pdf.moments_theoretical_hopkins(0, 1, 1, meanrho=1).keys()

theoretical_moments = {r:{T:{s:hopkins_pdf.moments_theoretical_hopkins(0, s, T, meanrho=r) for s in sigmas} for T in Tvals} for r in meanrhos}

grid = {k:np.array([[[theoretical_moments[r][T][s][k] for T in Tvals] for s in sigmas] for r in meanrhos]) for k in keys}
# grid.shape = [r,s,T]

rho = np.logspace(-35,35,50000,base=np.e)

real_moments = {r:{T:{s:hopkins_pdf.moments(rho, s, T, meanrho=r) for s in sigmas} for T in Tvals} for r in meanrhos}

real_grid = {k:np.array([[[real_moments[r][T][s][k] for T in Tvals] for s in sigmas] for r in meanrhos]) for k in keys}


pretty_labels = {'<rho>_V': r'$<\rho>_V = \rho_0$',
            '<ln rho>_V': r'$<\ln \rho>_V$',
            'S_rho,V': r'$S_{\rho,V}$',
            'S_logrho,V': r'$S_{\ln \rho,V} = \sigma^2$',
            '<rho>_M': r'$<\rho>_M$',
            '<ln rho>_M': r'$<\ln \rho>_M$',
            'S_rho,M': r'$S_{\rho,M}$',
            'S_logrho,M': r'$S_{\ln \rho,M}$',
            }

for fignum in (0,1,2):
    pl.figure(fignum,figsize=(20,10))
    pl.clf()
    pl.suptitle("$\\rho_0=%0.1f$" % meanrhos[fignum])
    pl.subplots_adjust(left=0.05,right=0.97,wspace=0.33)
    for ii,k in enumerate(keys):
        pl.subplot(2,4,ii+1)
        for jj,s in enumerate(sigmas):
            if ('ln' in k or 'log' in k):
                L = pl.plot(Tvals, grid[k][fignum,jj,:], marker='x', label='$\\sigma=%0.1f' % s, alpha=0.5)
                color = L[0].get_color()
                pl.plot(Tvals, real_grid[k][fignum,jj,:], marker='+', linestyle='--', label='$\\sigma=%0.1f' % s, alpha=0.5, color=color)
            else:
                L = pl.semilogy(Tvals, grid[k][fignum,jj,:], marker='x', label='$\\sigma=%0.1f' % s, alpha=0.5)
                color = L[0].get_color()
                pl.semilogy(Tvals, real_grid[k][fignum,jj,:], marker='+', linestyle='--', label='$\\sigma=%0.1f' % s, alpha=0.5, color=color)
        #pl.imshow(grid[k][:,:,0])
        #pl.colorbar()
        pl.xlabel("$T$",fontsize=18)
        pl.ylabel(pretty_labels[k],fontsize=18)
    pl.suptitle("$\\rho_0=%0.1f$" % meanrhos[fignum])


# pl.figure(2)
# pl.clf()
# for ii,k in enumerate(keys):
#     pl.subplot(2,4,ii)
#     pl.imshow(grid[k][:,:,0])
#     pl.title(k)
#     pl.colorbar()

for mm,signum in enumerate([1,3]):
    pl.figure(fignum+mm+1,figsize=(20,10))
    pl.clf()
    pl.suptitle(r'$\sigma=%0.1f$' % sigmas[signum])
    pl.subplots_adjust(left=0.05,right=0.97,wspace=0.33)
    for ii,k in enumerate(keys):  #['<rho>_V','<rho>_M','<ln rho>_V','<ln rho>_M']):
        for jj,T in enumerate(Tvals):
            pl.subplot(2,4,ii+1)
            if 'ln' in k or 'log' in k:
                pl.semilogx(meanrhos, grid[k][:,signum,jj],      marker='x', linewidth=2, alpha=0.5)
                pl.semilogx(meanrhos, real_grid[k][:,signum,jj], marker='+', linewidth=2, alpha=0.5, linestyle='--')
            else:
                pl.loglog(meanrhos, (grid[k][:,signum,jj]),      linewidth=2, alpha=0.5, marker='x', linestyle='-')
                pl.loglog(meanrhos, (real_grid[k][:,signum,jj]), linewidth=2, alpha=0.5, marker='+', linestyle='--')
            if 'S' not in k and 'ln' not in k:
                pl.plot(pl.gca().get_xlim(),pl.gca().get_xlim(),'k:',alpha=0.5)
        pl.ylabel(pretty_labels[k], fontsize=18)
        pl.xlabel("$\\rho_0$", fontsize=18)

    pl.show()



#for T in [0,0.001,0.01,0.1,1]:
#    for sigma in [0.5,1,2,4]:
#        m = moments(rho,sigma,T)
#        print "T=%6g S=%6g delta-sigma-V = %g" % (T, sigma, m['S_rho,V'] - m['S_rho,Vb'])

#        for k in m:
#                print sigma,k,round(moments_theoretical_hopkins(rho,sigma,T)[k]-moments(rho,sigma,T)[k],5)

