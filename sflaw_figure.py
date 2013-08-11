import turbulent_pdfs
import pylab as pl
import numpy as np
from scipy.stats import norm

if __name__ == "__main__":
    rho = np.logspace(-8,8,10000)
    prho = [turbulent_pdfs.lognormal(rho, 1, sigma=s) for s in [0.5,1,2]]

    thresh = 1e2

    pl.rc('font',size=24)

    pl.clf()
    ax1 = pl.subplot(2,1,1)
    
    integrals = []
    for p in prho[::-1]:
        L, = pl.loglog(rho, p, linestyle='-', linewidth=3, alpha=0.75)
        OK = rho > thresh
        pl.fill_between(rho[OK], OK[OK]*1e-30, p[OK], color=L.get_color(), alpha=0.25)
        integrals.append(p[OK].sum() * np.log10(rho[1]/rho[0]))

    sigmas = [-norm.ppf(i) for i in integrals]

    ax1.set_ylim(1e-30,1)

    ax2 = pl.subplot(2,1,2)

    for S,s in zip(sigmas,(0.5,1,2)):
        p = turbulent_pdfs.lognormal(rho, 10**-(S-sigmas[0]), sigma=s)
        L, = pl.loglog(rho, p, linestyle='-', linewidth=3, alpha=0.75)
        OK = rho > thresh
        pl.fill_between(rho[OK], OK[OK]*1e-30, p[OK], color=L.get_color(), alpha=0.25)

    ax2.set_ylim(1e-30,1)

    ax2.set_xlabel(r'$\rho$')
    ax1.set_ylabel(r'$P(\rho)$')
    ax2.set_ylabel(r'$P(\rho)$')
    pl.subplots_adjust(hspace=0)
    ax1.set_xticks([])
    ax1.set_yticks(ax2.get_yticks()[2:-1])

    pl.show()

    pl.savefig('figures/threshold_star_formation_law_lognormal.png',bbox_inches='tight')
