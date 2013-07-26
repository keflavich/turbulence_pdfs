"""
Turbulent PDF Diagnostics
=========================

Some simple code to examine the properties of the integrals of "turbulent"
distributions
"""

from turbulent_pdfs import lognormal,lognormal_massweighted
import numpy as np
from scipy.optimize import curve_fit

def integrate_distr(dens,meandens,sigma,distr_func):
    """
    Integrate a distribution, discarding the last point so that the
    differential can be estimated
    """
    distr = distr_func(dens,meandens,sigma)
    dlogrho = np.diff(np.log(dens))
    return (distr[:-1]*dlogrho).sum()

rho = np.logspace(-50,50,50000)

sigmas = np.linspace(0.1,10,100)

lognormal_integrals = [integrate_distr(rho, 1, s, lognormal) for s in sigmas]
lognormalmass_integrals = [integrate_distr(rho, 1, s, lognormal_massweighted) for s in sigmas]

meanrhos = 10**np.arange(1,8)
lognormal_integrals_rho = [integrate_distr(rho, r, 1, lognormal) for r in meanrhos]
lognormalmass_integrals_rho = [integrate_distr(rho, r, 1, lognormal_massweighted) for r in meanrhos]

import pylab as pl
pl.figure(1)
pl.clf()
pl.plot(sigmas, lognormal_integrals)
pl.xlabel(r'$\sigma_s$')
pl.ylabel(r'$\int P_V(s)$')
pl.show()

pl.figure(2)
pl.clf()
pl.plot(sigmas, lognormalmass_integrals)
pl.xlabel(r'$\sigma_s$')
pl.ylabel(r'$\int P_M(s)$')

pl.figure(3)
pl.clf()
fits = []
for s in sigmas[::10]:
    distr = lognormal_massweighted(rho, 1, s)
    L, = pl.loglog(rho,distr, label=r'$\sigma_s=%f' % s, linewidth=3, alpha=0.5)
    (fitted_rho0,fitted_s),cov = curve_fit(lognormal, rho, distr, p0=[np.exp(s**2),s])
    fitted_distr = lognormal(rho, fitted_rho0, fitted_s)
    pl.loglog(rho,fitted_distr,color=L.get_color(), linestyle='--')
    fits.append((fitted_rho0,fitted_s))
fits = np.array(fits)

pl.xlabel(r'$\rho$')
pl.ylabel(r'$P_M(s)$')
pl.axis([1e-10,1e30,1e-10,1])

pl.figure(4)
pl.clf()
pl.plot(meanrhos, lognormal_integrals_rho)
pl.xlabel(r'$\rho_0$')
pl.ylabel(r'$\int P_V(s)$')
pl.show()

pl.figure(5)
pl.clf()
pl.plot(meanrhos, lognormalmass_integrals_rho)
pl.xlabel(r'$\rho_0$')
pl.ylabel(r'$\int P_M(s)$')

pl.show()
