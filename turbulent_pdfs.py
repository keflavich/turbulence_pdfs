from scipy.special import iv
import numpy as np

# Bessel function 1st-order
iv1 = lambda x: iv(1,x)

# for rescaling log_e -> log_10
ln10 = np.log(10)

def hightail_distr(dens, meandens,sigma,alpha=1,offset=1, rescale=True):
    pind = np.argmin(abs(dens-(meandens+offset/ln10)))
    distr = np.exp(-((dens-meandens)*ln10)**2/(2.*sigma**2))
    powertail = (((10**dens)**-alpha))*(dens>=dens[pind])
    powertail *= distr[pind]/powertail[pind]
    expbg     = np.exp(-((dens-dens[pind])*ln10)**2/(2*sigma)**2)*distr[pind]*(dens<dens[pind])
    distr += powertail+expbg
    if rescale:
        distr_mean = (dens*distr).sum()/distr.sum()
        delta = distr_mean-meandens
        return hightail_distr(meandens-delta,sigma,alpha=alpha,dens=dens,offset=offset,rescale=False)
    return distr/distr.sum()

def lowtail_distr(dens, meandens, sigma, alpha=1, offset=1, rescale=True):
    pind = np.argmin(abs(dens-(meandens-offset/ln10)))
    distr = np.exp(-((dens-meandens)*ln10)**2/(2.*sigma**2))
    powertail = ((10**(dens[pind]-dens))**-alpha)*(dens<=dens[pind])
    powertail *= distr[pind]/powertail[pind]
    expbg     = np.exp(-((dens-dens[pind])*ln10)**2/(2*sigma)**2)*distr[pind]
    #powertail[powertail!=powertail] = expbg[powertail!=powertail]
    powertail[pind:] = expbg[pind:]
    distr += powertail#+expbg
    if rescale:
        distr_mean = (dens*distr).sum()/distr.sum()
        delta = distr_mean-meandens
        return lowtail_distr(meandens-delta,sigma,alpha=alpha,dens=dens,offset=offset,rescale=False)
    return distr/distr.sum()

def compressive_distr(dens, meandens, sigma, offset=1.5, sigma2=None, secondscale=0.8, rescale=True):
    """ two lognormals stuck together 
    offset is in ln units (log-base-e)
    For mach3, secondscale = 0.8, offset = 1.5
    for Mach 10, see federrath_mach10_rescaled_massweighted_fitted:
    offset = 1.9
    secondscale = 1.2
    sigma2 = 0.61 sigma
    """
    if sigma2 is None: sigma2 = sigma
    distr = np.exp(-((dens-meandens)*ln10)**2/(2.*sigma**2)) + np.exp(-((dens-(meandens+offset/ln10))*ln10)**2/(2.*(sigma2)**2))*secondscale
    if rescale:
        distr_mean = (dens*distr).sum()/distr.sum()
        delta = distr_mean-meandens
        return compressive_distr(meandens-delta,sigma,offset=offset,sigma2=sigma2,dens=dens,secondscale=secondscale,rescale=False)
    return distr/distr.sum()

def lognormal(dens, meandens, sigma):
    """ Lognormal distribution

    Parameters
    ----------
    dens : float
        Density (presumably in units of cm^-3 or g cm^-3)
        *not* log density
    meandens : float
        Rho_0, the mean of the volume-weighted density
    sigma : float
        sqrt(S_V), the standard deviation of the volume-weighted density
    """
    S = sigma**2
    s = np.log(dens/meandens)
    distr = 1./(2*np.pi*S)**0.5 * np.exp(-((s+S/2.))**2/(2.*S))
    return distr

def lognormal_massweighted(dens, meandens, sigma):
    """ Mass-weighted """
    distr = lognormal(dens,meandens,sigma) * dens
    return distr

lognormal_massweighted.__doc__ += lognormal.__doc__
