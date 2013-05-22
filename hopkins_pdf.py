from scipy.special import iv
import numpy as np

# Bessel function 1st-order
iv1 = lambda x: iv(1,x)

def loghopkins(rho, sigma, T):
    """
    Hopkins 2013 probability distribution:
    :math:`P_V(\ln(\rho)) d \ln(\rho) = I_1 (2 \sqrt{\lambda u}) \exp\left[-(\lambda+u)] \sqrt{\frac{\lambda}{u}} \d u`

    The Bessel function is approximated as
    :math:`I_1(x) = (2 \pi x)^{-1/2} \exp(x)` 
    for x>500

    Parameters
    ----------
    rho : np.ndarray
        Floating array of densities (not log density)
    sigma : float, positive
        Sigma_ln(rho), the lognormal distribution width.  Hopkins 2013 uses the
        parameter S_{ln(rho)}, the variance, which is (sigma_ln(rho))^2
    T : float, positive
        The adjustable parameter of the Hopkins distribution.  If T=0, a
        lognormal distribution will be used in place of the Hopkins
        distribution (to avoid divide-by-zero errors)

    Returns
    -------
    The log-probability of the density  
    """
    if T == 0:
        return (-(np.log(rho))**2/(2.*sigma**2)) - np.log(np.sqrt(2*np.pi*sigma**2))

    log_rho = np.zeros(rho.shape,dtype=rho.dtype)

    lam = sigma**2 / (2*T**2)
    u = lam/(1+T) - np.log(rho)/T
    arg = (2*(lam*u)**0.5)
    arg2 = (-(lam+u))

    # log( (2 pi x)^(-1/2) * exp(x) ) = -1/2 * log(2*pi*x) + x

    # need to put arg<500 in the parens: iv1(0) = 0, but iv1(750) * 0 = nan
    term1 = np.empty(rho.shape,dtype=rho.dtype)
    term1[arg<500] = np.log(iv1(arg[arg < 500]))
    term1[arg>=500] =  (-0.5 * np.log(2*np.pi*arg[arg>=500]) + arg[arg>=500])

    # p(rho)=0 for u<0 
    log_rho[u>0] = (term1 + arg2 + 0.5*(np.log(lam)-np.log(u)))[u>=0]
    log_rho[u<0] = -np.inf

    if np.any(np.isnan(log_rho)):
        raise ValueError("A value is NaN; this should not happen!")
    elif np.any(np.isinf(log_rho[log_rho>0])):
        raise ValueError("A positive infinite probability has occurred; this should not happen!")
    return log_rho


def hopkins(rho, sigma, T):
    """
    Hopkins 2013 probability distribution:
    :math:`P_V(\ln(\rho)) d \ln(\rho) = I_1 (2 \sqrt{\lambda u}) \exp\left[-(\lambda+u)] \sqrt{\frac{\lambda}{u}} \d u`

    The Bessel function is approximated as
    :math:`I_1(x) = (2 \pi x)^{-1/2} \exp(x)` 
    for x>500

    Parameters
    ----------
    rho : np.ndarray
        Floating array of densities (not log density)
    sigma : float, positive
        Sigma_ln(rho), the lognormal distribution width.  Hopkins 2013 uses the
        parameter S_{ln(rho)}, the variance, which is (sigma_ln(rho))^2
    T : float, positive
        The adjustable parameter of the Hopkins distribution.  If T=0, a
        lognormal distribution will be used in place of the Hopkins
        distribution (to avoid divide-by-zero errors)

    Returns
    -------
    The probability of the density, with integral = 1
    """
    return np.exp(loghopkins(rho,sigma,T))

def test_hopkins(savefigures=False):
    import pylab as pl
    pl.figure(1)
    pl.clf()
    rho = np.logspace(-15,10,10000,base=np.e)
    sigma = 2
    Tvals = [0,0.05,0.12,0.3,0.6]
    colors = ['k',(0.2,0.0,0.8),(0.4,0.0,0.6),(0.1,0.1,0.9),(0,0.7,0.7)]
    linestyles = ['-',':','--','-.','-']

    for T,col,ls in zip(Tvals,colors,linestyles):
        pdist = loghopkins(rho,sigma,T)
        #pdist += -1-pdist[np.isfinite(pdist)].max()
        pl.plot(np.log(rho), pdist, label="T=%0.2f" % T, color=col, linestyle=ls)
    pl.axis([-16,11,-10,0])
    pl.xlabel("$\\ln(\\rho)$",fontsize=18)
    pl.ylabel("$\\log(P[\\ln(\\rho)])$",fontsize=18)
    pl.legend(loc='best')
    pl.title("$\\sigma_{\ln(\\rho,V)}=2.0$",fontsize=18)

    if savefigures:
        pl.savefig("Hopkins2013_fig1a.png")

    pl.figure(2)
    pl.clf()
    sigmas = np.array([0.1,0.2,0.5,1,2,4,8])**0.5
    T = 0.12
    colors = ['k',(0.2,0.0,0.8),(0.4,0.0,0.6),(0.1,0.1,0.9),(0,0.7,0.7),(0,1,0),(1,0.7,0)]
    linestyles = ['-',':','--','-.','-','--','-']

    for sigma,col,ls in zip(sigmas,colors,linestyles):
        pdist = loghopkins(rho,sigma,T)
        #pdist += -1-pdist[np.isfinite(pdist)].max()
        pl.plot(np.log(rho), pdist, label="$\\sigma=%0.2f$" % sigma, color=col, linestyle=ls)
    pl.axis([-16,11,-10,0])
    pl.xlabel("$\\ln(\\rho)$",fontsize=18)
    pl.ylabel("$\\log(P[\\ln(\\rho)])$",fontsize=18)
    pl.legend(loc='best')
    pl.title("T=0.12",fontsize=18)

    if savefigures:
        pl.savefig("Hopkins2013_fig1b.png")

    pl.show()
