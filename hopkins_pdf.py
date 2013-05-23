from scipy.special import iv
import numpy as np

# Bessel function 1st-order
iv1 = lambda x: iv(1,x)

# for rescaling log_e -> log_10
ln10 = np.log(10)

def loghopkins(rho, sigma, T, meanrho=1):
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
        return (-(np.log(rho)-np.log(meanrho)+sigma**2/2.)**2/(2.*sigma**2)) - 0.5 * np.log(2*np.pi*sigma**2)

    log_rho = np.zeros(rho.shape,dtype=rho.dtype)

    lam = sigma**2 / (2*T**2)
    u = lam/(1+T) - np.log(rho/meanrho)/T
    arg = (2*(lam*u)**0.5)
    arg2 = (-(lam+u))

    # log( (2 pi x)^(-1/2) * exp(x) ) = -1/2 * log(2*pi*x) + x

    # need to put arg<500 in the parens: iv1(0) = 0, but iv1(750) * 0 = nan
    term1 = np.empty(rho.shape,dtype=rho.dtype)
    term1[arg<500] = np.log(iv1(arg[arg < 500]))
    term1[arg>=500] =  (-0.5 * np.log(2*np.pi*arg[arg>=500]) + arg[arg>=500])

    # p(rho)=0 for u<0 
    # du = d(ln rho) / T
    log_rho[u>0] = (term1 + arg2 + 0.5*(np.log(lam)-np.log(u)) - np.log(T))[u>0]
    log_rho[u<=0] = -np.inf

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

def hopkins_masspdf(rho, sigma_M, T):
    """
    Mass-weighted version of the Hopkins PDF
    """
    sigma = sigma_M * (1+T)**3
    return np.exp(loghopkins(rho,sigma,T))

def loghopkins_masspdf(rho, sigma_M, T):
    """
    Mass-weighted version of the Hopkins PDF
    """
    sigma = sigma_M * (1+T)**3
    return (loghopkins(rho,sigma,T))

def hopkins_masspdf_ofmeandens(rho, meanrho, T):
    sigma_M = np.sqrt( np.log(meanrho) * (1+3*T+2*T**2)/(1+T)**3)
    sigma = sigma_M * (1+T)**3
    return np.exp(loghopkins(rho,sigma,T))


def test_hopkins(savefigures=False):
    import pylab as pl
    pl.figure(1)
    pl.clf()
    rho = np.logspace(-16,11,50000,base=np.e)
    sigma = 2
    Tvals = [0,0.05,0.12,0.3,0.6,0.9,1.0]
    colors = ['k',(0.2,0.0,0.8),(0.4,0.0,0.6),(0.1,0.1,0.9),(0,0.7,0.7),(1,0,0),(1,0,1)]
    linestyles = ['-',':','--','-.','-','-','--']

    for T,col,ls in zip(Tvals,colors,linestyles):
        pdist = loghopkins(rho,sigma,T)
        pdist[pdist<-100] = -100 # plot negative infinities
        #pdist += -1-pdist[np.isfinite(pdist)].max()
        pl.plot(np.log(rho), pdist/ln10, label="T=%0.2f" % T, color=col, linestyle=ls)
    pl.axis([-16,11,-10,0])
    pl.xlabel("$\\ln(\\rho)$",fontsize=18)
    pl.ylabel("$\\log_{10}(P[\\ln(\\rho)])$",fontsize=18)
    pl.legend(loc='best')
    pl.title("$\\sigma_{\ln(\\rho,V)}=2.0$",fontsize=18)

    if savefigures:
        pl.savefig("figures/Hopkins2013_fig1a.png")

    pl.figure(2)
    pl.clf()
    sigmas = np.array([0.1,0.2,0.5,1,2,4,8])**0.5
    T = 0.12
    colors = ['k',(0.2,0.0,0.8),(0.4,0.0,0.6),(0.1,0.1,0.9),(0,0.7,0.7),(0,1,0),(1,0.7,0)]
    linestyles = ['-',':','--','-.','-','--','-']

    for sigma,col,ls in zip(sigmas,colors,linestyles):
        pdist = loghopkins(rho,sigma,T)
        pdist[pdist<-100] = -100 # plot negative infinities
        #pdist += -1-pdist[np.isfinite(pdist)].max()
        pl.plot(np.log(rho), pdist/ln10, label="$\\sigma=%0.2f$" % sigma, color=col, linestyle=ls)
    pl.axis([-16,11,-10,1])
    pl.xlabel("$\\ln(\\rho)$",fontsize=18)
    pl.ylabel("$\\log_{10}(P[\\ln(\\rho)])$",fontsize=18)
    pl.legend(loc='best')
    pl.title("T=0.12",fontsize=18)

    if savefigures:
        pl.savefig("figures/Hopkins2013_fig1b.png")

    pl.figure(3)
    pl.clf()
    rho = np.logspace(-16,11,50000,base=np.e)
    sigma = 2
    Tvals = [0,0.05,0.12,0.3,0.6,0.9,1.0]
    colors = ['k',(0.2,0.0,0.8),(0.4,0.0,0.6),(0.1,0.1,0.9),(0,0.7,0.7),(1,0,0),(1,0,1)]
    linestyles = ['-',':','--','-.','-','-','--']

    for T,col,ls in zip(Tvals,colors,linestyles):
        pdist = loghopkins_masspdf(rho,sigma,T)
        pdist[pdist<-100] = -100 # plot negative infinities
        #pdist += -1-pdist[np.isfinite(pdist)].max()
        pl.plot(np.log(rho), pdist/ln10, label="T=%0.2f" % T, color=col, linestyle=ls)
    pl.axis([-16,11,-10,0])
    pl.xlabel("$\\ln(\\rho_M)$",fontsize=18)
    pl.ylabel("$\\log_{10}(P[\\ln(\\rho_M)])$",fontsize=18)
    pl.legend(loc='best')
    pl.title("$\\sigma_{\ln(\\rho,M)}=2.0$",fontsize=18)

    if savefigures:
        pl.savefig("figures/Hopkins2013_fig1a_massweight.png")

    pl.figure(4)
    pl.clf()
    sigmas = np.array([0.1,0.2,0.5,1,2,4,8])**0.5
    T = 0.12
    colors = ['k',(0.2,0.0,0.8),(0.4,0.0,0.6),(0.1,0.1,0.9),(0,0.7,0.7),(0,1,0),(1,0.7,0)]
    linestyles = ['-',':','--','-.','-','--','-']

    for sigma,col,ls in zip(sigmas,colors,linestyles):
        pdist = loghopkins_masspdf(rho,sigma,T)
        pdist[pdist<-100] = -100 # plot negative infinities
        #pdist += -1-pdist[np.isfinite(pdist)].max()
        pl.plot(np.log(rho), pdist/ln10, label="$\\sigma_M=%0.2f$" % sigma, color=col, linestyle=ls)
    pl.axis([-16,11,-10,1])
    pl.xlabel("$\\ln(\\rho_M)$",fontsize=18)
    pl.ylabel("$\\log_{10}(P[\\ln(\\rho_M)])$",fontsize=18)
    pl.legend(loc='best')
    pl.title("T=0.12",fontsize=18)

    if savefigures:
        pl.savefig("figures/Hopkins2013_fig1b_massweight.png")


    pl.figure(5)
    pl.clf()
    T = 0.12
    meanrhos = 10**(np.arange(2,8))
    rho = np.logspace(-36,31,50000,base=np.e)
    for meanrho,col,ls in zip(meanrhos,colors,linestyles):
        pdist = hopkins_masspdf_ofmeandens(rho, meanrho, T)
        #pdist[pdist<-100] = -100 # plot negative infinities
        #pdist += -1-pdist[np.isfinite(pdist)].max()
        pl.plot(np.log(rho), np.log10(pdist), label="$<\\rho>_M=%0.2g$" % meanrho, color=col, linestyle=ls)
    pl.axis([-36,31,-10,0])
    pl.xlabel("$\\ln(\\rho_M)$",fontsize=18)
    pl.ylabel("$\\log_{10}(P[\\ln(\\rho_M)])$",fontsize=18)
    pl.legend(loc='best')
    pl.title("$T=0.12$ varying mean density",fontsize=18)

    if savefigures:
        pl.savefig("figures/Hopkins_MassPDFVsMeanMass.png")

    pl.show()


