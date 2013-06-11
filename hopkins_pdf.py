from scipy.special import iv
import scipy.optimize
import numpy as np


# Bessel function 1st-order
iv1 = lambda x: iv(1,x)

# for rescaling log_e -> log_10
ln10 = np.log(10)

def loghopkins(rho, sigma, T, meanrho=1, soften=False):
    r"""
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
    # du = - d(ln rho) / T
    log_rho[u>0] = (term1 + arg2 + 0.5*(np.log(lam)-np.log(u)) - np.log(T))[u>0]
    log_rho[u<=0] = -np.inf
    if soften:
        log_rho += (-lam)*np.exp(-(u**2/(2*(0.01*T)**2)))

    if np.any(np.isnan(log_rho)):
        raise ValueError("A value is NaN; this should not happen!")
    elif np.any(np.isinf(log_rho[log_rho>0])):
        raise ValueError("A positive infinite probability has occurred; this should not happen!")
    return log_rho


def hopkins(rho, sigma, T, meanrho=1.0, normalize=True):
    r"""
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
    normalize : bool
        If true, will normalize the returned array such that its sum is 1
        (rather than its integral)

    Returns
    -------
    The probability of the density, with integral = 1
    """
    P_V = np.exp(loghopkins(rho,sigma,T, meanrho=meanrho))
    if normalize:
        return P_V/P_V.sum()
    else:
        return P_V

def sigma_of_T(T, logform=False):
    """
    Use the relationship between sigma and T from Hopkins 2013 fig 3

    Parameters
    ----------
    T : float
        The T parameter of the distribution
    logform : bool
        If false, use
        T ~ 0.1 sigma_logM^2
        if True, use:
        T ~ 0.25 log(1+0.25 sigma_logM^4)

    Returns
    -------
    Sigma_logV, the input parameter for the distribution
    (sigma_logV^2 = (1+T)^3 sigma_logM^2
    """
    if logform:
        return ((np.exp(T*4)-1) * 4 )**0.25 * (1+T)**1.5
    else:
        return (10*T)**0.5 * (1+T)**1.5

def T_of_sigma(sigma, logform=False, maxT=8):
    """
    Inverse of sigma_of_T, again using the relation from Hopkins fig 3

    Uses scipy root finding because there are no analytic solutions for T of
    sigma_V

    Parameters
    ----------
    sigma : float
        sigma_logV, the defining width of the lognormal
    logform : bool
        If false, use
        T ~ 0.1 sigma_logM^2 = 0.1 sigma_logV^2 (1+T)**-3
        if True, use:
        T ~ 0.25 log(1+0.25 sigma_logM^4)
    maxT : float
        A parameter passed to the root finding algorithm.  For the linear
        equation, can't be more than about 10 or the root finder breaks.
        Hopkins only fit these equations for T<1 anyway.

    Returns
    -------
    T
    """
    if logform:
        def eq(T):
            return T - 0.25 * np.log(1+0.25*sigma**4 * (1+T)**-6)
    else:
        def eq(T):
            return T - 0.1 * sigma**2 * (1+T)**-3
    #print 'DEBUG: f(a),f(b):',eq(0),eq(maxT)
    return scipy.optimize.brentq(eq,0,maxT)

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

def hopkins_masspdf_ofmeandens(rho, vol_meanrho, sigma_volume, T, normalize=True):
    r"""
    Return the *mass-weighted* PDF

    .. math::

        P_M(\ln \rho) = \rho P_V(\ln \rho)

    Parameters
    ----------
    normalize : bool
        Re-normalize such that the *sum* of the probabilities = 1
    """
    # don't bother normalizing at this step; just adds cost.
    P_M = rho * hopkins(rho/vol_meanrho,sigma_volume,T,normalize=False)
    if normalize:
        return P_M/P_M.sum()
    else:
        return P_M


def moments(rho, sigma, T, meanrho=1):
    distribution = hopkins(rho=rho, sigma=sigma, T=T, meanrho=meanrho, normalize=False)

    # because the mass distribution is normalized, we have integral(f(x) dx) = 1 -> sum(f(x))=1
    #drho = np.concatenate([[0],np.diff(rho)])
    logrho = np.log(rho)
    dlogrho = np.diff(logrho)[0]

    E_rho = (rho*distribution*dlogrho).sum()
    E_logrho = (logrho*distribution*dlogrho).sum()
    E_rhosq = (rho**2 * distribution*dlogrho).sum()
    E_logrhosq = (logrho**2*distribution*dlogrho).sum()

    mass_distribution = rho/meanrho*distribution
    #print "DEBUG: distribution.sum(): ",(distribution*dlogrho).sum()," massdistribution.sum(): ",(mass_distribution*dlogrho).sum()
    E_M_logrho = (logrho*mass_distribution*dlogrho).sum()
    E_M_logrhosq = (logrho**2*mass_distribution*dlogrho).sum()
    E_M_rho = (rho*mass_distribution*dlogrho).sum()
    E_M_rhosq = (rho**2*mass_distribution*dlogrho).sum()

    if T != 0:  # add delta-function term
        lam = sigma**2/(2*T**2)
        E_rho += meanrho * np.exp(-lam/(1+T))
        E_rhosq += meanrho**2 * np.exp(lam*(T-1)/(T+1))
        E_M_rho += meanrho    * np.exp(lam*(T-1)/(T+1))
        E_M_rhosq += meanrho**2 * np.exp(lam*(2*T-1)/(T+1))
        E_logrho += np.exp(-lam) * (np.log(meanrho) + (lam*T)/(1+T))
        E_logrhosq += np.exp(-lam) * (np.log(meanrho) + (lam*T)/(1+T))**2
        #logmassexpectation += np.exp(-lam/(1+T)) * meanrho * (np.log(meanrho)+(lam*T/(1+T)))
        E_M_logrho += (lam*T/(1+T) + np.log(meanrho)) * np.exp(-lam/(T+1))
        E_M_logrhosq += (np.log(meanrho)+(lam*T/(1+T)))**2 * np.exp(-lam/(1+T))

    return {'<rho>_V': E_rho,
            '<ln rho>_V': E_logrho,
            #'S_rho,Vb': ( (rho-E_rho)**2 * distribution * dlogrho ).sum(), 
            'S_rho,V' : E_rhosq - E_rho**2,
            'S_logrho,V': E_logrhosq - E_logrho**2, #( (logrho-E_logrho)**2 * distribution * dlogrho ).sum(), 
            '<rho>_M': E_M_rho, 
            '<ln rho>_M': E_M_logrho, #(logrho*mass_distribution*dlogrho).sum(),
            'S_rho,M': E_M_rhosq - E_M_rho**2, #( rho**2 * mass_distribution*dlogrho ).sum() - massexpectation**2, 
            'S_logrho,M': E_M_logrhosq - E_M_logrho**2, #(logrho**2 * mass_distribution*dlogrho).sum() - logmassexpectation**2, 
            }

def moments_theoretical_lognormal(sigma,T,meanrho=1):
    return {'<rho>_V': meanrho,
            '<ln rho>_V': -sigma**2/2.,
            'S_rho,V': np.exp(sigma**2) - 1,
            'S_logrho,V': sigma**2,
            '<rho>_M': np.exp(sigma**2),
            '<ln rho>_M': sigma**2/2.,
            'S_rho,M': np.exp(3*sigma**2) - np.exp(2*sigma**2),
            'S_logrho,M': sigma**2,
            }

def moments_theoretical_hopkins(sigma,T,meanrho=1):
    E_rho = meanrho
    # have to simplify out lambda, since it can be zero
    # lam = sigma**2/(2*T**2)
    E_rhosq = meanrho**2 * np.exp(sigma**2/(1+3*T+2*T**2))
    E_M_rho = meanrho    * np.exp(sigma**2/(1+3*T+2*T**2))
    # E_logrho = log(rho_0) - lambda t**2 / (1+T)
    E_logrho = np.log(meanrho) - sigma**2/(2*(1+T))
    # computed in mathematica, including the Delta-function term
    # expanded as such to avoid 1/T -> 1/0 errors
    # There are some terms that are e^(-1/t^2) / t, which should be 0 for T=0
    E_logrhosq_terms = [
             sigma**2/(1 + T)**2,
             (2*T*sigma**2)/(1 + T)**2 ,
             (T**2*sigma**2)/(1 + T)**2 ,
             sigma**4/( 4*(1 + T)**2) ,
             -(sigma**2*np.log(meanrho))/(1 + T)**2 ,
             -(T*sigma**2*np.log(meanrho))/(1 + T)**2 ,
             np.log(meanrho)**2/(1 + T)**2 ,
             (2*T*np.log(meanrho)**2)/(1 + T)**2 ,
             (T**2*np.log(meanrho)**2)/(1 + T)**2 ,
        ]
    if T != 0:
        # these will raise errors if T == 0
        # They are 0 if T == 0
        E_logrhosq_terms += [
             (np.exp(-(sigma**2/(2*T**2)))*sigma**2*np.log(meanrho))/(T*(1 + T)),
             -(np.exp(-(sigma**2/(2*T**2))) * sigma**2*np.log(meanrho))/(T*(1 + T)**2) ,
             -(np.exp(-(sigma**2/(2*T**2))) * sigma**2*np.log(meanrho))/(1 + T)**2 ,
             -(np.exp(-(sigma**2/(2*T**2)))*np.log(meanrho)**2)/(1 + T)**2 ,
             -(2*np.exp(-(sigma**2/(2*T**2)))*T*np.log(meanrho)**2)/(1 + T)**2 ,
             -(np.exp(-(sigma**2/(2*T**2)))*T**2*np.log(meanrho)**2)/(1 + T)**2,
             np.exp(-(sigma**2/(2*T**2)))*np.log(meanrho)**2 ,
             ]
    E_logrhosq = np.sum(E_logrhosq_terms)
    E_M_logrho = np.log(meanrho) + sigma**2/2. / (1+T)**2

    return {'<rho>_V': E_rho,  # 1
            '<ln rho>_V': E_logrho, #-(sigma**2)/2. * (1/(1+T)),
            # E[rhosq] - E[rho]^2
            'S_rho,V': E_rhosq-E_rho**2, # np.exp(sigma**2/(1+3*T+2*T**2)) - 1,
            # E[logrhosq] - E[logrho]^2
            'S_logrho,V': E_logrhosq - E_logrho**2, # sigma**2,
            '<rho>_M': E_M_rho,  #np.exp(sigma**2/(1+3*T+2*T**2)),
            '<ln rho>_M': E_M_logrho,
            'S_rho,M': meanrho**2 *(np.exp(3*sigma**2/(1+4*T+3*T**2)) - np.exp(2*sigma**2/(1+3*T+2*T**2))),
            'S_logrho,M': sigma**2 * (1+T)**-3,
            }

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
    sigma = 1.0
    meanrhos = 10**(np.arange(2,8))
    rho = np.logspace(-36,31,50000,base=np.e)
    for meanrho,col,ls in zip(meanrhos,colors,linestyles):
        pdist = hopkins_masspdf_ofmeandens(rho, meanrho, sigma_volume=sigma, T=T)
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

    pl.figure(6)
    pl.clf()
    rho = np.exp(np.linspace(0.6,1.4,50000))
    sigma = 2
    Tvals = [0.9, 1.0, 1.1]

    for T in Tvals:
        pdist = loghopkins(rho,sigma,T,soften=False)
        pdistS = loghopkins(rho,sigma,T,soften=True)
        pdist[pdist<-100] = -100 # plot negative infinities
        #pdist += -1-pdist[np.isfinite(pdist)].max()
        pl.plot(np.log(rho), pdist/ln10, label="T=%0.2f" % T)
        pl.plot(np.log(rho), pdistS/ln10, label="T=%0.2f soft" % T)
    pl.axis([0.6,1.4,-1,0])
    pl.xlabel("$\\ln(\\rho)$",fontsize=18)
    pl.ylabel("$\\log_{10}(P[\\ln(\\rho)])$ Softened",fontsize=18)
    pl.legend(loc='best')
    pl.title("$\\sigma_{\ln(\\rho,V)}=2.0$",fontsize=18)

    if savefigures:
        pl.savefig("figures/hopkins_soften_test.png")

    sigma = 2
    rho = np.logspace(-50,35,50000,base=np.e)
    dlnrho = np.diff(np.log(rho)).mean()
    Tvals = np.linspace(0,3,100)
    integral = [(np.exp(loghopkins(rho,2,T,soften=False))*dlnrho).sum() for T in Tvals]
    soft_integral = [(np.exp(loghopkins(rho,2,T,soften=True))*dlnrho).sum() for T in Tvals]
    pl.rc('font',size=18)
    pl.figure(7)
    pl.clf()
    pl.plot(Tvals,integral,label="Integral")
    pl.plot(Tvals,soft_integral,label="Softened Integral")
    pl.plot(Tvals,integral+np.exp(-sigma**2/(2*Tvals**2)),label="Integral + $e^{-\\lambda}$")
    pl.legend(loc='best')
    pl.xlabel("T")
    pl.ylabel("$\\int_{-\infty}^{\infty} P_V d \\ln \\rho = 1$")

    if savefigures:
        pl.savefig("figures/hopkins_integrals.png")

    mass_integral = [(np.exp(loghopkins(rho,2,T,soften=False))*rho*dlnrho).sum() for T in Tvals]
    soft_mass_integral = [(np.exp(loghopkins(rho,2,T,soften=True))*rho*dlnrho).sum() for T in Tvals]

    pl.rc('font',size=18)
    pl.figure(8)
    pl.clf()
    pl.plot(Tvals,mass_integral,label="Integral")
    pl.plot(Tvals,soft_mass_integral,label="Softened Integral")
    # integral( rho exp(-lambda) delta(u) du ) 
    # du = -1/(rho * T) d rho
    # integral( rho exp(-lambda) delta(u) d rho/(rho* T)
    # = exp(-lambda)/T integral(delta(u) d rho)
    # = exp(-lambda)/T
    pl.plot(Tvals,mass_integral+np.exp(-sigma**2/(2*Tvals**2)*(1/(1+Tvals))),label="Integral + $e^{-\\lambda}$")
    pl.legend(loc='best')
    pl.xlabel("T")
    pl.ylabel("$\\int_{-\infty}^{\infty} \\rho P_V d \\ln \\rho = \\rho_0$")

    if savefigures:
        pl.savefig("figures/hopkins_mass_integrals.png")

    pl.show()

    return locals()
