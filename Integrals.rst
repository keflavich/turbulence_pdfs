Hopkins PDF Generalization
##########################
:date: 2013-06-04 15:00
:author: Adam (keflavich@gmail.com)
:tags: turbulence

`Hopkins 2013`_ presents a non-lognormal form of the probability distribution
function of density in turbulence, depending primarily on a parameter
:math:`T`.  I examine some of its properties here.

The conservation equations, conservation of probability and mass

.. math:: \int_{-\infty}^\infty P_V(\ln \rho) d \ln \rho \equiv 1

.. math:: \int_{-\infty}^\infty\rho P_V(\ln \rho) d \ln \rho \equiv \rho_0 = 1

The :math:`=1` is the definition used by `Hopkins 2013`_.  Part of the goal of this document
is to generalize to :math:`\rho_0 <> 1`

The most important thing to do is determine the values of the moments of the distribution, including
the integral corrections for the Dirac Delta function component of the PDF.

The PDF
-------
The full form of the Hopkins PDF is 

.. math:: P_V(\ln \rho/\rho_0) d \ln \rho =  \left[I_1(2\sqrt{\lambda u}) e^{-\lambda-u} \sqrt{\frac{\lambda}{u}} + e^{-\lambda} \delta(u)\right]du

The mass-weighted PDF is 

.. math:: P_M(\ln \rho/\rho_0) d \ln \rho =  \frac{\rho}{\rho_0} \left[I_1(2\sqrt{\lambda u}) e^{-\lambda-u} \sqrt{\frac{\lambda}{u}} + e^{-\lambda} \delta(u)\right]du

With definitions:

.. math:: u\equiv \frac{\lambda}{1+T} - \frac{\ln \rho/\rho_0}{T}  ;  (u \geq 0)
.. math:: \lambda \equiv \frac{S_{\ln \rho,V}}{2 T^2}

The ratio :math:`\rho/\rho_0` is needed to ensure self-consistency: the mass
PDF must also conserve probability, which means it must be scaled by
:math:`1/\rho_0`.

.. But note that both of these distributions can depend on :math:`\rho_0`, changing :math:`u` to be
.. 
.. .. math:: u\equiv \frac{\lambda}{1+T} - \frac{\ln (\rho/\rho_0)}{T}  ;  (u \geq 0)


Moment equations
~~~~~~~~~~~~~~~~

The important moments are the mean and variance of the volume and mass density
probability distribution and the volume and mass *log*-density probability
distribution.  

The definitions used here are :math:`E[\rho] \equiv <\rho>_V`, etc.  The
variance is computed as :math:`V[x]=E[x^2]-E[x]^2`, which means we'll need to
compute the moments of various combinations of variables.

Also, :math:`S_{\ln \rho,V} \equiv V[\ln \rho_V]`

.. math:: <\rho>_V \equiv \rho_0 \equiv \int_{-\infty}^{\infty} \rho P_V(\ln \rho/\rho_0) d \ln \rho

.. math:: <\ln \rho>_V \equiv \int_{-\infty}^{\infty} \ln \rho P_V(\ln \rho/\rho_0) d \ln \rho

.. math:: <\rho>_M \equiv \int_{-\infty}^{\infty} \rho (\rho P_V(\ln \rho/\rho_0)) d \ln \rho

.. math:: <\ln \rho>_M \equiv \int_{-\infty}^{\infty} \ln \rho (\rho P_V(\ln \rho/\rho_0)) d \ln \rho

.. math:: <\rho^3>_V \equiv \int_{-\infty}^{\infty} \rho^3 P_V(\ln \rho/\rho_0) d \ln \rho

.. math:: <\ln^2 \rho>_V \equiv \int_{-\infty}^{\infty} (\ln \rho)^2 P_V(\ln \rho/\rho_0)d \ln \rho

.. math:: <\ln^2 \rho>_M \equiv \int_{-\infty}^{\infty} (\ln \rho)^2 \rho P_V(\ln \rho/\rho_0)d \ln \rho

In the moment equations, only the input to the PDF is scaled by a mean density
:math:`\rho_0`; the "weighting factors" for the expectation value are not
(i.e., we are measuring :math:`E[\rho]`, not :math:`E[\rho/rho_0]`).

Delta terms
~~~~~~~~~~~
These can be integrated analytically.  In this section, I am just computing the
means; the variances are more complicated.

.. math:: <\rho>_V \equiv \rho_0 \equiv \int_{-\infty}^{\infty} \rho e^{-\lambda} \delta(\frac{\lambda}{1+T} - \frac{\ln\rho/\rho_0}{T}) \frac{d \ln \rho}{T}

.. math:: <\ln \rho>_V \equiv \int_{-\infty}^{\infty} \ln \rho e^{-\lambda} \delta(\frac{\lambda}{1+T} - \frac{\ln\rho/\rho_0}{T}) \frac{d \ln \rho}{T}

.. math:: <\rho>_M \equiv \int_{-\infty}^{\infty} \rho (\rho e^{-\lambda} \delta(\frac{\lambda}{1+T} - \frac{\ln\rho/\rho_0}{T})) \frac{d \ln \rho}{T}

.. math:: <\ln \rho>_M \equiv \int_{-\infty}^{\infty} \ln \rho (\rho e^{-\lambda} \delta(\frac{\lambda}{1+T} - \frac{\ln\rho/\rho_0}{T})) \frac{d \ln \rho}{T}

Substitution: :math:`v=\frac{\ln \rho/\rho_0}{T}`,
:math:`dv = \frac{1}{T} d \ln \rho`, :math:`\rho=\rho_0 e^{v*T}`, :math:`\ln \rho = v T + \ln \rho_0`

.. math:: <\rho>_{V\delta} \equiv \rho_0 \equiv \int_{-\infty}^{\infty} \rho_0 e^{vT} e^{-\lambda} \delta(\frac{\lambda}{1+T} - v) d v

.. math:: <\ln \rho>_{V\delta} \equiv \int_{-\infty}^{\infty} (vT + \ln \rho_0) e^{-\lambda} \delta(\frac{\lambda}{1+T} - v) d v

.. math:: <\rho>_{M\delta} \equiv \int_{-\infty}^{\infty} \rho_0^2 e^{2vT} ( e^{-\lambda} \delta(\frac{\lambda}{1+T} - v)) d v

.. math:: <\ln \rho>_{M\delta} \equiv \int_{-\infty}^{\infty} (vT + \ln \rho_0) \rho_0 e^{vT} ( e^{-\lambda} \delta(\frac{\lambda}{1+T} - v)) d v



Solutions:

.. math:: <\rho>_{V\delta} =  \rho_0 \exp\left[\frac{T \lambda }{1+T} - \lambda\right] =  \rho_0 \exp\left[-\lambda \frac{1}{1+T}\right]

.. math:: <\ln \rho>_{V\delta} =  e^{-\lambda} \frac{\lambda T}{1+T} + e^{-\lambda} \ln \rho_0

.. math:: <\rho>_{M\delta} =  \rho_0^2 \exp\left[\frac{2 T \lambda }{1+T} - \lambda\right] = \rho_0^2 \exp\left[\lambda\frac{T-1}{T+1}\right]

.. math:: <\ln \rho>_{M\delta} = \rho_0  \left( \frac{\lambda T}{1+T} + \ln \rho_0 \right) \exp\left[\frac{T \lambda }{1+T} - \lambda\right]
.. math::                      = \rho_0 \left( \frac{\lambda T}{1+T} + \ln \rho_0 \right) \exp\left[\frac{ -\lambda }{T+1}\right] 

(for these next 3, I skipped intermediate steps)

.. math:: <\rho^3>_{V\delta} =  \rho_0^3 \exp\left[\frac{3 T \lambda }{1+T} - \lambda\right] = \rho_0^3 \exp\left[\lambda\frac{2T-1}{T+1}\right]

.. math:: <\ln^2 \rho>_{M\delta} = \rho_0 \left( \frac{\lambda T}{1+T} + \ln \rho_0 \right)^2 \exp\left[\frac{ -\lambda }{T+1}\right] 

.. math:: <\ln^2 \rho>_{V\delta} = \left( \frac{\lambda T}{1+T} + \ln \rho_0 \right)^2 e^{-\lambda}

Using :math:`\rho_0=1` as defined in `Hopkins 2013`_ simplifies all of these a great deal.


PDF Integrals
~~~~~~~~~~~~~
These cannot be integrated analytically.

However, we can work from a few simple mathematica/sympy results:


.. math:: \int_0^\infty I_1(x) e^{-x^2/(4L)} dx = e^L - 1

.. math:: \int_0^\infty x^2 I_1(x) e^{-x^2/(4L)} dx = 4 L^2 * e^L

.. math:: \int_0^\infty x^4 I_1(x) e^{-x^2/(4L)} dx = 16 L^3 (L+2) * e^L

We use :math:`L` instead of :math:`\lambda` in these equations because it is often substituted in later equations.

Expectation Value of the Volume-Weighted Density :math:`E[\rho]`
````````````````````````````````````````````````````````````````

.. math:: E[\rho] \equiv \int \rho P_v(\ln \rho/\rho_0) d \ln \rho = \int \rho \left[I_1(2\sqrt{\lambda u}) e^{-\lambda-u} \sqrt{\frac{\lambda}{u}} + e^{-\lambda} \delta(u)\right]du

To get to the form of the above equations, we use the substitution

.. math:: x = 2\sqrt{\lambda u}

which gives us :math:`\rho` in terms of :math:`x`:

.. math:: \rho = \rho_0 \exp\left[T\left(-\frac{x^2}{4\lambda} + \frac{\lambda}{1+T}\right)\right]

and leads to the rearrangement:

.. math:: E[\rho] = \int \rho_0 \exp\left[T\left(-\frac{x^2}{4\lambda} + \frac{\lambda}{1+T}\right)\right] \left[I_1(x) e^{-x^2/(4\lambda)-\lambda} \right]dx + \rho_0 \exp\left(- \frac{\lambda}{1+T}\right)

where the rightmost term is kept from the first moment above.  The integral
term can straightforwardly be broken apart into equations of the form shown
above.

.. math:: L \rightarrow \frac{\lambda}{1+T}

.. math:: E[\rho] = \rho_0 \left[ \exp \left(-\lambda+\frac{T\lambda}{1+T}\right) \int  \left[I_1(x) e^{-x^2/(4L)} \right]dx +\exp\left(- \frac{\lambda}{1+T}\right) \right]
.. math::         = \rho_0 \left[ \exp \left(-\lambda+\frac{T\lambda}{1+T}\right)(e^L-1)  +\exp\left(- \frac{\lambda}{1+T}\right) \right]
.. math::         = \rho_0 \left[ \exp \left(-\lambda+\frac{T\lambda}{1+T}\right)(e^{\lambda/1+T}-1)  +\exp\left(- \frac{\lambda}{1+T}\right) \right]
.. math::         = \rho_0 \left[ e^{-\lambda/(1+T)}(e^{\lambda/1+T}-1)  +\exp\left(- \frac{\lambda}{1+T}\right) \right]
.. math::         = \rho_0


The same general approach can be followed for all expectation values, but we'll skip the detailed algebra.

Expectation Value of the Mass-Weighted Density :math:`E_M[\rho]`
````````````````````````````````````````````````````````````````
.. math:: E[\rho^2] = \rho_0 \left[ \exp\left(\lambda\frac{2 T^2}{1+3T+2T^2}\right) - \exp\left(\lambda\frac{T-1}{T+1}\right) + \exp\left(\lambda\frac{T-1}{T+1}\right) \right]

The right 2 terms cancel, yielding the value shown in Equation 7 of Hopkins
2013 scaled by :math:`\rho_0^2`.  However, the right-most term is the
correction factor from the Dirac Delta term needed to correct any
numerical computation of the mass-weighted density.

Variance of the Volume-Weighted Density :math:`V[\rho]=S_{\ln \rho,V}`
``````````````````````````````````````````````````````````````````````

.. math:: V[\rho] = E[\rho^2] - E[\rho]^2 = \rho_0^2 \left[  \exp\left(\lambda\frac{2 T^2}{1+3T+2T^2}\right) - 1 \right]

However, the "correction factor" is still important:

.. math:: V_\delta[\rho] = \rho_0^2 \left[ \exp\left(\lambda\frac{T-1}{T+1}\right) - \exp\left(-2\frac{\lambda}{1+T}\right) \right]


Expectation Value of the Volume-Weighted Log Density :math:`E[\rho]`
````````````````````````````````````````````````````````````````



.. _Hopkins 2013: http://adsabs.harvard.edu/abs/2013MNRAS.430.1880H
