Hopkins PDF Generalization
##########################
:date: 2013-06-04 15:00
:author: Adam (keflavich@gmail.com)
:tags: turbulence

`Hopkins 2013`_ presents a non-lognormal form of the probability distribution
function of density in turbulence, depending primarily on a parameter
:math:`T`.  I examine some of its properties here.  All of the equations shown
on this page are implemented in https://github.com/keflavich/turbulence_pdfs,
in particular the functions ``moments_theoretical_hopkins`` and ``moments`` in
`hopkins_pdf.py
<https://github.com/keflavich/turbulence_pdfs/blob/master/hopkins_pdf.py>`__.

The conservation equations, conservation of probability and mass

.. math:: \int_{-\infty}^\infty P_V(\ln \rho) d \ln \rho \equiv 1

.. math:: \int_{-\infty}^\infty\rho P_V(\ln \rho) d \ln \rho \equiv \rho_0 = 1

:math:`\rho_0=1` is the definition used by `Hopkins 2013`_.  Part of the goal
of this document is to generalize to :math:`\rho_0 <> 1`

The most important thing to do is determine the values of the moments of the
distribution, including the integral corrections for the Dirac Delta function
component of the PDF.

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

These distributions are slightly different than those shown in `Hopkins 2013`_.
The :math:`m=0` term in the summation version of the PDF yields a Dirac
:math:`\delta` function in the integral form of the PDF.  To quote Phil::

    If you check, the summation expression is exact, and does exactly integrate
    to unity (and properly conserve mass) for any value of T or lambda (with
    u>0). But this is with the implicit understanding that the gamma-function
    distribution with exponent "m" should be taken to go to a delta-function
    about u=0 when m=0 (since otherwise it is ill defined; and physically this
    is the "no event/multiplication" case, so that's the only allowed case for
    the term in u). 

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

.. math:: <\rho>_M \equiv \int_{-\infty}^{\infty} \rho (\frac{\rho}{\rho_0} e^{-\lambda} \delta(\frac{\lambda}{1+T} - \frac{\ln\rho/\rho_0}{T})) \frac{d \ln \rho}{T}

.. math:: <\ln \rho>_M \equiv \int_{-\infty}^{\infty} \ln \rho (\frac{\rho}{\rho_0} e^{-\lambda} \delta(\frac{\lambda}{1+T} - \frac{\ln\rho/\rho_0}{T})) \frac{d \ln \rho}{T}

Substitution: :math:`v=\frac{\ln \rho/\rho_0}{T}`,
:math:`dv = \frac{1}{T} d \ln \rho`, :math:`\rho=\rho_0 e^{v*T}`, :math:`\ln \rho = v T + \ln \rho_0`

.. math:: <\rho>_{V\delta} \equiv \rho_0 \equiv \int_{-\infty}^{\infty} \rho_0 e^{vT} e^{-\lambda} \delta(\frac{\lambda}{1+T} - v) d v

.. math:: <\ln \rho>_{V\delta} \equiv \int_{-\infty}^{\infty} (vT + \ln \rho_0) e^{-\lambda} \delta(\frac{\lambda}{1+T} - v) d v

.. math:: <\rho>_{M\delta} \equiv \int_{-\infty}^{\infty} \rho_0 e^{2vT} ( e^{-\lambda} \delta(\frac{\lambda}{1+T} - v)) d v

.. math:: <\ln \rho>_{M\delta} \equiv \int_{-\infty}^{\infty} (vT + \ln \rho_0) e^{vT} ( e^{-\lambda} \delta(\frac{\lambda}{1+T} - v)) d v



Solutions:

.. math:: <\rho>_{V\delta} =  \rho_0 \exp\left[\frac{T \lambda }{1+T} - \lambda\right] =  \rho_0 \exp\left[-\lambda \frac{1}{1+T}\right]

.. math:: <\ln \rho>_{V\delta} =  e^{-\lambda} \frac{\lambda T}{1+T} + e^{-\lambda} \ln \rho_0

.. math:: <\rho>_{M\delta} =  \rho_0 \exp\left[\frac{2 T \lambda }{1+T} - \lambda\right] = \rho_0 \exp\left[\lambda\frac{T-1}{T+1}\right]

.. math:: <\ln \rho>_{M\delta} = \left( \frac{\lambda T}{1+T} + \ln \rho_0 \right) \exp\left[\frac{T \lambda }{1+T} - \lambda\right]
.. math::                      = \left( \frac{\lambda T}{1+T} + \ln \rho_0 \right) \exp\left[\frac{ -\lambda }{T+1}\right] 

(for these next 3, I skipped intermediate steps)

.. math:: <\rho^3>_{V\delta} =  \rho_0^2 \exp\left[\frac{3 T \lambda }{1+T} - \lambda\right] = \rho_0^2 \exp\left[\lambda\frac{2T-1}{T+1}\right]

.. math:: <\ln^2 \rho>_{M\delta} = \left( \frac{\lambda T}{1+T} + \ln \rho_0 \right)^2 \exp\left[\frac{ -\lambda }{T+1}\right] 

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


Variance of the Volume-Weighted Density :math:`V[\rho]=S_{\ln \rho,V}`
``````````````````````````````````````````````````````````````````````

.. math:: V[\rho] = E[\rho^2] - E[\rho]^2 = \rho_0^2 \left[  \exp\left(\lambda\frac{2 T^2}{1+3T+2T^2}\right) - 1 \right]

However, the "correction factor" is still important:

.. math:: V_\delta[\rho] = \rho_0^2 \left[ \exp\left(\lambda\frac{T-1}{T+1}\right) - \exp\left(-2\frac{\lambda}{1+T}\right) \right]

Expectation Value of the Mass-Weighted Density :math:`E_M[\rho]`
````````````````````````````````````````````````````````````````
Start from halfway through :math:`E[\rho]`, simply adding a factor of 2 in the exponent:

.. math:: E_M[\rho] = \int \rho_0 \exp\left[2T\left(-\frac{x^2}{4\lambda} + \frac{\lambda}{1+T}\right)\right] \left[I_1(x) e^{-x^2/(4\lambda)-\lambda} \right]dx + \rho_0 \exp\left(- \frac{\lambda(T-1)}{1+T}\right)

Follow the same math, with :math:`L=\frac{\lambda}{1+2T}`

.. math::         = \rho_0 \left[ \exp \left(-\lambda+\frac{2T\lambda}{1+T}\right)(e^L-1)  +\exp\left(- \frac{\lambda(T-1)}{1+T}\right) \right]
.. math::         = \rho_0 \left[ \exp \left(\frac{(T-1)\lambda}{1+T}\right)(e^{\lambda/(1+2T)}-1)  +\exp\left(- \frac{\lambda}{1+T}\right) \right]

.. math:: E_M[\rho] = \rho_0 \left[ \exp\left(\lambda\frac{2 T^2}{1+3T+2T^2}\right) - \exp\left(\lambda\frac{T-1}{T+1}\right) + \exp\left(\lambda\frac{T-1}{T+1}\right) \right]

The right 2 terms cancel, yielding the value shown in Equation 7 of `Hopkins 2013`_ 
scaled by :math:`\rho_0^2`.  However, the right-most term is the
correction factor from the Dirac Delta term needed to correct any
numerical computation of the mass-weighted density.

.. math:: E_{\delta,M}[\rho] = \exp\left(\lambda\frac{T-1}{T+1}\right)

.. math:: E_M[\rho] = \rho_0  \exp\left(\lambda\frac{2 T^2}{1+3T+2T^2}\right) 

Expectation Value of the Mass-Weighted Density Squared :math:`E_M[\rho^2]`
``````````````````````````````````````````````````````````````````````````
.. math:: E_M[\rho^2] = \int \rho^2 \frac{\rho}{\rho_0} e^{-\lambda} \left[I_1(x) e^{-x^2/(4\lambda)} \right]dx + \int \rho^2 \frac{\rho}{\rho_0} e^{-\lambda} \delta(u) du
.. math:: E_{\delta,M}[\rho^2] = \rho_0^2 \exp\left[\lambda\frac{2T-1}{T+1}\right]
.. math:: E_{left}[\rho^2] = e^{-\lambda} \int 
        \rho_0^2 \exp\left[3T\left(-\frac{x^2}{4\lambda} + \frac{\lambda}{1+T}\right)\right]
        \left[I_1(x) e^{-x^2/(4\lambda)} \right]
        dx
.. math::
         = \rho_0^2 \exp\left[\frac{(2T-1)\lambda}{1+T}\right] 
        \int I_1(x) e^{-(3T+1)x^2/(4\lambda)}  dx
.. math::
         = \rho_0^2 \exp\left[\frac{(2T-1)\lambda}{1+T}\right] 
        \left( \exp\left[\frac{\lambda}{3T+1}\right] - 1\right)

.. math::
         = \rho_0^2 \left(\exp\left[\frac{6\lambda T^2}{3T^2+4T+1}\right] - \exp\left[\frac{(2T-1)\lambda}{1+T}\right] \right)

.. math:: E_{M}[\rho^2] = \rho_0^2 \exp\left[\frac{6\lambda T^2}{3T^2+4T+1}\right] 

Variance of the Mass-Weighted Density :math:`V_M[\rho] = E_M[\rho^2] - E_M[\rho]^2`
```````````````````````````````````````````````````````````````````````````````````
Since correction factors are given for :math:`E_M[\rho^2]` and
:math:`E_M[\rho]`, they are not included separately here:

.. math:: V_M[\rho] = E_M[\rho^2] - E_M[\rho]^2 
          = \rho_0^2 \left( \exp\left[\frac{6\lambda T^2}{3T^2+4T+1}\right] 
          -\exp\left[\lambda\frac{4 T^2}{1+3T+2T^2}\right]
          \right)




Expectation Value of the Volume-Weighted Log Density :math:`E[\ln \rho]`
````````````````````````````````````````````````````````````````````````

.. math:: E[\ln \rho] = \int \ln \rho e^{-\lambda} \left[I_1(x) e^{-x^2/(4\lambda)} \right]dx + \int \ln \rho e^{-\lambda} \delta(u) du
.. math:: E_\delta[\ln \rho] = e^{-\lambda} \left[ \frac{\lambda T}{1+T} + \ln \rho_0 \right]
.. math:: E_{left}[\ln \rho] = \int \left[\ln \rho_0 + T\left(-\frac{x^2}{4\lambda} + \frac{\lambda}{1+T}\right) \right] e^{-\lambda} \left[I_1(x) e^{-x^2/(4\lambda)} \right]dx
.. math::  = e^{-\lambda} \left( \int \left[\ln \rho_0 + \frac{T\lambda}{1+T}\right] \left[I_1(x) e^{-x^2/(4\lambda)} \right]dx
        - \int \frac{T x^2}{4\lambda} \left[I_1(x) e^{-x^2/(4\lambda)} \right]dx \right)
.. math:: = e^{-\lambda} \left( \int \left[\ln \rho_0 + \frac{T\lambda}{1+T}\right](e^{\lambda}-1)
        - \frac{4 T \lambda^2 e^{\lambda}}{4\lambda} \right)
.. math:: = \left( \left[\ln \rho_0 + \frac{T\lambda}{1+T}\right](1-e^{-\lambda})
        - T \lambda  \right)
.. math:: E[\ln \rho] = \ln \rho_0 + \frac{T\lambda}{1+T} - T \lambda 
.. math:: = \ln \rho_0 - \frac{T^2\lambda}{1+T}

Expectation Value of the Mass-Weighted Log Density :math:`E_M[\ln \rho]`
````````````````````````````````````````````````````````````````````````

.. math:: E_M[\ln \rho] = \int \ln \rho \frac{\rho}{\rho_0} e^{-\lambda} \left[I_1(x) e^{-x^2/(4\lambda)} \right]dx + \int \ln \rho \frac{\rho}{\rho_0} e^{-\lambda} \delta(u) du
.. math:: E_\delta[\ln \rho] = \left( \frac{\lambda T}{1+T} + \ln \rho_0 \right) \exp\left[\frac{ -\lambda }{T+1}\right] 
.. math:: E_{left}[\ln \rho] = e^{-\lambda} \int \left[ 
        \left(\ln \rho_0 + T\left(-\frac{x^2}{4\lambda} + \frac{\lambda}{1+T}\right) \right) 
        \exp\left(T\left(-\frac{x^2}{4\lambda} + \frac{\lambda}{1+T}\right)\right) \right] 
        \left[I_1(x) e^{-x^2/(4\lambda)} \right]dx

Again, separate into integrable terms:

.. math:: = \exp\left(\frac{T\lambda}{1+T} -\lambda\right) \left[
        \left(\ln \rho_0 + \frac{T\lambda}{1+T} \right)  \left[I_1(x) e^{-(1+T)x^2/(4\lambda)} \right] +
        \left(-\frac{Tx^2}{4\lambda} \right)  \left[I_1(x) e^{-(1+T)x^2/(4\lambda)} \right]
        \right]

.. math:: L = \frac{\lambda}{1+T}
.. math:: E_{left}[\ln \rho] = \exp\left(\frac{T\lambda}{1+T} -\lambda\right) \left[
        \left(\ln \rho_0 + \frac{T\lambda}{1+T} \right)  \left(\exp\left[\frac{\lambda}{1+T}\right]-1\right) +
        \left(-\frac{T}{4\lambda} \right)  \left(\frac{4  \lambda^2}{(1+T)^2}  \exp\left[\frac{\lambda}{1+T}\right]\right)
        \right]
.. math:: E_{left}[\ln \rho] = \exp\left(\frac{-\lambda}{1+T}\right) \left[
        \left(\ln \rho_0 + \frac{T\lambda}{1+T} \right)  \left(\exp\left[\frac{\lambda}{1+T}\right]-1\right) +
        \left(-\frac{T}{4\lambda} \right)  \left(\frac{4  \lambda^2}{(1+T)^2}  \exp\left[\frac{\lambda}{1+T}\right]\right)
        \right]
.. math:: E_{left}[\ln \rho] = 
        \left(\ln \rho_0 + \frac{T\lambda}{1+T} \right)  \left(1-\exp\left[\frac{-\lambda}{1+T}\right]\right) -
        \left(\frac{ T \lambda}{(1+T)^2}  \right)

.. math:: E_M[\ln \rho] = \left(\ln \rho_0 + \frac{T\lambda}{1+T} \right) - 
        \left(\frac{ T \lambda}{(1+T)^2}  \right)
.. math:: = \ln \rho_0 + \frac{T^2\lambda}{(1+T)^2}


Expectation Value of the Mass-Weighted Log Density Squared :math:`E_M[\ln^2 \rho]`
``````````````````````````````````````````````````````````````````````````````````

.. math:: E_M[\ln^2 \rho] = \int (\ln \rho)^2 \frac{\rho}{\rho_0} e^{-\lambda} \left[I_1(x) e^{-x^2/(4\lambda)} \right]dx + \int (\ln \rho)^2 \frac{\rho}{\rho_0} e^{-\lambda} \delta(u) du

.. math:: E_\delta[\ln^2 \rho] = \left( \frac{\lambda T}{1+T} + \ln \rho_0 \right)^2 \exp\left[\frac{ -\lambda }{T+1}\right] 
.. math:: E_{left}[\ln^2 \rho] = e^{-\lambda} \int \left[ 
        \left(\ln \rho_0 + T\left(-\frac{x^2}{4\lambda} + \frac{\lambda}{1+T}\right) \right)^2 
        \exp\left(T\left(-\frac{x^2}{4\lambda} + \frac{\lambda}{1+T}\right)\right) \right] 
        \left[I_1(x) e^{-x^2/(4\lambda)} \right]dx

This time it's just too ugly.  Define a new variable:

.. math:: Q = \ln \rho_0 + \frac{T\lambda}{1+T}

.. math:: E_{left}[\ln^2 \rho] = e^{-\lambda/(1+T)} \int \left[ 
        \left(Q  -\frac{T x^2}{4\lambda} \right)^2 
        \exp\left(-\frac{Tx^2}{4\lambda} \right) \right] 
        \left[I_1(x) e^{-x^2/(4\lambda)} \right]dx

.. math:: E_{left}[\ln^2 \rho] = e^{-\lambda/(1+T)} \int \left[ 
        \left(Q^2  - 2 Q \frac{T x^2}{4\lambda} + \frac{T^2 x^4}{16\lambda^2} \right)
        \left[I_1(x) e^{-(1+T)x^2/(4\lambda)} \right]
        \right]dx

.. math:: E_{left}[\ln^2 \rho] = e^{-\lambda/(1+T)}  \left[ 
        Q^2 (e^{\lambda/(1+T)}-1) 
        - 2 Q \frac{T}{4\lambda} \left(\frac{4\lambda^2}{(1+T)^2} e^{\lambda/(1+T)}\right)
        + \frac{T^2}{16\lambda^2} \left(16 \frac{\lambda^3}{(1+T)^3} (\frac{\lambda}{1+T}+2) e^{\lambda/(1+T)} \right)
        \right]

.. math:: E_{left}[\ln^2 \rho] = 
        Q^2 (1-e^{-\lambda/(1+T)}) 
        - 2 Q \frac{T\lambda}{(1+T)^2} 
        + \frac{\lambda T^2}{(1+T)^3} \left(\frac{\lambda}{1+T}+2\right) 

Add back the :math:`\delta` termG

.. math:: E_M[\ln^2 \rho] = 
        Q^2
        - 2 Q \frac{T\lambda}{(1+T)^2} 
        + \frac{\lambda T^2}{(1+T)^3} \left(\frac{\lambda}{1+T}+2\right) 

Re-expand to see if it's simplifiable....

.. math:: E_M[\ln^2 \rho] = 
        \left(\ln \rho_0 + \frac{T\lambda}{1+T}\right)^2
        - 2 \left(\ln \rho_0 + \frac{T\lambda}{1+T}\right) \frac{T\lambda}{(1+T)^2} 
        + \frac{\lambda T^2}{(1+T)^3} \left(\frac{\lambda}{1+T}+2\right) 

.. math:: E_M[\ln^2 \rho] = 
        \ln^2 \rho_0 + 2 \ln \rho_0 \frac{T\lambda}{1+T} + \frac{T^2\lambda^2}{(1+T)^2}
        - 2 \ln \rho_0 \frac{T\lambda}{(1+T)^2} - 2 \frac{T\lambda}{1+T} \frac{T\lambda}{(1+T)^2} 
        + 2 \frac{\lambda T^2}{(1+T)^3}
        + \frac{\lambda T^2}{(1+T)^3} \frac{\lambda}{1+T}

.. math:: E_M[\ln^2 \rho] = 
        \ln^2 \rho_0 + 2 \ln \rho_0 \left(\frac{T\lambda}{1+T} - \frac{T\lambda}{(1+T)^2}\right) 
        + \frac{T^2\lambda^2(1+T)}{(1+T)^3}
        - 2 \frac{T^2\lambda^2}{(1+T)^3}
        + 2 \frac{\lambda T^2}{(1+T)^3}
        + \frac{\lambda^2 T^2}{(1+T)^4}

.. math:: E_M[\ln^2 \rho] = 
        \ln^2 \rho_0 + 2 \ln \rho_0 \left(\frac{T^2\lambda}{1+T}\right) 
        + \frac{T^2\lambda^2(1+T)}{(1+T)^3}
        - 2 \frac{T^2\lambda^2}{(1+T)^3}
        + 2 \frac{\lambda T^2}{(1+T)^3}
        + \frac{\lambda^2 T^2}{(1+T)^4}


.. math:: = \ln^2 \rho_0 + 2 \ln \rho_0 \frac{T^2 \lambda}{(1+T)^2} 
        + \left(\frac{\lambda T^2}{(1+T)^2}\right)^2 
        + \frac{2\lambda T^2}{(1+T)^3}

.. math:: = \left(\ln \rho_0 + \frac{T^2\lambda}{(1+T)^2}\right)^2 +
        \frac{2\lambda T^2}{(1+T)^3}


As expected, we recover the correct relation from `Hopkins 2013`_:

.. math:: S_{\ln \rho,M} = E_M[\ln \rho^2] - E_M[\ln \rho]^2 = 
        \ln^2 \rho_0 + 2 \ln \rho_0 \frac{T^2 \lambda}{(1+T)^2} 
        + \left(\frac{\lambda T^2}{(1+T)^2}\right)^2 + \frac{2\lambda T^2}{(1+T)^3}
        - \left(\ln \rho_0 + \frac{T^2\lambda}{(1+T)^2}\right)^2

.. math:: = \frac{2\lambda T^2}{(1+T)^3}

*independent* of :math:`\rho_0`.


Properties of the Hopkins PDF
=============================
I began this investigation in order to find out whether a different parameter,
i.e. something related to the compressiveness of the turbulent driving, could
be responsible for the "discrepancy" between the Formaldehyde-derived density
and the volume-averaged density of some clouds.

I naively expected that, in compressive turbulence, more of the mass will be
concentrated at higher density, which should drive up the mass-weighted mean
density.  

`Hopkins 2013`_ showed that :math:`T\sim\mathcal{M}_C`.  If this is taken on
face value, without recognizing the relation between :math:`T` and
:math:`\sigma`, it means that for *fixed* :math:`\sigma`,
:math:`<\rho>_M\propto e^{-T^2}`. 

This figure shows the relation of the various moments and :math:`T`.  The plots
show both the "theoretical" result (i.e., doing the integral by hand) and the
numerical result with 50,000 data points plus a correction for the
:math:`\delta` terms; the agreement is essentially perfect excepting some
numerical noise (the only visible discrepancy is for a perfect lognormal at
high :math:`\sigma_V`):

.. image:: |filename|/images/rho1_varyT_colorSigma.png
    :width: 800

However, `Hopkins 2013`_ also found that simulations generally produced
:math:`T\sim0.1\sigma_{s,M}^2` (though a relation of the form :math:`T\sim 0.25
\ln(1+0.25\sigma_{s,M}^4)` is also a good fit).

If you use the simpler of these relations, the expected relationship (mean mass
increasing with increasing "compressiveness") is recovered.  However, it comes
with an increasing :math:`\sigma_s`.  In these plots, half are labeled with
:math:`\sigma` and the others are labeled with :math:`T`, though the two are
equivalent.  The highest :math:`\sigma_s` observed in any of the simulations
shown in `Hopkins 2013`_ was about :math:`\sigma_s=4` (marked with a black
square below), so the maximum :math:`\rho_M/\rho_V \sim 10^2`.

.. image:: |filename|/images/TSigma.png
    :width: 800

.. _Hopkins 2013: http://adsabs.harvard.edu/abs/2013MNRAS.430.1880H
