"""
Exponential parameter-sensitivity analysis for turbulence-regulated
star-formation theories.

This module provides the *analytic mechanism* behind the numerical results of
Barnes et al. 2017 (MNRAS 469, 2263, Appendix C) and Schruba, Kruijssen &
Leroy 2019 (ApJ 883, 2): turbulence-regulated star-formation theories (KM05,
PN11, HC11/HC13, unified by Federrath & Klessen 2012, ApJ 761, 156; hereafter
FK12) predict the star-formation efficiency per free-fall time ``eps_ff`` by
integrating a stationary lognormal density PDF above a critical log-density
``s_crit``.  Because the prediction depends *exponentially* on
``s_crit**2 / sigma_s**2``, and ``sigma_s**2`` contains the free O(1) driving
parameter ``b``, the theory's output envelope is exponentially wide by
construction.  The observed phenomenon (eps_ff ~ 0.3-3%, Lee et al. 2016;
Utomo et al. 2018) is narrower than that envelope, so agreement with data
cannot confirm the theory.

Conventions
-----------
**All internal math uses the natural-log density contrast**
``s = ln(rho / rho_0)``.  The volume-weighted density PDF is a lognormal in
``s``,

    p_V(s) = N(s; mu, sigma_s**2),   mu = -sigma_s**2 / 2

where the mean ``mu = -sigma_s**2 / 2`` encodes closed-box mass conservation
(``<rho> = rho_0``).  This pins the PDF *peak* location by bookkeeping, not by
physics (cf. Lombardi, Alves & Lada 2015, A&A 576, L1 for the observational
analogue); see the module caveats below.

Much of the rest of this repository (``turbulent_pdfs.py``, ``hopkins_pdf.py``)
mixes ``log10`` and ``ln`` conventions.  To avoid inheriting that ambiguity,
this module works entirely in ``ln`` internally and exposes explicit
converters (`ln_from_log10`, `log10_from_ln`, `sigma_ln_from_log10`, ...) for
interfacing with the ``log10`` arrays used elsewhere.

Two sensitivity questions
-------------------------
1. *Theory-parameter freedom* -- how far ``eps_ff`` moves when the free
   parameters (chiefly the driving parameter ``b``) are varied over their
   plausible ranges.  See `elasticity_sigma`, `elasticity_b`, `jacobian`,
   `envelope_epsff`.
2. *Observational error propagation* -- how measurement uncertainties in the
   physical observables (mean density, cloud size, velocity dispersion,
   temperature) propagate to ``eps_ff``.  See `observable_jacobian`,
   `uncertainty_budget`, `montecarlo_epsff`.

The second analysis exposes a structural asymmetry that mirrors the first:
the **mean density** (set by mass/volume) enters ``eps_ff`` *only* through the
collapse threshold ``s_crit`` -- with no compensating term -- so even a small
mass/volume uncertainty propagates undamped, just as ``b`` does through the PDF
width.  The **velocity dispersion**, by contrast, enters *both* the width
``sigma_s`` (broadening the PDF) *and* the threshold (through
``alpha_vir M^2 ∝ sigma_v^4``); these partially cancel, so ``eps_ff`` is far
*less* sensitive to ``sigma_v`` -- the same cancellation mechanism that makes
``M`` less leveraged than ``b``.  Consequently the mean density (mass/volume),
which is also the *most poorly constrained* observable, dominates the predicted
``eps_ff`` uncertainty budget, while the well-measured velocity dispersion
contributes negligibly.

Caveats (see also the paper text these support)
-----------------------------------------------
- ``mu = -sigma_s**2 / 2`` is closed-box mass-conservation bookkeeping; the PDF
  peak location is not an independent physical prediction.
- Multi-free-fall tail integrals can exceed unity at high ``sigma_s``.  This is
  a known pathology of the FK12 multi-free-fall construction, *not* a bug here.
  `epsff` and `tail_moment` **warn** rather than clip.
- The analysis assumes the lognormal holds up to ``s_crit``.  Self-gravitating
  gas develops power-law tails (Kritsuk, Norman & Wagner 2011; Girichidis et
  al. 2014), a *separate* (arguably stronger) failure mode.  This module
  quantifies the parameter-freedom problem *conditional on* the theory's own
  lognormal PDF assumption.
- The PN11 / HC13 ``s_crit`` coefficients are transcribed from FK12 Table 1 and
  should be re-verified against the paper before use in a publication.

References
----------
Krumholz & McKee 2005, ApJ 630, 250 (KM05);
Padoan & Nordlund 2011, ApJ 730, 40 (PN11);
Hennebelle & Chabrier 2011, ApJL 743, L29; 2013, ApJ 770, 150 (HC);
Federrath & Klessen 2012, ApJ 761, 156 (FK12);
Molina et al. 2012, MNRAS 423, 2680;
Federrath et al. 2008, ApJL 688, L79; 2010, A&A 512, A81;
Barnes et al. 2017, MNRAS 469, 2263;
Schruba, Kruijssen & Leroy 2019, ApJ 883, 2;
Lee, Miville-Deschenes & Murray 2016, ApJ 833, 229;
Utomo et al. 2018, ApJL 861, L18.
"""
import warnings

import numpy as np
from scipy.special import erfc

# natural-log <-> log10 rescaling (matches ``ln10`` in turbulent_pdfs.py)
ln10 = np.log(10)


# ---------------------------------------------------------------------------
# log-base converters (s = ln(rho/rho_0)  <->  the log10 arrays used elsewhere)
# ---------------------------------------------------------------------------
def ln_from_log10(x_log10):
    """Convert a log10 density contrast to natural log: s = ln10 * log10."""
    return np.asarray(x_log10) * ln10


def log10_from_ln(s):
    """Convert a natural-log density contrast s = ln(rho/rho_0) to log10."""
    return np.asarray(s) / ln10


def sigma_ln_from_log10(sigma_log10):
    """Convert a log10 PDF width to the natural-log width sigma_s."""
    return np.asarray(sigma_log10) * ln10


def sigma_log10_from_ln(sigma_s):
    """Convert a natural-log PDF width sigma_s to a log10 PDF width."""
    return np.asarray(sigma_s) / ln10


# ---------------------------------------------------------------------------
# variance-Mach relation (Molina et al. 2012 magnetized form)
# ---------------------------------------------------------------------------
def _beta_factor(beta):
    """Magnetic correction beta/(beta+1); -> 1 as beta -> inf (hydro)."""
    beta = np.asarray(beta, dtype=float)
    # beta -> inf gives 1; guard the divide so np.inf is handled cleanly
    # (np.where still evaluates both branches, hence the errstate).
    with np.errstate(invalid='ignore'):
        return np.where(np.isinf(beta), 1.0, beta / (beta + 1.0))


def sigma_s_sq(b, mach, beta=np.inf):
    r"""
    Variance of the natural-log density PDF (Molina et al. 2012).

    .. math::
        \sigma_s^2 = \ln\!\left(1 + b^2 M^2 \frac{\beta}{\beta+1}\right)

    Parameters
    ----------
    b : float
        Turbulent driving parameter, 1/3 (solenoidal) -> 1 (compressive)
        (Federrath et al. 2008, 2010).
    mach : float
        Sonic Mach number M.
    beta : float
        Plasma beta; ``beta -> inf`` recovers the hydrodynamic form.

    Returns
    -------
    float or ndarray
        ``sigma_s**2`` (the *variance*, not the standard deviation).
    """
    b = np.asarray(b, dtype=float)
    mach = np.asarray(mach, dtype=float)
    return np.log(1.0 + b ** 2 * mach ** 2 * _beta_factor(beta))


def sigma_s(b, mach, beta=np.inf):
    """Standard deviation ``sigma_s`` of the ln-density PDF.  See `sigma_s_sq`."""
    return np.sqrt(sigma_s_sq(b, mach, beta))


# ---------------------------------------------------------------------------
# critical log-density models (FK12 Table 1)
# ---------------------------------------------------------------------------
def _fbeta_km(beta):
    r"""
    KM05 magnetic correction (FK12 Table 1, KM row).

    .. math::
        f(\beta) = \left(1 + \beta^{-1}\right)^{-1} = \frac{\beta}{\beta+1}

    Reduces to 1 in the hydrodynamic limit ``beta -> inf``.
    """
    return _beta_factor(beta)


def _fbeta_pn(beta):
    r"""
    PN11 magnetic correction (FK12 Eq. 31), used by the PN and multi-ff PN
    models *only* (per the Table 1 caption -- do NOT apply it to KM):

    .. math::
        f(\beta) = \frac{\left(1 + 0.925\,\beta^{-3/2}\right)^{2/3}}
                        {\left(1 + \beta^{-1}\right)^{2}}

    Reduces to 1 as ``beta -> inf``.
    """
    beta = np.asarray(beta, dtype=float)
    inv = np.where(np.isinf(beta), 0.0, 1.0 / np.where(np.isinf(beta), 1.0, beta))
    return (1.0 + 0.925 * inv ** 1.5) ** (2.0 / 3.0) / (1.0 + inv) ** 2


def scrit_km05(alpha_vir, mach, beta=np.inf, phi_x=1.12, theta=0.0,
               y_cut=0.1):
    r"""
    KM05 critical log-density, FK12 Table 1 form.

    .. math::
        s_{\rm crit} = \ln\!\left(\frac{\pi^2}{5}\,\phi_x^2\,\alpha_{\rm vir}\,
                                  M^2\, f(\beta)\right)

    Parameters
    ----------
    alpha_vir : float
        Virial parameter.
    mach : float
        Sonic Mach number.
    beta : float
        Plasma beta (``inf`` -> hydro).
    phi_x : float
        Order-unity KM05 fudge factor (fiducial 1.12).
    theta : float
        Unused for KM05; accepted so all ``scrit_*`` share a signature.

    Returns
    -------
    float or ndarray
        ``s_crit`` in natural-log units.
    """
    alpha_vir = np.asarray(alpha_vir, dtype=float)
    mach = np.asarray(mach, dtype=float)
    return np.log((np.pi ** 2 / 5.0) * phi_x ** 2 * alpha_vir * mach ** 2
                  * _fbeta_km(beta))


def scrit_pn11(alpha_vir, mach, beta=np.inf, phi_x=1.12, theta=0.35,
               y_cut=0.1):
    r"""
    PN11 critical log-density, FK12 Table 1 form.

    .. math::
        s_{\rm crit} = \ln\!\left(0.067\,\theta^{-2}\,\alpha_{\rm vir}\,M^2\,
                                  f(\beta)\right)

    with the FK12 fiducial ``theta = 0.35``.  The magnetic correction
    ``f(beta)`` here is the PN11-specific FK12 Eq. (31) form (`_fbeta_pn`),
    which differs from the KM05 factor -- per the FK12 Table 1 caption.

    Parameters
    ----------
    theta : float
        PN11 order-unity parameter (fiducial 0.35).

    See `scrit_km05` for the remaining parameters.
    """
    alpha_vir = np.asarray(alpha_vir, dtype=float)
    mach = np.asarray(mach, dtype=float)
    return np.log(0.067 * theta ** -2 * alpha_vir * mach ** 2
                  * _fbeta_pn(beta))


def scrit_hc13(alpha_vir, mach, beta=np.inf, phi_x=1.12, theta=0.0,
               y_cut=0.1):
    r"""
    HC11/HC13 critical log-density, FK12 Table 1 (Eqs. 37-39).

    Unlike KM05/PN11 (single power of ``M``), the HC critical density is a
    **sum of a thermal and a turbulent term inside the log**:

    .. math::
        s_{\rm crit} = \ln\!\left(
            \underbrace{\frac{\pi^2}{5}\, y_{\rm cut}^{-2}\,\alpha_{\rm vir}\,
                        M^{-2}\,(1 + \beta^{-1})}_{\text{thermal (Eq.\ 38)}}
          + \underbrace{\frac{\pi^2}{15}\, y_{\rm cut}^{-1}\,
                        \alpha_{\rm vir}}_{\text{turbulent (Eq.\ 39)}}\right)

    Note the thermal term scales as ``M^{-2}`` (inverse!) and carries the
    magnetic factor ``(1 + beta^{-1})`` in the *numerator* -- opposite to the
    KM/PN sense.  ``y_cut = 0.1`` is the HC cutoff scale; ``phi_x`` / ``theta``
    are unused (accepted for a common signature).

    See `scrit_km05` for the remaining parameters.
    """
    alpha_vir = np.asarray(alpha_vir, dtype=float)
    mach = np.asarray(mach, dtype=float)
    beta = np.asarray(beta, dtype=float)
    inv = np.where(np.isinf(beta), 0.0, 1.0 / np.where(np.isinf(beta), 1.0, beta))
    thermal = (np.pi ** 2 / 5.0) * y_cut ** -2 * alpha_vir * mach ** -2 * (1.0 + inv)
    turbulent = (np.pi ** 2 / 15.0) * y_cut ** -1 * alpha_vir
    return np.log(thermal + turbulent)


SCRIT_MODELS = {'km05': scrit_km05, 'pn11': scrit_pn11, 'hc13': scrit_hc13}


# ---------------------------------------------------------------------------
# master integral: exponential moment of a Gaussian tail  (spec section 2.3)
# ---------------------------------------------------------------------------
def tail_moment(s_c, sigma, lam=1.0, warn=True):
    r"""
    Exponential moment of the volume-weighted lognormal tail.

    For weight ``exp(lam * s)`` and ``p_V(s) = N(s; mu, sigma**2)`` with
    ``mu = -sigma**2 / 2``,

    .. math::
        I_\lambda(s_c) = \int_{s_c}^{\infty} e^{\lambda s}\, p_V(s)\, ds
                       = e^{\lambda \mu + \lambda^2 \sigma^2 / 2}\,
                         \tfrac12 \operatorname{erfc}\!
                         \left(\frac{s_c - \mu - \lambda \sigma^2}
                                    {\sqrt{2}\,\sigma}\right)

    Everything else in this module is a special case:

    - ``lam = 1``   -> single-free-fall mass fraction (KM05-type), `massfrac_above`
    - ``lam = 3/2`` -> multi-free-fall integrand (FK12), `multiff_above`

    Parameters
    ----------
    s_c : float or ndarray
        Lower integration limit (critical log-density).
    sigma : float or ndarray
        Standard deviation ``sigma_s`` of the ln-density PDF.
    lam : float
        Weight exponent ``lambda``.
    warn : bool
        If True, warn when the returned value exceeds 1 (a known FK12
        multi-free-fall pathology at high ``sigma``; not clipped).

    Returns
    -------
    float or ndarray
        The tail moment ``I_lambda(s_c)``.
    """
    sigma = np.asarray(sigma, dtype=float)
    s_c = np.asarray(s_c, dtype=float)
    mu = -sigma ** 2 / 2.0
    prefactor = np.exp(lam * mu + lam ** 2 * sigma ** 2 / 2.0)
    z = (s_c - mu - lam * sigma ** 2) / (np.sqrt(2.0) * sigma)
    val = prefactor * 0.5 * erfc(z)
    if warn and np.any(val > 1.0):
        warnings.warn(
            "tail_moment exceeds 1 (lam={}): a known FK12 multi-free-fall "
            "pathology at high sigma; value not clipped.".format(lam),
            stacklevel=2)
    return val


def massfrac_above(s_c, sigma):
    r"""
    Single-free-fall mass fraction above ``s_c`` (KM05-type; ``lam = 1``).

    .. math::
        f(>s_c) = \tfrac12 \operatorname{erfc}\!
                  \left(\frac{s_c - \sigma^2/2}{\sqrt{2}\,\sigma}\right)
    """
    return tail_moment(s_c, sigma, lam=1.0)


def multiff_above(s_c, sigma):
    r"""
    Multi-free-fall tail integrand above ``s_c`` (FK12; ``lam = 3/2``).

    .. math::
        g(>s_c) = e^{3\sigma^2/8}\, \tfrac12 \operatorname{erfc}\!
                  \left(\frac{s_c - \sigma^2}{\sqrt{2}\,\sigma}\right)

    Can exceed unity at high ``sigma`` (warned by `tail_moment`).
    """
    return tail_moment(s_c, sigma, lam=1.5)


# ---------------------------------------------------------------------------
# eps_ff assembly (FK12 conventions)
# ---------------------------------------------------------------------------
def epsff(b, mach, alpha_vir, beta=np.inf, model='km05', multiff=True,
          eps_core=0.5, phi_t=1.0, s_crit=None, warn=True, **scrit_kwargs):
    r"""
    Star-formation efficiency per free-fall time.

    .. math::
        \epsilon_{\rm ff} = \frac{\epsilon_{\rm core}}{\phi_t}\,
            \begin{cases} g(>s_{\rm crit}) & \text{multi-free-fall} \\
                          f(>s_{\rm crit}) & \text{single-free-fall} \end{cases}

    ``eps_core`` (default 0.5) and ``phi_t`` (default 1.0) are overall
    normalizations: they scale ``eps_ff`` but do **not** affect elasticities
    with respect to ``b`` (or any other shape parameter).

    Parameters
    ----------
    b, mach, alpha_vir, beta : float
        Physical parameters (see `sigma_s_sq`, `scrit_km05`).
    model : {'km05', 'pn11', 'hc13'}
        Critical-density model.  Ignored if ``s_crit`` is given.
    multiff : bool
        Multi-free-fall (``lam = 3/2``) vs single-free-fall (``lam = 1``).
    eps_core, phi_t : float
        Pass-through normalizations.
    s_crit : float, optional
        Decoupled mode: use this ``s_crit`` directly instead of computing it
        from ``model``.
    scrit_kwargs
        Extra keyword args forwarded to the ``scrit_*`` function
        (e.g. ``phi_x``, ``theta``, ``y_cut``).

    Returns
    -------
    float or ndarray
        ``eps_ff``.  Warns if it exceeds 1 (unphysical; not clipped).
    """
    sig = sigma_s(b, mach, beta)
    if s_crit is None:
        s_crit = SCRIT_MODELS[model](alpha_vir, mach, beta=beta, **scrit_kwargs)
    lam = 1.5 if multiff else 1.0
    tail = tail_moment(s_crit, sig, lam=lam, warn=warn)
    val = (eps_core / phi_t) * tail
    if warn and np.any(val > 1.0):
        warnings.warn(
            "eps_ff exceeds 1 (unphysical); value not clipped.", stacklevel=2)
    return val


# ---------------------------------------------------------------------------
# analytic sensitivity  (spec section 3)
# ---------------------------------------------------------------------------
def _z_of(s_c, sigma, lam):
    """erfc argument z = (s_c - mu - lam*sigma^2)/(sqrt(2)*sigma), mu=-sigma^2/2."""
    mu = -sigma ** 2 / 2.0
    return (s_c - mu - lam * sigma ** 2) / (np.sqrt(2.0) * sigma)


def elasticity_sigma(s_c, sigma, multiff=False, method='exact'):
    r"""
    Logarithmic sensitivity ``d ln I_lambda / d ln sigma`` of the tail moment.

    This is the core sensitivity result.  Three evaluation methods:

    ``method='asymptotic'``
        The two-term leading expression from the ``erfc(t) ~
        exp(-t^2)/(t sqrt(pi))`` expansion of the *single-free-fall* form:

        .. math::
            \frac{d \ln f}{d \ln \sigma} \approx
            \frac{s_c^2}{\sigma^2} - \frac{\sigma^2}{4}

        (``multiff`` is ignored for this branch: it is the single-ff asymptote.)

    ``method='exact'``
        The exact closed-form derivative of the erfc expression in
        `tail_moment`.  Writing ``I = A(sigma) * 0.5 * erfc(z)`` with
        ``ln A = lambda(lambda-1) sigma^2 / 2``,

        .. math::
            \frac{d \ln I}{d \ln \sigma} =
            \lambda(\lambda-1)\sigma^2
            - \frac{2}{\sqrt{\pi}} \frac{e^{-z^2}}{\operatorname{erfc}(z)}
              \, \sigma \frac{dz}{d\sigma}

    ``method='fd'``
        Central finite-difference of ``ln I`` in ``ln sigma`` (cross-check).

    The exact value exceeds the two-term asymptote by a positive O(1)
    prefactor contribution.  Fiducial single-ff case
    (``s_c=4``, ``sigma^2=2.49412``): asymptote 5.79, exact 7.14.

    Parameters
    ----------
    s_c : float or ndarray
        Critical log-density.
    sigma : float or ndarray
        ln-density PDF width.
    multiff : bool
        ``lam = 3/2`` (multi-ff) if True, else ``lam = 1`` (single-ff).
        Ignored when ``method='asymptotic'``.
    method : {'exact', 'asymptotic', 'fd'}

    Returns
    -------
    float or ndarray
        ``d ln I / d ln sigma``.
    """
    sigma = np.asarray(sigma, dtype=float)
    s_c = np.asarray(s_c, dtype=float)
    lam = 1.5 if multiff else 1.0

    if method == 'asymptotic':
        # single-ff two-term leading expression
        return s_c ** 2 / sigma ** 2 - sigma ** 2 / 4.0

    if method == 'exact':
        z = _z_of(s_c, sigma, lam)
        # z = s_c/(sqrt2 sigma) + (1/2 - lam) sigma/sqrt2
        # sigma * dz/dsigma:
        sig_dz = -s_c / (np.sqrt(2.0) * sigma) + (0.5 - lam) * sigma / np.sqrt(2.0)
        # d ln(erfc z)/dz = -2/sqrt(pi) exp(-z^2)/erfc(z)
        dln_erfc = -2.0 / np.sqrt(np.pi) * np.exp(-z ** 2) / erfc(z)
        return lam * (lam - 1.0) * sigma ** 2 + dln_erfc * sig_dz

    if method == 'fd':
        h = 1e-6
        fp = tail_moment(s_c, sigma * (1 + h), lam=lam, warn=False)
        fm = tail_moment(s_c, sigma * (1 - h), lam=lam, warn=False)
        return (np.log(fp) - np.log(fm)) / (2 * h)

    raise ValueError("method must be 'exact', 'asymptotic', or 'fd'")


def dlnsigma_dlnb(b, mach, beta=np.inf):
    r"""
    Chain-rule factor ``d ln sigma_s / d ln b``.

    With ``sigma_s^2 = ln(1 + b^2 M^2 beta')`` and ``beta' = beta/(beta+1)``,

    .. math::
        \frac{d \ln \sigma_s}{d \ln b} =
        \frac{b^2 M^2 \beta'}{(1 + b^2 M^2 \beta')\,
              \ln(1 + b^2 M^2 \beta')}

    (``b`` enters *only* through ``sigma_s`` -- unlike ``M``, which also enters
    ``s_crit``.  This is why ``b`` is the maximally-leveraged parameter.)

    Note: the factor of 2 from ``x \propto b^2`` cancels the ``1/2`` from
    ``sigma_s = (\ln(1+x))^{1/2}``, so there is *no* residual factor of 2 in
    the denominator (verified against finite differences; the spec draft's
    extra 2 was an error).
    """
    b = np.asarray(b, dtype=float)
    mach = np.asarray(mach, dtype=float)
    x = b ** 2 * mach ** 2 * _beta_factor(beta)
    return x / ((1.0 + x) * np.log(1.0 + x))


def elasticity_b(b, mach, alpha_vir=None, beta=np.inf, multiff=False,
                 model='km05', s_crit=None, method='exact', **scrit_kwargs):
    r"""
    Elasticity of the tail moment with respect to ``b``:
    ``d ln I / d ln b = elasticity_sigma * (d ln sigma / d ln b)``.

    ``s_crit`` is held fixed with respect to ``b`` (``b`` does not appear in any
    ``s_crit`` model), so the chain rule is a single product.

    Provide either ``s_crit`` directly (decoupled mode) or ``alpha_vir`` (and
    ``model``) to compute it.
    """
    sig = sigma_s(b, mach, beta)
    if s_crit is None:
        if alpha_vir is None:
            raise ValueError("provide s_crit or alpha_vir")
        s_crit = SCRIT_MODELS[model](alpha_vir, mach, beta=beta, **scrit_kwargs)
    es = elasticity_sigma(s_crit, sig, multiff=multiff, method=method)
    return es * dlnsigma_dlnb(b, mach, beta)


# ---------------------------------------------------------------------------
# full Jacobian  (spec section 3.3)
# ---------------------------------------------------------------------------
def jacobian(fiducials, model='km05', multiff=True,
             params=('b', 'mach', 'alpha_vir', 'beta', 'phi_x'),
             rel_step=1e-5):
    r"""
    Log-log Jacobian ``d ln eps_ff / d ln x`` for each parameter ``x``.

    Computed by central finite difference in ``ln x`` of `epsff`, so it fully
    captures the structure the tornado figure must reveal: **M enters both
    ``sigma^2`` (denominator) and ``s_crit`` (numerator), partially cancelling,
    while ``b`` enters only ``sigma^2``** -- so ``b`` is the maximally-leveraged
    parameter.  (This is computed, not asserted.)

    Parameters
    ----------
    fiducials : dict
        Fiducial values.  Must contain the physical parameters
        (``b``, ``mach``, ``alpha_vir``) and may contain ``beta``, ``phi_x``,
        ``theta``, ``y_cut``, ``eps_core``, ``phi_t``.
    model : {'km05', 'pn11', 'hc13'}
    multiff : bool
    params : sequence of str
        Which parameters to differentiate.  ``beta`` is skipped automatically
        if the fiducial ``beta`` is infinite (elasticity -> 0 there).
    rel_step : float
        Relative step in ``ln x`` for the central difference.

    Returns
    -------
    dict
        ``{param: d ln eps_ff / d ln param}``.
    """
    base = dict(fiducials)
    base.setdefault('beta', np.inf)

    # keys accepted by epsff / scrit
    scrit_keys = ('phi_x', 'theta', 'y_cut')

    def eval_epsff(kw):
        call = dict(b=kw['b'], mach=kw['mach'], alpha_vir=kw['alpha_vir'],
                    beta=kw.get('beta', np.inf), model=model, multiff=multiff,
                    eps_core=kw.get('eps_core', 0.5),
                    phi_t=kw.get('phi_t', 1.0), warn=False)
        for k in scrit_keys:
            if k in kw:
                call[k] = kw[k]
        return epsff(**call)

    out = {}
    for p in params:
        if p == 'beta' and not np.isfinite(base.get('beta', np.inf)):
            out[p] = 0.0  # b/(b+1) is flat at beta=inf; elasticity vanishes
            continue
        if p not in base:
            continue
        x0 = base[p]
        if not np.isfinite(x0) or x0 == 0:
            out[p] = 0.0
            continue
        hi = dict(base); lo = dict(base)
        hi[p] = x0 * (1 + rel_step)
        lo[p] = x0 * (1 - rel_step)
        fhi = eval_epsff(hi); flo = eval_epsff(lo)
        out[p] = (np.log(fhi) - np.log(flo)) / (2 * rel_step)
    return out


# ---------------------------------------------------------------------------
# envelope of predictions  (the money figure)
# ---------------------------------------------------------------------------
def envelope_epsff(mach_grid, b_range=(1.0 / 3, 1.0), alpha_vir=1.0,
                   beta=np.inf, model='km05', multiff=True,
                   eps_core=0.5, phi_t=1.0, n_b=41, **scrit_kwargs):
    r"""
    Lower/upper envelope of ``eps_ff`` over the driving-parameter range.

    Spans ``b`` from ``b_range[0]`` (solenoidal, ~1/3) to ``b_range[1]``
    (compressive, 1) at each Mach number, returning the min and max ``eps_ff``.
    The width of ``(lo, hi)`` at fixed ``M`` is the theory's parameter envelope
    -- the quantity the "money figure" compares against the observed band.

    Parameters
    ----------
    mach_grid : ndarray
        Mach numbers.
    b_range : (float, float)
        Driving-parameter range to span.
    n_b : int
        Number of ``b`` samples used to find the envelope (``eps_ff`` is
        monotone in ``b`` here, so the extremes fall at the endpoints, but we
        sample to be robust).

    Returns
    -------
    (lo, hi) : (ndarray, ndarray)
        Lower and upper ``eps_ff`` envelopes over ``mach_grid``.
    """
    mach_grid = np.asarray(mach_grid, dtype=float)
    b_vals = np.linspace(b_range[0], b_range[1], n_b)
    # shape (n_b, n_mach)
    grid = np.empty((n_b, mach_grid.size))
    for i, b in enumerate(b_vals):
        grid[i] = epsff(b, mach_grid, alpha_vir, beta=beta, model=model,
                        multiff=multiff, eps_core=eps_core, phi_t=phi_t,
                        warn=False, **scrit_kwargs)
    return grid.min(axis=0), grid.max(axis=0)


# ---------------------------------------------------------------------------
# Observational error propagation
# ---------------------------------------------------------------------------
# How the *dimensionless theory inputs* (alpha_vir, mach, b) depend on the
# *physical observables*, as power laws: entries are exponents
# ``d ln(theory input) / d ln(observable)``.  These follow from the standard
# definitions and hold for every s_crit model (the model only changes how
# eps_ff depends on alpha_vir/mach, which is handled separately):
#
#   Mach number      M       = sigma_v / c_s
#   virial parameter alpha_vir = 5 sigma_v^2 R / (G M_cloud)
#
# Two self-consistent observable bases:
#
# 'density'  -- independent observables {mean_density rho_0, radius R,
#               sigma_v, c_s, b}; the cloud mass M_cloud = rho_0 * (4/3) pi R^3
#               is *derived*.  Then
#                   alpha_vir = (15 / 4 pi) sigma_v^2 / (G rho_0 R^2)
#               so   ln alpha_vir = 2 ln sigma_v - ln rho_0 - 2 ln R + const.
#               rho_0 enters ONLY the threshold (via alpha_vir), with no
#               compensating term -- the "pure threshold" observable, the
#               mean-density analog of b (the "pure width" parameter).
#
# 'mass_volume' -- independent observables {mass M_cloud, volume V, sigma_v,
#               c_s, b}; here R = (3V/4pi)^{1/3}, so
#                   alpha_vir = 5 sigma_v^2 R / (G M_cloud) ∝ sigma_v^2 V^{1/3} / M
#               and the mean density rho_0 = M/V is derived.  Note volume enters
#               only weakly (exponent 1/3): the rho_0 (V^-1) and R^2 (V^{2/3})
#               dependences of alpha_vir partially cancel -- so a mean-density
#               error propagates very differently depending on whether it comes
#               from the mass or the volume.
OBSERVABLE_EXPONENTS = {
    'density': {
        'mean_density': {'alpha_vir': -1.0},
        'radius':       {'alpha_vir': -2.0},
        'sigma_v':      {'alpha_vir': 2.0, 'mach': 1.0},
        'c_s':          {'mach': -1.0},
        'b':            {'b': 1.0},
    },
    'mass_volume': {
        'mass':    {'alpha_vir': -1.0},
        'volume':  {'alpha_vir': 1.0 / 3.0},
        'sigma_v': {'alpha_vir': 2.0, 'mach': 1.0},
        'c_s':     {'mach': -1.0},
        'b':       {'b': 1.0},
    },
}

# Fiducial fractional (natural-log) uncertainties on the observables, chosen to
# reflect the *observational reality*, not just the theory:
#   - mean density (mass/volume): poorly constrained -- distance^2 for masses,
#     assumed geometry for volumes, chemical abundances for tracers.  Factor
#     ~1.6 (sigma_ln ~ 0.5) is generous.
#   - cloud radius: boundary definition + distance; factor ~1.35.
#   - velocity dispersion: measured directly from line widths; ~10%.
#   - sound speed: from temperature (c_s ∝ sqrt(T)); ~10%.
#   - b (driving parameter): a *theory* freedom spanning 1/3 -> 1 (factor 3),
#     sigma_ln ~ 0.45; kept separate from the observational budget.
DEFAULT_LN_SIGMA = {
    'mean_density': 0.5, 'radius': 0.3, 'sigma_v': 0.1, 'c_s': 0.1,
    'mass': 0.5, 'volume': 0.5, 'b': 0.45,
}


def theory_inputs_from_observables(factors, fiducial, basis='density'):
    r"""
    Map multiplicative perturbations of the observables to the dimensionless
    theory inputs ``(b, mach, alpha_vir)``.

    Parameters
    ----------
    factors : dict
        ``{observable: multiplicative factor}`` relative to the fiducial
        (missing observables default to 1, i.e. no perturbation).
    fiducial : dict
        Fiducial dimensionless inputs; must contain ``b``, ``mach``,
        ``alpha_vir``.
    basis : {'density', 'mass_volume'}

    Returns
    -------
    (b, mach, alpha_vir) : tuple of float
    """
    exps = OBSERVABLE_EXPONENTS[basis]
    ln = {'b': 0.0, 'mach': 0.0, 'alpha_vir': 0.0}
    for obs, factor in factors.items():
        if obs not in exps:
            raise KeyError("unknown observable {!r} for basis {!r}"
                           .format(obs, basis))
        for tin, e in exps[obs].items():
            ln[tin] += e * np.log(factor)
    return (fiducial['b'] * np.exp(ln['b']),
            fiducial['mach'] * np.exp(ln['mach']),
            fiducial['alpha_vir'] * np.exp(ln['alpha_vir']))


def epsff_from_observables(factors, fiducial, model='km05', multiff=True,
                           basis='density', eps_core=0.5, phi_t=1.0,
                           **scrit_kwargs):
    """
    ``eps_ff`` as a function of multiplicative perturbations to the observables.

    Thin wrapper: maps ``factors`` through `theory_inputs_from_observables` and
    evaluates `epsff`.  See `observable_jacobian` for the log-log sensitivities.
    """
    b, mach, alpha_vir = theory_inputs_from_observables(factors, fiducial,
                                                        basis=basis)
    return epsff(b, mach, alpha_vir, model=model, multiff=multiff,
                 eps_core=eps_core, phi_t=phi_t, warn=False, **scrit_kwargs)


def observable_jacobian(fiducial, model='km05', multiff=True, basis='density',
                        rel_step=1e-6):
    r"""
    Log-log sensitivity of ``eps_ff`` to each **physical observable**:
    ``d ln eps_ff / d ln X`` for ``X`` in the chosen ``basis``.

    This is the observational counterpart of `jacobian` (which differentiates
    w.r.t. the dimensionless theory inputs).  It reveals the physically crucial
    structure:

    - **mean density** (and, in the mass/volume basis, the **mass**) enters
      ``eps_ff`` *only* through the threshold ``s_crit`` -- no compensating
      term -- so its uncertainty propagates undamped.
    - **velocity dispersion** enters *both* the PDF width ``sigma_s`` (broaden
      -> more dense gas) *and* the threshold via ``alpha_vir M^2 ∝ sigma_v^4``
      (raise -> less dense gas); these partially cancel, so ``eps_ff`` is far
      *less* sensitive to ``sigma_v`` than one might expect.  (Same mechanism
      as the b-vs-M cancellation in `jacobian`.)

    Computed by central finite difference of `epsff_from_observables` (the
    exponent map is the single source of truth).

    Returns
    -------
    dict
        ``{observable: d ln eps_ff / d ln observable}``.
    """
    out = {}
    for obs in OBSERVABLE_EXPONENTS[basis]:
        fhi = epsff_from_observables({obs: 1 + rel_step}, fiducial, model=model,
                                     multiff=multiff, basis=basis)
        flo = epsff_from_observables({obs: 1 - rel_step}, fiducial, model=model,
                                     multiff=multiff, basis=basis)
        out[obs] = (np.log(fhi) - np.log(flo)) / (2 * rel_step)
    return out


def uncertainty_budget(fiducial, ln_sigma=None, model='km05', multiff=True,
                       basis='density'):
    r"""
    Linearized (first-order Gaussian) propagation of observational
    uncertainties to ``eps_ff``.

    For independent observables with fractional log-uncertainties
    ``sigma_{ln X}``,

    .. math::
        \sigma_{\ln \epsilon_{\rm ff}}^2 =
            \sum_X \left(\frac{d\ln\epsilon_{\rm ff}}{d\ln X}\right)^2
                   \sigma_{\ln X}^2

    Parameters
    ----------
    fiducial : dict
        Fiducial ``b``, ``mach``, ``alpha_vir``.
    ln_sigma : dict, optional
        Per-observable fractional log-uncertainties; defaults to
        `DEFAULT_LN_SIGMA`.
    model, multiff, basis
        As in `observable_jacobian`.

    Returns
    -------
    dict
        ``{'elasticity': {X: dlneps/dlnX}, 'contribution': {X: variance share},
           'sigma_ln_epsff': float, 'factor': exp(sigma_ln_epsff)}``.
        ``contribution[X]`` is the *variance* contributed by ``X``
        (so the shares sum to ``sigma_ln_epsff**2``).
    """
    if ln_sigma is None:
        ln_sigma = DEFAULT_LN_SIGMA
    elas = observable_jacobian(fiducial, model=model, multiff=multiff,
                               basis=basis)
    contrib = {x: (elas[x] * ln_sigma.get(x, 0.0)) ** 2 for x in elas}
    var = sum(contrib.values())
    return {'elasticity': elas, 'contribution': contrib,
            'sigma_ln_epsff': np.sqrt(var), 'factor': np.exp(np.sqrt(var))}


def montecarlo_epsff(fiducial, ln_sigma=None, model='km05', multiff=True,
                     basis='density', n=20000, seed=0, observables=None):
    r"""
    Monte-Carlo propagation of observational uncertainties to ``eps_ff``
    (the nonlinear check on the linearized `uncertainty_budget`).

    Draws each observable as a lognormal with the given fractional log-width,
    evaluates ``eps_ff``, and returns the sample.

    Parameters
    ----------
    observables : sequence of str, optional
        Restrict the perturbed observables (e.g. ``['mean_density']`` to
        isolate the density-only spread).  Default: all observables in
        ``basis``.
    seed : int
        Seed for reproducibility.

    Returns
    -------
    ndarray
        ``eps_ff`` samples (length ``n``).
    """
    if ln_sigma is None:
        ln_sigma = DEFAULT_LN_SIGMA
    if observables is None:
        observables = list(OBSERVABLE_EXPONENTS[basis])
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        factors = {x: np.exp(rng.normal(0.0, ln_sigma.get(x, 0.0)))
                   for x in observables}
        out[i] = epsff_from_observables(factors, fiducial, model=model,
                                        multiff=multiff, basis=basis)
    return out
