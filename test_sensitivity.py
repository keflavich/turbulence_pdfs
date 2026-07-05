"""
Regression + property tests for sensitivity.py.

Numerical targets are verified against scipy.integrate.quad to ~1e-9 relative
(the analytic erfc form and the quadrature agree to full float precision).

Note: a few absolute targets differ from the original spec draft in the 4th-5th
decimal.  The values here are the ones for which ``erfc``-form == ``quad`` to
15 digits at the stated fiducials (s_c=4, M=10, beta=inf); the spec's
mass-fraction draft values were slightly imprecise.  The *b-swing ratios*
(5.30 single-ff, 7.98 multi-ff) match the spec exactly.
"""
import warnings

import numpy as np
import pytest
from scipy.integrate import quad

import sensitivity as sv


# fiducials
SC = 4.0
MACH = 10.0
S1_SQ = 2.4941233048929243   # sigma_s(1/3, 10)**2
S2_SQ = 4.61512051684126     # sigma_s(1.0, 10)**2
S1 = np.sqrt(S1_SQ)
S2 = np.sqrt(S2_SQ)


def _quad_tail(s_c, sigma, lam):
    """Reference: direct quadrature of exp(lam s) * N(s; -sig^2/2, sig^2)."""
    mu = -sigma ** 2 / 2.0

    def integrand(s):
        return np.exp(lam * s) * np.exp(-(s - mu) ** 2 / (2 * sigma ** 2)) \
            / np.sqrt(2 * np.pi * sigma ** 2)

    return quad(integrand, s_c, 80.0, limit=400)[0]


# ---------------------------------------------------------------------------
# variance-Mach relation
# ---------------------------------------------------------------------------
def test_sigma_sq_values():
    assert sv.sigma_s_sq(1 / 3, 10) == pytest.approx(2.49412, abs=1e-5)
    assert sv.sigma_s_sq(1.0, 10) == pytest.approx(4.61512, abs=1e-5)


def test_sigma_hydro_limit_matches_finite_beta():
    # beta -> large recovers the hydro form
    assert sv.sigma_s_sq(0.5, 8, beta=1e12) == pytest.approx(
        sv.sigma_s_sq(0.5, 8, beta=np.inf), rel=1e-9)


# ---------------------------------------------------------------------------
# master integral vs quad
# ---------------------------------------------------------------------------
def test_massfrac_regression():
    assert sv.massfrac_above(SC, S1) == pytest.approx(0.0406527, abs=1e-6)
    assert sv.massfrac_above(SC, S2) == pytest.approx(0.2154038, abs=1e-6)


def test_multiff_regression():
    assert sv.tail_moment(SC, S1, lam=1.5) == pytest.approx(0.4335698, abs=1e-6)
    assert sv.tail_moment(SC, S2, lam=1.5) == pytest.approx(3.4582766, abs=1e-6)


def test_b_swing_ratios():
    single = sv.massfrac_above(SC, S2) / sv.massfrac_above(SC, S1)
    multi = (sv.tail_moment(SC, S2, lam=1.5)
             / sv.tail_moment(SC, S1, lam=1.5))
    assert single == pytest.approx(5.30, abs=0.01)
    assert multi == pytest.approx(7.98, abs=0.01)


@pytest.mark.parametrize("lam", [1.0, 1.5])
@pytest.mark.parametrize("s_c", [1.0, 2.5, 4.0, 6.0, 8.0])
@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0, 3.0])
def test_tail_moment_vs_quad_grid(s_c, sigma, lam):
    got = sv.tail_moment(s_c, sigma, lam=lam, warn=False)
    ref = _quad_tail(s_c, sigma, lam)
    assert got == pytest.approx(ref, rel=1e-8)


def test_multiff_can_exceed_one_and_warns():
    # high sigma, low s_c -> multi-ff integral > 1
    with pytest.warns(UserWarning):
        val = sv.tail_moment(1.0, 3.0, lam=1.5)
    assert val > 1.0


def test_epsff_warns_when_above_one():
    # eps_core/phi_t=1, low s_crit, high sigma -> eps_ff > 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning):
            sv.epsff(1.0, 50, alpha_vir=0.1, multiff=True,
                     eps_core=1.0, phi_t=1.0, s_crit=1.0)


# ---------------------------------------------------------------------------
# elasticity
# ---------------------------------------------------------------------------
def test_elasticity_exact_single_ff():
    assert sv.elasticity_sigma(SC, S1, multiff=False, method='exact') == \
        pytest.approx(7.14, abs=0.02)


def test_elasticity_exact_multi_ff():
    assert sv.elasticity_sigma(SC, S1, multiff=True, method='exact') == \
        pytest.approx(7.99, abs=0.02)


def test_elasticity_asymptote():
    assert sv.elasticity_sigma(SC, S1, method='asymptotic') == \
        pytest.approx(5.79, abs=0.01)


def test_exact_exceeds_asymptote():
    exact = sv.elasticity_sigma(SC, S1, multiff=False, method='exact')
    asymp = sv.elasticity_sigma(SC, S1, method='asymptotic')
    assert exact > asymp  # positive O(1) prefactor contribution


@pytest.mark.parametrize("multiff", [False, True])
@pytest.mark.parametrize("s_c,sigma", [(4.0, S1), (4.0, S2), (6.0, 2.0),
                                       (2.0, 1.0)])
def test_elasticity_exact_matches_fd(s_c, sigma, multiff):
    exact = sv.elasticity_sigma(s_c, sigma, multiff=multiff, method='exact')
    fd = sv.elasticity_sigma(s_c, sigma, multiff=multiff, method='fd')
    assert exact == pytest.approx(fd, rel=1e-4)


# ---------------------------------------------------------------------------
# chain rule to b
# ---------------------------------------------------------------------------
def test_dlnsigma_dlnb_matches_fd():
    b0 = 0.5
    ana = sv.dlnsigma_dlnb(b0, MACH)
    h = 1e-6
    fp = np.log(sv.sigma_s(b0 * (1 + h), MACH))
    fm = np.log(sv.sigma_s(b0 * (1 - h), MACH))
    fd = (fp - fm) / (2 * h)
    assert ana == pytest.approx(fd, rel=1e-5)


def test_elasticity_b_matches_fd():
    b0, av = 0.5, 1.0
    ana = sv.elasticity_b(b0, MACH, alpha_vir=av, multiff=True, model='km05')
    h = 1e-6
    fp = np.log(sv.epsff(b0 * (1 + h), MACH, av, multiff=True, warn=False))
    fm = np.log(sv.epsff(b0 * (1 - h), MACH, av, multiff=True, warn=False))
    fd = (fp - fm) / (2 * h)
    assert ana == pytest.approx(fd, rel=1e-4)


# ---------------------------------------------------------------------------
# jacobian structure
# ---------------------------------------------------------------------------
def test_jacobian_b_dominates_mach():
    fid = dict(b=0.5, mach=10.0, alpha_vir=1.0, phi_x=1.12)
    jac = sv.jacobian(fid, model='km05', multiff=True)
    # b enters only sigma^2; M enters both sigma^2 and s_crit (partial cancel)
    assert abs(jac['b']) > abs(jac['mach'])


def test_jacobian_beta_infinite_is_zero():
    fid = dict(b=0.5, mach=10.0, alpha_vir=1.0, beta=np.inf, phi_x=1.12)
    jac = sv.jacobian(fid, model='km05', multiff=True, params=('beta',))
    assert jac['beta'] == 0.0


def test_jacobian_matches_elasticity_b():
    fid = dict(b=0.5, mach=10.0, alpha_vir=1.0, phi_x=1.12)
    jac = sv.jacobian(fid, model='km05', multiff=True)
    ana = sv.elasticity_b(0.5, 10.0, alpha_vir=1.0, multiff=True, model='km05')
    assert jac['b'] == pytest.approx(ana, rel=1e-3)


# ---------------------------------------------------------------------------
# s_crit models (FK12 Table 1, verified against arXiv:1209.2856)
# ---------------------------------------------------------------------------
def test_scrit_km05_closed_form():
    # ln((pi^2/5) phi_x^2 alpha_vir M^2) at hydro limit
    expect = np.log((np.pi ** 2 / 5) * 1.12 ** 2 * 1.0 * 100.0)
    assert sv.scrit_km05(1.0, 10.0) == pytest.approx(expect)


def test_scrit_pn11_closed_form():
    expect = np.log(0.067 * 0.35 ** -2 * 1.0 * 100.0)
    assert sv.scrit_pn11(1.0, 10.0) == pytest.approx(expect)


def test_scrit_hc13_is_two_term_sum():
    # thermal (M^-2) + turbulent term inside the log (FK12 Eq. 38+39)
    thermal = (np.pi ** 2 / 5) * 0.1 ** -2 * 1.0 * 10.0 ** -2 * 1.0
    turb = (np.pi ** 2 / 15) * 0.1 ** -1 * 1.0
    assert sv.scrit_hc13(1.0, 10.0) == pytest.approx(np.log(thermal + turb))


def test_scrit_hydro_limits_of_fbeta():
    # all magnetic factors -> 1 as beta -> inf
    assert sv._fbeta_km(np.inf) == pytest.approx(1.0)
    assert sv._fbeta_pn(np.inf) == pytest.approx(1.0)
    # PN Eq.31 differs from KM at finite beta
    assert sv._fbeta_pn(1.0) != pytest.approx(sv._fbeta_km(1.0))


def test_scrit_finite_beta_lowers_km_scrit():
    # magnetic support reduces the KM critical density
    assert sv.scrit_km05(1.0, 10.0, beta=1.0) < sv.scrit_km05(1.0, 10.0)


# ---------------------------------------------------------------------------
# envelope
# ---------------------------------------------------------------------------
def test_envelope_ordering_and_width():
    mach = np.array([2.0, 10.0, 50.0])
    lo, hi = sv.envelope_epsff(mach, alpha_vir=1.0, model='km05', multiff=True)
    assert np.all(hi >= lo)
    # the envelope should span more than a factor of a few at high Mach
    assert np.any(hi / lo > 3.0)


# ---------------------------------------------------------------------------
# log converters round-trip
# ---------------------------------------------------------------------------
def test_log_converters_roundtrip():
    s = np.array([-1.0, 0.0, 2.5])
    assert np.allclose(sv.ln_from_log10(sv.log10_from_ln(s)), s)
    sig = 1.7
    assert sv.sigma_ln_from_log10(sv.sigma_log10_from_ln(sig)) == \
        pytest.approx(sig)
