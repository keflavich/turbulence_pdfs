"""
Figures for the exponential parameter-sensitivity analysis (see sensitivity.py).

One function per figure; each saves a PNG (and PDF) to ``figures/``.  Running
this module as a script regenerates all figures.

Figures
-------
1. `fig_envelope`   -- the "money figure": eps_ff(M) band spanning the driving
   parameter b = 1/3 -> 1, vs the observed eps_ff = 0.3-3% band.  The theory
   band contains the data band => agreement cannot confirm the theory.
2. `fig_elasticity_heatmap` -- d ln f / d ln sigma over the (s_c, sigma) plane,
   with the three s_crit model loci overplotted.
3. `fig_tornado`    -- |d ln eps_ff / d ln x| per parameter, per s_crit model,
   showing b dominating and the M partial-cancellation.
4. `fig_envelope_hopkins` (stretch) -- figure 1 recomputed with the repo's
   Hopkins (2013) intermittency PDF, to show the conclusion is not
   lognormal-specific.
"""
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sensitivity as sv

FIGDIR = 'figures'

# observed SFE-per-free-fall-time band (Lee+2016 scatter; Utomo+2018 median ~0.7%)
OBS_LO, OBS_HI, OBS_MED = 0.003, 0.03, 0.007

# consistent, colorblind-safe model colors
MODEL_COLORS = {'km05': '#0072B2', 'pn11': '#D55E00', 'hc13': '#009E73'}
MODEL_LABELS = {'km05': 'KM05', 'pn11': 'PN11', 'hc13': 'HC13'}


def _ensure_figdir():
    if not os.path.isdir(FIGDIR):
        os.makedirs(FIGDIR)


def _save(fig, name):
    _ensure_figdir()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIGDIR, '{}.{}'.format(name, ext)),
                    bbox_inches='tight', dpi=150)
    return os.path.join(FIGDIR, name + '.png')


# ---------------------------------------------------------------------------
# Figure 1: the envelope ("money") figure
# ---------------------------------------------------------------------------
def fig_envelope(model='km05', multiff=True, alpha_virs=(1.0, 2.0, 5.0),
                 mach=None, eps_core=0.5, phi_t=1.0):
    """
    eps_ff vs Mach with a band spanning b = 1/3 (solenoidal) -> 1 (compressive)
    at each alpha_vir, against the observed eps_ff band.

    The band's vertical extent is the theory's parameter envelope from the
    single least-constrained free parameter (b).  It brackets the observed band
    -- so a measured eps_ff cannot discriminate the theory.
    """
    if mach is None:
        mach = np.logspace(np.log10(2), np.log10(50), 200)

    linestyles = ['-', '--', ':']
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # observed band
    ax.axhspan(OBS_LO, OBS_HI, color='0.6', alpha=0.35, zorder=0)
    ax.axhline(OBS_MED, color='0.35', lw=1.2, ls='-', zorder=1)
    ax.text(2.2, OBS_MED * 1.15, 'observed (Lee+16 scatter; Utomo+18 median)',
            fontsize=9, color='0.25', va='bottom')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # eps_ff>1 handled by the plot itself
        for av, ls in zip(alpha_virs, linestyles):
            lo, hi = sv.envelope_epsff(mach, alpha_vir=av, model=model,
                                       multiff=multiff, eps_core=eps_core,
                                       phi_t=phi_t)
            ax.fill_between(mach, lo, hi, color=MODEL_COLORS[model],
                            alpha=0.18, zorder=2)
            ax.plot(mach, hi, color=MODEL_COLORS[model], lw=1.8, ls=ls,
                    zorder=3, label=r'$\alpha_{{\rm vir}}={:g}$'.format(av))
            ax.plot(mach, lo, color=MODEL_COLORS[model], lw=1.8, ls=ls,
                    zorder=3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(mach.min(), mach.max())
    ax.set_ylim(1e-3, 3.0)
    ax.set_xlabel(r'sonic Mach number $\mathcal{M}$')
    ax.set_ylabel(r'$\epsilon_{\rm ff}$ (SFE per free-fall time)')
    ax.set_title('{} {}: driving-parameter envelope ($b=1/3 \\to 1$) '
                 'vs data'.format(MODEL_LABELS[model],
                                  'multi-ff' if multiff else 'single-ff'))
    ax.legend(loc='upper left', frameon=False, fontsize=10,
              title='band = solenoidal$\\to$compressive')
    ax.text(0.98, 0.02,
            'theory band $\\supset$ data band  $\\Rightarrow$  unconfirmable',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', fc='white', ec='0.7'))
    return _save(fig, 'sensitivity_envelope_{}'.format(model))


# ---------------------------------------------------------------------------
# Figure 2: elasticity heatmap over (s_c, sigma)
# ---------------------------------------------------------------------------
def fig_elasticity_heatmap(multiff=False):
    """
    d ln f / d ln sigma over the (s_c, sigma) plane, with the fiducial loci of
    the three s_crit models overplotted and contours at elasticity = 5, 10.
    """
    s_c = np.linspace(0.5, 10, 300)
    sigma = np.linspace(0.5, 3.5, 300)
    SC, SIG = np.meshgrid(s_c, sigma)
    E = sv.elasticity_sigma(SC, SIG, multiff=multiff, method='exact')

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.pcolormesh(SC, SIG, E, cmap='magma', shading='auto',
                       vmin=0, vmax=30)
    cs = ax.contour(SC, SIG, E, levels=[5, 10], colors='white',
                    linewidths=1.3)
    ax.clabel(cs, fmt=r'$%d$', fontsize=9)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r'$d\ln f / d\ln\sigma$  (logarithmic sensitivity)')

    # model fiducial loci: sigma(b, M) and s_crit(model) traced over Mach,
    # at b = 0.4 (mixed driving), alpha_vir = 1
    mach = np.linspace(2, 50, 100)
    for model in ('km05', 'pn11', 'hc13'):
        sig_line = sv.sigma_s(0.4, mach)
        sc_line = sv.SCRIT_MODELS[model](1.0, mach)
        keep = (sc_line > s_c.min()) & (sc_line < s_c.max()) & \
               (sig_line > sigma.min()) & (sig_line < sigma.max())
        ax.plot(sc_line[keep], sig_line[keep], color=MODEL_COLORS[model],
                lw=2.4, label=MODEL_LABELS[model])

    ax.set_xlim(s_c.min(), s_c.max())
    ax.set_ylim(sigma.min(), sigma.max())
    ax.set_xlabel(r'critical log-density $s_{\rm crit}$')
    ax.set_ylabel(r'PDF width $\sigma_s$')
    ax.set_title('Sensitivity of the mass fraction to $\\sigma_s$\n'
                 '(model loci: $b=0.4,\\ \\alpha_{\\rm vir}=1,\\ '
                 '\\mathcal{M}=2\\to50$)')
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    return _save(fig, 'sensitivity_elasticity_heatmap')


# ---------------------------------------------------------------------------
# Figure 3: tornado / Jacobian bar chart
# ---------------------------------------------------------------------------
def fig_tornado(fiducials=None, multiff=True):
    """
    |d ln eps_ff / d ln x| for x in {b, M, alpha_vir, beta, phi_x} at the
    fiducial point, grouped by s_crit model.  b dominates; M is suppressed by
    the numerator/denominator partial cancellation.
    """
    if fiducials is None:
        fiducials = dict(b=0.4, mach=10.0, alpha_vir=1.0, beta=10.0,
                         phi_x=1.12, theta=0.35, y_cut=0.1)
    params = ('b', 'mach', 'alpha_vir', 'beta', 'phi_x')
    plabels = {'b': r'$b$', 'mach': r'$\mathcal{M}$',
               'alpha_vir': r'$\alpha_{\rm vir}$', 'beta': r'$\beta$',
               'phi_x': r'$\phi_x$'}
    models = ('km05', 'pn11', 'hc13')

    jacs = {m: sv.jacobian(fiducials, model=m, multiff=multiff, params=params)
            for m in models}

    y = np.arange(len(params))
    h = 0.25
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for i, m in enumerate(models):
        vals = [abs(jacs[m].get(p, 0.0)) for p in params]
        ax.barh(y + (i - 1) * h, vals, height=h, color=MODEL_COLORS[m],
                label=MODEL_LABELS[m])

    ax.set_yticks(y)
    ax.set_yticklabels([plabels[p] for p in params])
    ax.invert_yaxis()
    ax.set_xlabel(r'$|\,d\ln\epsilon_{\rm ff} / d\ln x\,|$  (elasticity)')
    ax.set_title('Parameter leverage at fiducial '
                 '($b=0.4,\\ \\mathcal{M}=10,\\ \\alpha_{\\rm vir}=1,\\ '
                 '\\beta=10$)')
    ax.legend(frameon=False, fontsize=10)
    ax.text(0.98, 0.05,
            r'$b$ enters only $\sigma_s^2$; $\mathcal{M}$ enters both '
            r'$\sigma_s^2$ and $s_{\rm crit}$ (partial cancellation)',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', ec='0.7'))
    return _save(fig, 'sensitivity_tornado')


# ---------------------------------------------------------------------------
# Figure 4 (stretch): the same envelope with the Hopkins (2013) PDF
# ---------------------------------------------------------------------------
def _hopkins_tail_moment(s_c, sigma, lam, n=4000):
    """
    exp(lam s)-weighted tail integral above s_c using the Hopkins (2013)
    intermittency PDF instead of the lognormal, with T = T_of_sigma(sigma).

    Numerically integrates the normalized volume-weighted PDF in s = ln(rho).
    """
    import hopkins_pdf as hp
    T = hp.T_of_sigma(sigma, logform=True)
    s = np.linspace(-6 * sigma - 2, s_c + 12 * sigma + 20, n)
    rho = np.exp(s)
    logp = hp.loghopkins(rho, sigma, T, meanrho=1.0)
    pv = np.exp(logp)
    pv[~np.isfinite(pv)] = 0.0
    norm = np.trapz(pv, s)
    integ = np.exp(lam * s) * pv
    above = s >= s_c
    return np.trapz(integ[above], s[above]) / norm


def fig_envelope_hopkins(model='km05', alpha_vir=1.0, mach=None,
                         eps_core=0.5, phi_t=1.0):
    """
    Stretch figure: repeat `fig_envelope` (single alpha_vir) with the Hopkins
    (2013) PDF, showing the parameter-envelope conclusion is not specific to
    the lognormal.
    """
    if mach is None:
        mach = np.logspace(np.log10(2), np.log10(50), 40)

    lam = 1.5  # multi-free-fall
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.axhspan(OBS_LO, OBS_HI, color='0.6', alpha=0.35, zorder=0)
    ax.axhline(OBS_MED, color='0.35', lw=1.2, zorder=1)

    for pdf_name, tail_fn, color in (
            ('lognormal', lambda sc, sg: sv.tail_moment(sc, sg, lam=lam,
                                                         warn=False),
             MODEL_COLORS['km05']),
            ('Hopkins 2013', _hopkins_tail_moment_wrap, MODEL_COLORS['pn11'])):
        lo, hi = [], []
        for M in mach:
            sc = sv.SCRIT_MODELS[model](alpha_vir, M)
            vals = [(eps_core / phi_t) * tail_fn(sc, sv.sigma_s(b, M))
                    for b in (1.0 / 3, 1.0)]
            lo.append(min(vals))
            hi.append(max(vals))
        lo, hi = np.array(lo), np.array(hi)
        ax.fill_between(mach, lo, hi, color=color, alpha=0.20, zorder=2)
        ax.plot(mach, hi, color=color, lw=1.8, zorder=3,
                label='{} band'.format(pdf_name))
        ax.plot(mach, lo, color=color, lw=1.8, zorder=3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(mach.min(), mach.max())
    ax.set_ylim(1e-3, 3.0)
    ax.set_xlabel(r'sonic Mach number $\mathcal{M}$')
    ax.set_ylabel(r'$\epsilon_{\rm ff}$')
    ax.set_title('{} envelope ($b=1/3\\to1$): lognormal vs Hopkins PDF'
                 .format(MODEL_LABELS[model]))
    ax.legend(loc='upper left', frameon=False, fontsize=10)
    return _save(fig, 'sensitivity_envelope_hopkins')


def _hopkins_tail_moment_wrap(sc, sg):
    return _hopkins_tail_moment(sc, sg, lam=1.5)


# ---------------------------------------------------------------------------
# Observable elasticities & error propagation (velocity vs mean density)
# ---------------------------------------------------------------------------
OBS_LABELS = {'mean_density': r'mean density $\rho_0$', 'radius': r'radius $R$',
              'sigma_v': r'velocity disp. $\sigma_v$', 'c_s': r'sound speed $c_s$',
              'b': r'driving $b$', 'mass': r'mass $M$', 'volume': r'volume $V$'}


def fig_observable_tornado(fiducial=None, multiff=True, basis='density'):
    """
    d ln eps_ff / d ln X for each physical observable X, per s_crit model.

    Shows the structural asymmetry: the mean density (mass) enters only the
    threshold -> undamped; the velocity dispersion enters both the width and
    the threshold -> cancels -> small.
    """
    if fiducial is None:
        fiducial = dict(b=0.4, mach=10.0, alpha_vir=1.0)
    order = [k for k in OBSERVABLE_ORDER(basis)]
    models = ('km05', 'pn11', 'hc13')
    jacs = {m: sv.observable_jacobian(fiducial, model=m, multiff=multiff,
                                      basis=basis) for m in models}

    y = np.arange(len(order))
    h = 0.25
    fig, ax = plt.subplots(figsize=(8, 5.0))
    for i, m in enumerate(models):
        vals = [jacs[m][o] for o in order]
        ax.barh(y + (i - 1) * h, vals, height=h, color=MODEL_COLORS[m],
                label=MODEL_LABELS[m])
    ax.axvline(0, color='0.4', lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([OBS_LABELS[o] for o in order])
    ax.invert_yaxis()
    ax.set_xlabel(r'$d\ln\epsilon_{\rm ff} / d\ln X$  (signed elasticity)')
    ax.set_title('Sensitivity of $\\epsilon_{\\rm ff}$ to physical observables '
                 '(fiducial $b=0.4,\\ \\mathcal{M}=10,\\ \\alpha_{\\rm vir}=1$)')
    ax.legend(frameon=False, fontsize=10, loc='lower right')
    ax.text(0.02, 0.03,
            r'$\rho_0$ (mass/vol.) enters only $s_{\rm crit}$ $\Rightarrow$ '
            r'undamped;  $\sigma_v$ enters width $+$ threshold $\Rightarrow$ '
            r'cancels', transform=ax.transAxes, ha='left', va='bottom',
            fontsize=9, bbox=dict(boxstyle='round', fc='white', ec='0.7'))
    return _save(fig, 'sensitivity_observable_tornado')


def fig_uncertainty_budget(fiducial=None, ln_sigma=None, multiff=True,
                           basis='density'):
    """
    Variance contribution of each observable to sigma(ln eps_ff), per model,
    with realistic input uncertainties.  Mean density + size dominate the
    *observational* budget; velocity is negligible.
    """
    if fiducial is None:
        fiducial = dict(b=0.4, mach=10.0, alpha_vir=1.0)
    order = [k for k in OBSERVABLE_ORDER(basis)]
    models = ('km05', 'pn11', 'hc13')
    buds = {m: sv.uncertainty_budget(fiducial, ln_sigma=ln_sigma, model=m,
                                     multiff=multiff, basis=basis)
            for m in models}

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(8, 5.0))
    # stacked bars of sqrt(variance-contribution) is not additive; stack variance
    bottom = np.zeros(len(models))
    cmap = plt.get_cmap('tab10')
    for j, o in enumerate(order):
        vals = np.array([buds[m]['contribution'][o] for m in models])
        ax.bar(x, vals, bottom=bottom, label=OBS_LABELS[o],
               color=cmap(j % 10))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylabel(r'variance contribution to $\sigma^2_{\ln\epsilon_{\rm ff}}$')
    tot = [buds[m]['sigma_ln_epsff'] for m in models]
    for xi, m in zip(x, models):
        ax.text(xi, bottom[list(models).index(m)] * 1.01,
                r'$\sigma_{{\ln\epsilon}}={:.2f}$'.format(
                    buds[m]['sigma_ln_epsff']), ha='center', va='bottom',
                fontsize=9)
    ax.set_title('$\\epsilon_{\\rm ff}$ uncertainty budget '
                 '(realistic observational errors)')
    ax.legend(frameon=False, fontsize=9, ncol=2, loc='upper right')
    ax.margins(y=0.15)
    return _save(fig, 'sensitivity_uncertainty_budget')


def fig_elasticity_vs_regime(fiducial=None, multiff=True, model='km05'):
    """
    |d ln eps_ff / d ln X| vs eps_ff, sweeping alpha_vir to move deeper into the
    tail.  The density/size sensitivities GROW toward the observed low-eps_ff
    regime, while the velocity sensitivity stays small -- so exactly where the
    theory is "tested", a tiny mass/volume error propagates most.
    """
    if fiducial is None:
        fiducial = dict(b=0.4, mach=10.0, alpha_vir=1.0)
    alpha_grid = np.logspace(-0.5, 1.6, 60)
    eps = np.array([sv.epsff(fiducial['b'], fiducial['mach'], av, model=model,
                             multiff=multiff, warn=False) for av in alpha_grid])
    keys = ['mean_density', 'radius', 'sigma_v', 'c_s']
    E = {k: [] for k in keys}
    for av in alpha_grid:
        j = sv.observable_jacobian(dict(fiducial, alpha_vir=av), model=model,
                                   multiff=multiff, basis='density')
        for k in keys:
            E[k].append(abs(j[k]))

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    colors = {'mean_density': '#0072B2', 'radius': '#56B4E9',
              'sigma_v': '#D55E00', 'c_s': '#E69F00'}
    for k in keys:
        ax.plot(eps, E[k], color=colors[k], lw=2.2, label=OBS_LABELS[k])
    ax.axvspan(OBS_LO, OBS_HI, color='0.6', alpha=0.3, zorder=0)
    ax.text(np.sqrt(OBS_LO * OBS_HI), ax.get_ylim()[1] * 0.9, 'observed',
            ha='center', fontsize=9, color='0.3')
    ax.set_xscale('log')
    ax.invert_xaxis()  # deep tail (low eps_ff) to the right
    ax.set_xlabel(r'$\epsilon_{\rm ff}$  (deeper tail $\rightarrow$)')
    ax.set_ylabel(r'$|\,d\ln\epsilon_{\rm ff} / d\ln X\,|$')
    ax.set_title('{}: observable sensitivities grow toward the observed '
                 'regime'.format(MODEL_LABELS[model]))
    ax.legend(frameon=False, fontsize=10, loc='upper left')
    return _save(fig, 'sensitivity_elasticity_vs_regime')


def fig_montecarlo_density_vs_velocity(fiducial=None, multiff=True,
                                       model='km05', n=40000):
    """
    Monte-Carlo eps_ff distributions when ONLY the mean density is uncertain
    vs ONLY the velocity dispersion (same realistic fractional errors), plus
    the all-observables spread.  The density-only spread dwarfs the
    velocity-only spread -- the money figure for the observational point.
    """
    if fiducial is None:
        fiducial = dict(b=0.4, mach=10.0, alpha_vir=1.0)
    mc_rho = sv.montecarlo_epsff(fiducial, model=model, multiff=multiff,
                                 basis='density', observables=['mean_density'],
                                 n=n, seed=1)
    mc_v = sv.montecarlo_epsff(fiducial, model=model, multiff=multiff,
                               basis='density', observables=['sigma_v'],
                               n=n, seed=2)
    mc_all = sv.montecarlo_epsff(fiducial, model=model, multiff=multiff,
                                 basis='density', n=n, seed=3)
    eps0 = sv.epsff(fiducial['b'], fiducial['mach'], fiducial['alpha_vir'],
                    model=model, multiff=multiff, warn=False)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    bins = np.linspace(-4, 0.5, 80)
    for data, color, lab, sd in (
            (mc_v, '#D55E00', r'$\sigma_v$ only ($\sigma_{\ln}=0.1$)', np.std(np.log10(mc_v))),
            (mc_rho, '#0072B2', r'$\rho_0$ only ($\sigma_{\ln}=0.5$)', np.std(np.log10(mc_rho))),
            (mc_all, '0.4', 'all observables', np.std(np.log10(mc_all)))):
        ax.hist(np.log10(data), bins=bins, histtype='stepfilled', alpha=0.45,
                color=color, density=True,
                label='{}  ($\\Delta={:.2f}$ dex)'.format(lab, sd))
    ax.axvline(np.log10(eps0), color='k', lw=1.0, ls='--')
    ax.axvspan(np.log10(OBS_LO), np.log10(OBS_HI), color='green', alpha=0.12,
               zorder=0)
    ax.set_xlabel(r'$\log_{10}\,\epsilon_{\rm ff}$')
    ax.set_ylabel('probability density')
    ax.set_title('{}: a small mean-density (mass/volume) error dominates the '
                 '$\\epsilon_{{\\rm ff}}$ spread'.format(MODEL_LABELS[model]))
    ax.legend(frameon=False, fontsize=9.5, loc='upper left')
    return _save(fig, 'sensitivity_montecarlo_density_vs_velocity')


def OBSERVABLE_ORDER(basis):
    """Display order of observables for a basis (b last: it's a theory param)."""
    if basis == 'mass_volume':
        return ['mass', 'volume', 'sigma_v', 'c_s', 'b']
    return ['mean_density', 'radius', 'sigma_v', 'c_s', 'b']


def make_all():
    """Regenerate every figure; returns the list of saved PNG paths."""
    paths = [
        fig_envelope(model='km05', multiff=True),
        fig_elasticity_heatmap(multiff=False),
        fig_tornado(multiff=True),
        fig_envelope_hopkins(model='km05'),
        fig_observable_tornado(),
        fig_uncertainty_budget(),
        fig_elasticity_vs_regime(),
        fig_montecarlo_density_vs_velocity(),
    ]
    return paths


if __name__ == '__main__':
    for p in make_all():
        print('wrote', p)
