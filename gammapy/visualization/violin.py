# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import astropy.units as u
from scipy.stats import gaussian_kde, norm


def plot_samples_violin_vs_energy(
    ax,
    energy_edges,
    samples_per_band,
    weights_per_band=None,
    energy_power=None,
    bw_method="scott",
    grid_size=200,
    errorbar_kwargs=None,
    errorbar_ul_kwargs=None,
    violin_kwargs=None,
    violin_clip=None,
    y_label="dN/dE",
):
    """
    Plot flux-sample violin distributions per energy bin on log–log axes.
    Unlike standard error bars which only shows summary statistics,
    a violin plot shows the full probability density for each energy bin.

    This function draws one violin per energy interval, using weighted
    kernel-density estimation (KDE) in log-space. The median and
    1σ-equivalent quantiles are overlaid using Gammapy-style flux-point
    error bars. Users can optionally apply an ``E_center**p`` scaling to the
    flux values and clip violin tails to a specified containment interval.
    Samples of other energy-dependent parameters can also be plotted,
    in that case the `y_label` should be changed accordingly.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Axes object on which to draw the violins and error bars.
    energy_edges : array-like or `~astropy.units.Quantity`
        Monotonically increasing energy bin edges of length ``nbins + 1``.
        Must be positive and finite for log scaling.
    samples_per_band : list of `~astropy.units.Quantity`
        List of sample arrays, one per energy bin. Each array contains
        draws from the flux posterior (or other distribution) in that bin.
    weights_per_band : list of array-like, optional
        Per-sample weights for each bin, same length as ``samples_per_band``.
        If omitted, uniform weights are assumed. Requires NumPy ≥ 2.0 for
        percentile computation with weights.
    energy_power : float, optional
        Power-law scaling applied as ``E_center**energy_power * flux``.
        Default is None.
    bw_method : str or float, optional
        Bandwidth selection passed to ``scipy.stats.gaussian_kde``.
    grid_size : int, optional
        Number of evaluation points in log-flux space for the KDE grid.
    errorbar_kwargs : dict, optional
        Keyword arguments forwarded to ``ax.errorbar`` for the quantile bars.
    errorbar_ul_kwargs : dict, optional
        Keyword arguments forwarded to ``ax.errorbar`` for the ul bars.
    violin_kwargs : dict, optional
        Keyword arguments forwarded to ``ax.fill`` for violin plot.
    violin_clip : (float, float), optional
        Lower and upper containment fractions (in [0, 1]) used to clip
        the violin tails in log-space. If omitted, defaults to
        ``(norm.cdf(-4), norm.cdf(4))``.
    y_label : str, optional
        Base label for the y-axis, before unit and scaling prefixes.

    Returns
    -------
    artists : list
        A list of Matplotlib artists corresponding to the violin polygons
        and error-bar elements added to the axes.

    Notes
    -----
    - KDE is performed on the transformed variable ``log10(E^p * F)``,
      where ``p = energy_power``. The violin polygon is constructed by
      mirroring the normalized log-density around the bin center in
      log-energy.
    - Violin clipping is applied only to the KDE grid, not to the
      quantile-based error bars.
    - Negative flux samples are replaced by a small positive surrogate
      (10% of the smallest positive sample) to allow log-space evaluation.
    - The plotted error bars correspond to the 16%, 50%, and 84%
      weighted percentiles of the *unclipped* samples, unless the median
      is non-positive, in which case an upper limit marker is drawn
      at the 98% percentile. For a gaussian distribution these values corresponds to
      1σ errors and 2σ upper limit plotted by default by the `FluxPoints.plot` method.
    """

    # Axis scaling
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Energy edges
    energy_edges = u.Quantity(energy_edges)
    unit_x = f"{energy_edges.unit}"
    edges = np.asarray(energy_edges.to_value())
    _validate_energy_edges(edges)

    unit_y = f"{samples_per_band[0].unit}"
    nbins = len(edges) - 1

    samples_per_band, weights_per_band = _validate_inputs(
        samples_per_band, weights_per_band, nbins
    )

    # clip defaults
    if violin_clip is None:
        violin_clip = (norm.cdf(-4), norm.cdf(4))
    lc, uc = violin_clip
    if not (0 <= lc < uc <= 1):
        raise ValueError("violin_clip must satisfy 0 <= low < high <= 1.")

    # Errorbar styles
    if errorbar_kwargs is None:
        errorbar_kwargs = dict()
    errorbar_kwargs_defaults = dict(
        marker="o",
        ms=4.5,
        mec="black",
        mfc="white",
        color="black",
        capsize=2.5,
        elinewidth=1.2,
        lw=1.2,
    )
    for key in errorbar_kwargs_defaults.keys():
        errorbar_kwargs.setdefault(key, errorbar_kwargs_defaults[key])

    if errorbar_ul_kwargs is None:
        errorbar_ul_kwargs = dict()
    errorbar_ul_kwargs_defaults = dict(
        marker="v",
        ms=7,
        mec="black",
        mfc="white",
        color="black",
        capsize=2.5,
        elinewidth=1.2,
        lw=1.2,
    )
    for key in errorbar_ul_kwargs_defaults.keys():
        errorbar_ul_kwargs.setdefault(key, errorbar_ul_kwargs_defaults[key])

    if violin_kwargs is None:
        violin_kwargs = dict()
    violin_kwargs_defaults = dict(
        alpha=0.45,
        color="C0",
        edgecolor="black",
        lw=0.8,
    )
    for key in violin_kwargs_defaults.keys():
        violin_kwargs.setdefault(key, violin_kwargs_defaults[key])

    quantiles = [100 * norm.cdf(-1), 50, 100 * norm.cdf(1), 100 * norm.cdf(2)]

    # Precompute bin geometry
    emins = edges[:-1]
    emaxs = edges[1:]
    xlog_min = np.log10(emins)
    xlog_max = np.log10(emaxs)

    centers = 0.5 * (xlog_min + xlog_max)
    halfwidths = 0.5 * (xlog_max - xlog_min) * 0.99

    artists = []

    for samples, weights, xlog_c, hwlog, emin, emax in zip(
        samples_per_band, weights_per_band, centers, halfwidths, emins, emaxs
    ):
        samples = samples.to_value(unit_y)
        s, smin_pos = _sanitize_samples(samples)
        w = _sanitize_weights(weights, len(s))
        if w is None or s.size == 0:
            continue

        # transform
        ylog, scale = _compute_log_transformed(s, xlog_c, energy_power)

        # KDE
        grid_full, dens_full = _kde_evaluate(ylog, w, grid_size, bw_method)

        # clipping interval
        y_high = np.percentile(s, uc * 100, weights=w, method="inverted_cdf") * scale
        y_low = np.percentile(s, lc * 100, weights=w, method="inverted_cdf") * scale
        y_low = max(smin_pos * scale, y_low)

        if y_high > y_low > 0:
            ymin = max(grid_full.min(), np.log10(y_low))
            ymax = min(grid_full.max(), np.log10(y_high))
            mask = (grid_full >= ymin) & (grid_full <= ymax)
            ygrid_log, dens = grid_full[mask], dens_full[mask]
        else:
            ygrid_log, dens = grid_full, dens_full

        # violin polygon
        artists.extend(_draw_violin(ax, xlog_c, hwlog, ygrid_log, dens, violin_kwargs))

        # quantile bars
        artists.append(
            _draw_errorbar(
                ax,
                xlog_c,
                emin,
                emax,
                samples,
                weights,
                quantiles,
                scale,
                errorbar_kwargs,
                errorbar_ul_kwargs,
            )
        )

    # Labels
    ax.set_xlabel(f"Energy [{unit_x}]")

    if energy_power:
        p = f"{energy_power:g}"
        unit_str = rf"[{unit_x}$^{p}$ × {unit_y}]"
        ax.set_ylabel(rf"$E^{p}\,\times$ {y_label} {unit_str}")
    else:
        ax.set_ylabel(f"{y_label} [{unit_y}]")

    return artists


def _validate_inputs(samples_per_band, weights_per_band, nbins):
    """Validate that samples and weightes."""
    if len(samples_per_band) != nbins:
        raise ValueError("samples_per_band must match number of bins.")

    if weights_per_band is None:
        weights_per_band = [None] * nbins
    elif len(weights_per_band) != nbins:
        raise ValueError("weights_per_band must match number of bins.")
    return samples_per_band, weights_per_band


def _validate_energy_edges(edges):
    """Validate that energy_edges is 1D, positive, finite, and strictly increasing."""
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("energy_edges must be 1D and contain at least two values.")
    if (
        not np.all(np.isfinite(edges))
        or np.any(edges <= 0)
        or not np.all(np.diff(edges) > 0)
    ):
        raise ValueError("energy_edges must be strictly increasing, finite, and > 0.")


def _sanitize_samples(samples):
    """Return finite samples with negative values adjusted for log-scale use."""
    s = samples.copy()
    mask = np.isfinite(s)
    s = s[mask]

    s_pos = s[s > 0]
    if np.any(s < 0):
        if s_pos.size > 0:
            s[s < 0] = 0.1 * s_pos.min()

    return s, s_pos.min()


def _sanitize_weights(weights, size):
    """Return normalised finite non-negative weights or None if empty."""
    if weights is None:
        w = np.ones(size, float)
    else:
        w = np.asarray(weights.copy(), float)
        w = w[np.isfinite(w) & (w >= 0)]

    total = w.sum()
    return w / total if total > 0 else None


def _compute_log_transformed(s, xlog_center, energy_power):
    """Transform samples to log10(E^p × F) and return log-values and linear scale."""
    if energy_power:
        shift = energy_power * xlog_center
        scale = (10**xlog_center) ** energy_power
    else:
        shift = 0.0
        scale = 1.0

    return np.log10(s) + shift, scale


def _kde_evaluate(ylog, w, grid_size, bw_method):
    """Evaluate weighted KDE on a fixed log-grid and normalise density."""
    grid = np.linspace(ylog.min(), ylog.max(), int(grid_size))
    kde = gaussian_kde(ylog, weights=w, bw_method=bw_method)
    dens = kde(grid)
    peak = dens.max()
    return grid, dens / peak if peak > 0 else dens


def _draw_violin(ax, x_center, hwlog, ygrid_log, dens, violin_kwargs):
    """Draw a symmetric violin polygon in log-space."""
    xlog_left = x_center - hwlog * dens
    xlog_right = x_center + hwlog * dens

    x_left = 10**xlog_left
    x_right = 10**xlog_right
    y_lin = 10**ygrid_log

    x_poly = np.concatenate([x_left, x_right[::-1]])
    y_poly = np.concatenate([y_lin, y_lin[::-1]])

    return ax.fill(x_poly, y_poly, zorder=2, **violin_kwargs)


def _draw_errorbar(ax, xlog_c, emin, emax, s, w, quantiles, scale, err_kw, err_ul_kw):
    """Draw Gammapy-style flux-point error bars or upper limits."""
    y_qs = [
        np.percentile(s, q, weights=w, method="inverted_cdf") * scale for q in quantiles
    ]

    y_lo, y_med, y_hi, y_ul = y_qs[0], y_qs[1], y_qs[2], y_qs[3]
    if y_med > 0:
        yerr = np.array([[max(0.0, y_med - y_lo)], [max(0.0, y_hi - y_med)]])
        kwargs = err_kw
    else:
        yerr = None
        y_med = y_ul
        kwargs = err_ul_kw
    x_c = 10 ** (xlog_c)
    xerr = np.array([[x_c - emin], [emax - x_c]])
    return ax.errorbar(x_c, y_med, xerr=xerr, yerr=yerr, **(kwargs or {}), zorder=5)
