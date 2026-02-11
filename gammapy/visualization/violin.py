# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import astropy.units as u
from scipy.stats import gaussian_kde, norm


def plot_flux_violin(
    ax,
    energy_edges,
    samples_per_band,
    weights_per_band=None,
    energy_power=0.0,
    color="C0",
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

    This function draws one violin per energy interval, using weighted
    kernel-density estimation (KDE) in log-space. The median and
    1σ-equivalent quantiles are overlaid using Gammapy-style flux-point
    error bars. Users can optionally apply an ``E_center**p`` scaling to the
    flux values and clip violin tails to a specified containment interval.

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
        Useful for plotting SED-like quantities.
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
      (corresponding to the 2σ one-sided percentile).
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
        errorbar_kwargs = dict(
            marker="o",
            ms=4.5,
            mec="black",
            mfc="white",
            color="black",
            capsize=2.5,
            elinewidth=1.2,
            lw=1.2,
        )
    if errorbar_ul_kwargs is None:
        errorbar_ul_kwargs = dict(
            marker="v",
            ms=7,
            mec="black",
            mfc="white",
            color="black",
            capsize=2.5,
            elinewidth=1.2,
            lw=1.2,
        )
    if violin_kwargs is None:
        violin_kwargs = dict(
            alpha=0.45,
            color="C0",
            edgecolor="black",
            lw=0.8,
        )

    quantiles = [100 * norm.cdf(-1), 50, 100 * norm.cdf(1)]
    ul_level = 100 * norm.cdf(2)

    # Precompute bin geometry
    emins = edges[:-1]
    emaxs = edges[1:]
    xlog_min = np.log10(emins)
    xlog_max = np.log10(emaxs)

    centers = 0.5 * (xlog_min + xlog_max)
    halfwidths = 0.5 * (xlog_max - xlog_min) * 0.99

    artists = []

    for samples, weights, xc, hwlog, emin, emax in zip(
        samples_per_band, weights_per_band, centers, halfwidths, emins, emaxs
    ):
        s = _sanitize_samples(samples, unit_y)
        w = _sanitize_weights(weights, len(s))
        if w is None or s.size == 0:
            continue

        # transform
        ylog, scale = _compute_log_transformed(s, xc, energy_power)

        # KDE
        grid_full, dens_full = _kde_evaluate(ylog, w, grid_size, bw_method)

        # clipping interval
        y_low = np.percentile(s, lc * 100, weights=w, method="inverted_cdf") * scale
        y_high = np.percentile(s, uc * 100, weights=w, method="inverted_cdf") * scale

        if y_high > y_low > 0:
            ymin = max(grid_full.min(), np.log10(y_low))
            ymax = min(grid_full.max(), np.log10(y_high))
            ygrid_log, dens = _clip_violin(grid_full, dens_full, ymin, ymax)
        else:
            ygrid_log, dens = grid_full, dens_full

        # violin polygon
        artists.extend(_draw_violin(ax, xc, hwlog, ygrid_log, dens, violin_kwargs))

        # quantile bars
        y_lo, y_med, y_hi = _quantiles(s, w, quantiles, scale)
        if y_med <= 0:
            y_med = np.percentile(s, ul_level, weights=w, method="inverted_cdf") * scale

        x_center_lin = 10**xc

        artists.append(
            _draw_errorbar(
                ax,
                x_center_lin,
                emin,
                emax,
                y_med,
                y_lo,
                y_hi,
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


def _sanitize_samples(samples, unit_y):
    """Return finite samples with negative values adjusted for log-scale use."""
    s = samples.copy().to_value(unit_y)
    mask = np.isfinite(s)
    s = s[mask]

    if np.any(s < 0):
        s_pos = s[s > 0]
        if s_pos.size > 0:
            s[s < 0] = 0.1 * s_pos.min()

    return s


def _sanitize_weights(weights, size):
    """Return normalised finite non-negative weights or None if empty."""
    if weights is None:
        w = np.ones(size, float)
    else:
        w = np.asarray(weights, float)
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


def _clip_violin(grid, dens, ymin, ymax):
    """Clip KDE grid and density to a specified log-range."""
    mask = (grid >= ymin) & (grid <= ymax)
    return grid[mask], dens[mask]


def _quantiles(samples, weights, qlist, scale):
    """Compute weighted percentiles scaled by E^p if needed."""
    return [
        np.percentile(samples, q, weights=weights, method="inverted_cdf") * scale
        for q in qlist
    ]


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


def _draw_errorbar(ax, x_center_lin, emin, emax, y_med, y_lo, y_hi, err_kw, err_ul_kw):
    """Draw Gammapy-style flux-point error bars or upper limits."""
    if y_med > 0:
        yerr = np.array([[max(0, y_med - y_lo)], [max(0, y_hi - y_med)]])
        style = err_kw
    else:
        yerr = None
        style = err_ul_kw

    xerr = np.array([[x_center_lin - emin], [emax - x_center_lin]])

    return ax.errorbar(x_center_lin, y_med, xerr=xerr, yerr=yerr, **style, zorder=5)
