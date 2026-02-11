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
    alpha=0.45,
    edgecolor="black",
    lw=0.8,
    bw_method="scott",
    grid_size=200,
    errorbar_kwargs=None,
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
    color : str, optional
        Fill color of the violin polygons.
    alpha : float, optional
        Fill transparency of the violins.
    edgecolor : str, optional
        Color of the violin edges.
    lw : float, optional
        Line width of the violin polygon edges.
    bw_method : str or float, optional
        Bandwidth selection passed to ``scipy.stats.gaussian_kde``.
    grid_size : int, optional
        Number of evaluation points in log-flux space for the KDE grid.
    errorbar_kwargs : dict, optional
        Keyword arguments forwarded to ``ax.errorbar`` for the quantile bars.
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

    quantile_levels = (100 * norm.cdf(-1), 50, 100 * norm.cdf(1))
    ul_level = 100 * norm.cdf(2)
    widths_scale = 0.99  # shrink half log-width so adjacent violins don’t touch
    if violin_clip is None:
        violin_clip = [norm.cdf(-4), norm.cdf(4)]
    ax.set_xscale("log")
    ax.set_yscale("log")

    energy_edges = u.Quantity(energy_edges)
    x_unit_symbol = f"{energy_edges.unit}"
    edges = np.asarray(energy_edges.to_value())

    y_unit = f"{samples_per_band[0].unit}"

    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("energy_edges must be a 1D array with length >= 2 (nbins+1).")
    if (
        not np.all(np.isfinite(edges))
        or np.any(edges <= 0)
        or not np.all(np.diff(edges) > 0)
    ):
        raise ValueError(
            "Edges must be strictly increasing, finite, and > 0 for log axes."
        )

    Emins = edges[:-1]
    Emaxs = edges[1:]
    nbins = Emins.size

    if len(samples_per_band) != nbins:
        raise ValueError(
            f"len(samples_per_band)={len(samples_per_band)} "
            f"must equal len(energy_edges)-1={nbins}."
        )
    if weights_per_band is None:
        weights_per_band = [None] * nbins
    elif len(weights_per_band) != nbins:
        raise ValueError(
            f"len(weights_per_band)={len(weights_per_band)} "
            f"must equal len(energy_edges)-1={nbins}."
        )

    # centers and half widths in log(E)
    xlog_min = np.log10(Emins)
    xlog_max = np.log10(Emaxs)
    xlog_centers = 0.5 * (xlog_min + xlog_max)
    half_widths_log = 0.5 * (xlog_max - xlog_min) * float(widths_scale)

    # weighted KDE in log-space
    def kde_on_grid_weighted(ylog, w, grid_size, bw_method):
        ygrid = np.linspace(ylog.min(), ylog.max(), int(grid_size))
        kde = gaussian_kde(ylog, bw_method=bw_method, weights=w)
        dens = kde(ygrid)
        m = dens.max()
        dens = dens / m if m > 0 else dens
        return ygrid, dens

    # default errorbar style
    errorbar_kwargs = dict(
        marker="o",
        ms=4.5,
        mec=edgecolor,
        mfc="white",
        color=edgecolor,
        capsize=2.5,
        elinewidth=1.2,
        lw=1.2,
    )
    errorbar_kwargs_ul = dict(
        marker="v",
        ms=7,
        mec=edgecolor,
        mfc="white",
        color=edgecolor,
        capsize=2.5,
        elinewidth=1.2,
        lw=1.2,
    )

    # validate clip bounds
    if violin_clip is not None:
        lc, uc = float(violin_clip[0]), float(violin_clip[1])
        if not (0.0 <= lc < uc <= 1.0):
            raise ValueError(
                "violin_clip must be a (low, high) tuple with 0 <= low < high <= 1"
            )
    artists = []

    # --- iterate bins ---
    for samples, weights, xlog_c, hwlog, Emin, Emax in zip(
        samples_per_band, weights_per_band, xlog_centers, half_widths_log, Emins, Emaxs
    ):
        s = samples.copy().to_value(y_unit)
        n_neg = sum(s < 0)
        smin_pos = s[s > 0].min()
        if n_neg > 0:
            s[s < 0] = smin_pos * 0.1

        if weights is None:
            w = np.ones_like(s, float)
        else:
            w = np.asarray(weights.copy(), float)
        mask = np.isfinite(s) & np.isfinite(w) & (w >= 0)
        s = s[mask]
        w = w[mask]
        if s.size == 0 or w.sum() == 0:
            continue
        w = w / w.sum()

        # transformed log variable: log(E^p * F) = log(F) + p * log(E_center)
        shift = energy_power * xlog_c if energy_power else 0.0
        scale = (10**xlog_c) ** energy_power
        ylog_all = np.log10(s) + shift

        # Tail clipping bounds
        lc, uc = violin_clip
        y_low = np.percentile(s, 100 * lc, weights=w, method="inverted_cdf") * scale
        y_high = np.percentile(s, 100 * uc, weights=w, method="inverted_cdf") * scale
        if n_neg > 0:
            y_low = max(smin_pos * scale, y_low)
        # If degenerate or invalid, skip clipping
        if y_high <= y_low:
            y_low, y_high = None, None

        # KDE grid limits
        if (y_low is not None) and (y_high is not None):
            ygrid_log_min = max(np.min(ylog_all), np.log10(y_low))
            ygrid_log_max = min(np.max(ylog_all), np.log10(y_high))
            # Build KDE within the clipped interval by trimming the grid after KDE eval
            ygrid_log_full, dens_full = kde_on_grid_weighted(
                ylog_all, w, grid_size, bw_method
            )
            mask_clip = (ygrid_log_full >= ygrid_log_min) & (
                ygrid_log_full <= ygrid_log_max
            )
            ygrid_log = ygrid_log_full[mask_clip]
            dens = dens_full[mask_clip]
        else:
            ygrid_log, dens = kde_on_grid_weighted(ylog_all, w, grid_size, bw_method)

        # Symmetric left/right edges in log(E)
        xlog_left = xlog_c - hwlog * dens
        xlog_right = xlog_c + hwlog * dens
        x_left, x_right = 10 ** (xlog_left), 10 ** (xlog_right)
        ygrid = 10 ** (ygrid_log)
        x_poly = np.concatenate([x_left, x_right[::-1]])
        y_poly = np.concatenate([ygrid, ygrid[::-1]])
        poly = ax.fill(
            x_poly,
            y_poly,
            facecolor=color,
            edgecolor=edgecolor,
            alpha=alpha,
            lw=lw,
            zorder=2,
        )
        artists.extend(poly)

        # error bars
        if quantile_levels:
            y_qs = [
                np.percentile(samples.value, q, weights=weights, method="inverted_cdf")
                * scale
                for q in quantile_levels
            ]

            y_lo, y_med, y_hi = y_qs[0], y_qs[1], y_qs[2]
            if y_med > 0:
                yerr = np.array([[max(0.0, y_med - y_lo)], [max(0.0, y_hi - y_med)]])
                fp_kwargs = errorbar_kwargs
            else:
                yerr = None
                y_med = (
                    np.percentile(
                        samples.value, ul_level, weights=weights, method="inverted_cdf"
                    )
                    * scale
                )
                fp_kwargs = errorbar_kwargs_ul
            x_cen = 10 ** (xlog_c)
            xerr = np.array([[x_cen - Emin], [Emax - x_cen]])
            eb = ax.errorbar(
                x_cen, y_med, xerr=xerr, yerr=yerr, **(fp_kwargs or {}), zorder=5
            )
            artists.append(eb)

    # axes labels
    # X
    ax.set_xlabel(f"Energy [{x_unit_symbol}]")

    # Y with E^p × prefix and proper unit string
    if energy_power and abs(energy_power) > 0:
        p_str = f"{energy_power:g}"
        prefix = rf"$E^{p_str}\,\times$ "
        if x_unit_symbol is not None:
            if y_unit:
                unit_str = rf"[{x_unit_symbol}$^{p_str}$ × {y_unit}]"
            else:
                unit_str = rf"[{x_unit_symbol}$^{p_str}$]"
        else:
            unit_str = f"[{y_unit}]" if y_unit else ""
        ax.set_ylabel(prefix + y_label + (" " + unit_str if unit_str else ""))
    else:
        ylab = y_label + (f" [{y_unit}]" if y_unit else "")
        ax.set_ylabel(ylab)

    return artists
