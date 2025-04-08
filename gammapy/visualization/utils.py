# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging as log
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.stats import norm
from astropy.visualization import make_lupton_rgb
import matplotlib.axes as maxes
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

__all__ = [
    "add_colorbar",
    "plot_contour_line",
    "plot_map_rgb",
    "plot_theta_squared_table",
    "plot_distribution",
]


ARTIST_TO_LINE_PROPERTIES = {
    "color": "markeredgecolor",
    "edgecolor": "markeredgecolor",
    "ec": "markeredgecolor",
    "facecolor": "markerfacecolor",
    "fc": "markerfacecolor",
    "linewidth": "markeredgewidth",
    "lw": "markeredgewidth",
}


def add_colorbar(img, ax, axes_loc=None, **kwargs):
    """
    Add colorbar to a given axis.

    Parameters
    ----------
    img : `~matplotlib.image.AxesImage`
        The image to plot the colorbar for.
    ax : `~matplotlib.axes.Axes`
        Matplotlib axes.
    axes_loc : dict, optional
        Keyword arguments passed to `~mpl_toolkits.axes_grid1.axes_divider.AxesDivider.append_axes`.
    kwargs : dict, optional
        Keyword arguments passed to `~matplotlib.pyplot.colorbar`.

    Returns
    -------
    cbar : `~matplotlib.pyplot.colorbar`
        The colorbar.

    Examples
    --------
    .. testcode::

        from gammapy.maps import Map
        from gammapy.visualization import add_colorbar
        import matplotlib.pyplot as plt
        map_ = Map.read("$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz")
        axes_loc = {"position": "right", "size": "2%", "pad": "10%"}
        kwargs_colorbar = {'label':'Colorbar label'}

        # Example outside gammapy
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        img = ax.imshow(map_.sum_over_axes().data[0,:,:])
        add_colorbar(img, ax=ax, axes_loc=axes_loc, **kwargs_colorbar)

        # `add_colorbar` is available for the `plot` function here:
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        map_.sum_over_axes().plot(ax=ax, add_cbar=True, axes_loc=axes_loc,
                                  kwargs_colorbar=kwargs_colorbar)  # doctest: +SKIP

    """
    kwargs.setdefault("use_gridspec", True)
    kwargs.setdefault("orientation", "vertical")

    axes_loc = axes_loc or {}
    axes_loc.setdefault("position", "right")
    axes_loc.setdefault("size", "5%")
    axes_loc.setdefault("pad", "2%")
    axes_loc.setdefault("axes_class", maxes.Axes)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(**axes_loc)
    cbar = plt.colorbar(img, cax=cax, **kwargs)
    return cbar


def plot_map_rgb(map_, ax=None, **kwargs):
    """
    Plot RGB image on matplotlib WCS axes.

    This function is based on the `~astropy.visualization.make_lupton_rgb` function. The input map must
    contain 1 non-spatial axis with exactly 3 bins. If this is not the case, the map has to be resampled
    before using the `plot_map_rgb` function (e.g. as shown in the code example below).

    Parameters
    ----------
    map_ : `~gammapy.maps.WcsNDMap`
        WCS map. The map must contain 1 non-spatial axis with exactly 3 bins.
    ax : `~astropy.visualization.wcsaxes.WCSAxes`, optional
        WCS axis object to plot on.
    **kwargs : dict
        Keyword arguments passed to `~astropy.visualization.make_lupton_rgb`.

    Returns
    -------
    ax : `~astropy.visualization.wcsaxes.WCSAxes`
        WCS axis object.

    Examples
    --------
    >>> from gammapy.visualization import plot_map_rgb
    >>> from gammapy.maps import Map, MapAxis
    >>> import astropy.units as u
    >>> map_ = Map.read("$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz")
    >>> axis_rgb = MapAxis.from_energy_edges(
    ...     [0.1, 0.2, 0.5, 10], unit=u.TeV, name="energy", interp="log"
    ... )
    >>> map_ = map_.resample_axis(axis_rgb)
    >>> kwargs = {"stretch": 0.5, "Q": 1, "minimum": 0.15}
    >>> plot_map_rgb(map_.smooth(0.08*u.deg), **kwargs) #doctest: +SKIP
    """
    geom = map_.geom
    if len(geom.axes) != 1 or geom.axes[0].nbin != 3:
        raise ValueError(
            "One non-spatial axis with exactly 3 bins is needed to plot an RGB image"
        )

    data = [data_slice / np.nanmax(data_slice.flatten()) for data_slice in map_.data]
    data = make_lupton_rgb(*data, **kwargs)

    ax = map_._plot_default_axes(ax=ax)
    ax.imshow(data)

    if geom.is_allsky:
        ax = map_._plot_format_allsky(ax)
    else:
        ax = map_._plot_format(ax)

    # without this the axis limits are changed when calling scatter
    ax.autoscale(enable=False)

    return ax


def plot_contour_line(ax, x, y, **kwargs):
    """Plot smooth curve from contour points."""
    xf = x
    yf = y

    # close contour
    if not (x[0] == x[-1] and y[0] == y[-1]):
        xf = np.append(x, x[0])
        yf = np.append(y, y[0])

    # curve parametrization must be strictly increasing
    # so we use the cumulative distance of each point from the first one
    dist = np.sqrt(np.diff(xf) ** 2.0 + np.diff(yf) ** 2.0)
    dist = [0] + list(dist)
    t = np.cumsum(dist)
    ts = np.linspace(0, t[-1], 50)

    # 1D cubic spline interpolation
    cs = CubicSpline(t, np.c_[xf, yf], bc_type="periodic")
    out = cs(ts)

    # plot
    if "marker" in kwargs.keys():
        marker = kwargs.pop("marker")
    else:
        marker = "+"
    if "color" in kwargs.keys():
        color = kwargs.pop("color")
    else:
        color = "b"

    ax.plot(out[:, 0], out[:, 1], "-", color=color, **kwargs)
    ax.plot(xf, yf, linestyle="", marker=marker, color=color)


def plot_theta_squared_table(table):
    """Plot the theta2 distribution of counts, excess and significance.

    Take the table containing the ON counts, the OFF counts, the acceptance,
    the off acceptance and the alpha (normalisation between ON and OFF)
    for each theta2 bin.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Required columns: theta2_min, theta2_max, counts, counts_off and alpha
    """
    from gammapy.maps import MapAxis
    from gammapy.maps.axes import UNIT_STRING_FORMAT
    from gammapy.maps.utils import edges_from_lo_hi

    theta2_edges = edges_from_lo_hi(
        table["theta2_min"].quantity, table["theta2_max"].quantity
    )
    theta2_axis = MapAxis.from_edges(theta2_edges, interp="lin", name="theta_squared")

    ax0 = plt.subplot(2, 1, 1)

    x = theta2_axis.center.value
    x_edges = theta2_axis.edges.value
    xerr = (x - x_edges[:-1], x_edges[1:] - x)

    ax0.errorbar(
        x,
        table["counts"],
        xerr=xerr,
        yerr=np.sqrt(table["counts"]),
        linestyle="None",
        label="Counts",
    )

    ax0.errorbar(
        x,
        table["counts_off"],
        xerr=xerr,
        yerr=np.sqrt(table["counts_off"]),
        linestyle="None",
        label="Counts Off",
    )

    ax0.errorbar(
        x,
        table["excess"],
        xerr=xerr,
        yerr=(table["excess_errn"], table["excess_errp"]),
        fmt="+",
        linestyle="None",
        label="Excess",
    )

    ax0.set_ylabel("Counts")
    ax0.set_xticks([])
    ax0.set_xlabel("")
    ax0.legend()

    ax1 = plt.subplot(2, 1, 2)
    ax1.errorbar(x, table["sqrt_ts"], xerr=xerr, linestyle="None")
    ax1.set_xlabel(f"Theta [{theta2_axis.unit.to_string(UNIT_STRING_FORMAT)}]")
    ax1.set_ylabel("Significance")


def plot_distribution(
    wcs_map,
    mask=None,
    ax=None,
    ncols=3,
    func=None,
    kwargs_hist=None,
    kwargs_axes=None,
    kwargs_fit=None,
):
    """
    Plot the 1D distribution of data inside a map as an histogram. If the dimension of the map is smaller than 2,
    a unique plot will be displayed. Otherwise, if the dimension is 3 or greater, a grid of plot will be displayed.

    Parameters
    ----------
    wcs_map : `~gammapy.maps.WcsNDMap`
        A map that contains data to be plotted.
    mask : `~gammapy.maps.WcsNDMap`, optional
        2D mask defining the input data region.
    ax : `~matplotlib.axes.Axes` or list of `~matplotlib.axes.Axes`
        Axis object to plot on. If a list of Axis is provided it has to be the same length as the length of _map.data.
    ncols : int
        Number of columns to plot if a "plot grid" was to be done.
    func : function object or str
        The function used to fit a map data histogram or "norm". Default is None.
        If None, no fit will be performed. If "norm" is given, `scipy.stats.norm.pdf`
        will be passed to `scipy.optimize.curve_fit`.
    kwargs_hist : dict
        Keyword arguments to pass to `matplotlib.pyplot.hist`.
    kwargs_axes : dict
        Keyword arguments to pass to `matplotlib.axes.Axes`.
    kwargs_fit : dict
        Keyword arguments to pass to `scipy.optimize.curve_fit`

    Returns
    -------
    axes : `~numpy.ndarray` of `~matplotlib.pyplot.Axes`
        Array of Axes.
    result_list : list of dict
        List of dictionnary that contains the results of `scipy.optimize.curve_fit`. The number of elements in the list
        correspond to the dimension of the non-spatial axis of the map.
        The dictionnary contains:

            * `axis_edges` : the edges of the non-spatial axis bin used
            * `param` : the best-fit parameters of the input function `func`
            * `covar` : the covariance matrix for the fitted parameters `param`
            * `info_dict` : the `infodict` return of `scipy.optimize.curve_fit`

    Examples
    --------
    >>> from gammapy.datasets import MapDataset
    >>> from gammapy.estimators import TSMapEstimator
    >>> from scipy.stats import norm
    >>> from gammapy.visualization import plot_distribution
    >>> dataset = MapDataset.read("$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz")
    >>> tsmap_est = TSMapEstimator().run(dataset)
    >>> axs, res = plot_distribution(tsmap_est.sqrt_ts, func="norm", kwargs_hist={'bins': 75, 'range': (-10, 10), 'density': True})
    >>> # Equivalently, one can do the following:
    >>> func = lambda x, mu, sig : norm.pdf(x, loc=mu, scale=sig)
    >>> axs, res = plot_distribution(tsmap_est.sqrt_ts, func=func, kwargs_hist={'bins': 75, 'range': (-10, 10), 'density': True})
    """

    from gammapy.maps import WcsNDMap  # import here to avoid circular import

    if not isinstance(wcs_map, WcsNDMap):
        raise TypeError(
            f"map_ must be an instance of gammapy.maps.WcsNDMap, given {type(wcs_map)}"
        )

    kwargs_hist = kwargs_hist or {}
    kwargs_axes = kwargs_axes or {}
    kwargs_fit = kwargs_fit or {}

    kwargs_hist.setdefault("density", True)
    kwargs_fit.setdefault("full_output", True)

    cutout, cutout_mask = wcs_map.cutout_and_mask_region()

    if mask:
        if mask.geom != wcs_map.geom.to_image():
            raise ValueError("Map and mask spatial geometry must agree!")

        if not mask.is_mask:
            raise ValueError(
                f"Mask map must be of boolean type, got {mask.data.dtype} instead."
            )

        cutout_mask.data = np.logical_and(cutout_mask.data, mask.data)

    idx_x, idx_y = np.where(cutout_mask)

    data = cutout.data[..., idx_x, idx_y]

    if ax is None:
        n_plot = len(data)
        cols = min(ncols, n_plot)
        rows = 1 + (n_plot - 1) // cols

        width = 12
        figsize = (width, width * rows / cols)

        fig, axes = plt.subplots(
            nrows=rows,
            ncols=cols,
            figsize=figsize,
        )
        cells_in_grid = rows * cols
    else:
        axes = np.array([ax])
        cells_in_grid = axes.size

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    result_list = []

    for idx in range(cells_in_grid):
        axe = axes.flat[idx]
        if idx > len(data) - 1:
            axe.set_visible(False)
            continue
        d = data[idx][np.isfinite(data[idx])]
        n, bins, _ = axe.hist(d, **kwargs_hist)

        if func is not None:
            kwargs_plot_fit = {"label": "Fit"}
            centers = 0.5 * (bins[1:] + bins[:-1])

            if func == "norm":

                def func_to_fit(x, mu, sigma):
                    return norm.pdf(x, mu, sigma)

                pars, cov, infodict, message, _ = curve_fit(
                    func_to_fit, centers, n, **kwargs_fit
                )

                mu, sig = pars[0], pars[1]
                err_mu, err_sig = np.sqrt(cov[0][0]), np.sqrt(cov[1][1])

                label_norm = (
                    r"$\mu$ = {:.2f} ± {:.2E}\n$\sigma$ = {:.2f} ± {:.2E}".format(
                        mu, err_mu, sig, err_sig
                    )
                ).replace(r"\n", "\n")
                kwargs_plot_fit["label"] = label_norm

            else:
                func_to_fit = func

                pars, cov, infodict, message, _ = curve_fit(
                    func_to_fit, centers, n, **kwargs_fit
                )

            axis_edges = (
                wcs_map.geom.axes[-1].edges[idx],
                wcs_map.geom.axes[-1].edges[idx + 1],
            )
            result_dict = {
                "axis_edges": axis_edges,
                "param": pars,
                "covar": cov,
                "info_dict": infodict,
            }
            result_list.append(result_dict)
            log.info(message)

            xmin, xmax = kwargs_hist.get("range", (np.min(d), np.max(d)))
            x = np.linspace(xmin, xmax, 1000)

            axe.plot(x, func_to_fit(x, *pars), lw=2, color="black", **kwargs_plot_fit)

        axe.set(**kwargs_axes)
        axe.legend()

    return axes, result_list


def plot_pulse_profile_3PC(
    source,
    n_period=2,
    add_radio_profile=True,
    add_best_fit_profile=True,
    add_error=True,
):
    """"""
    # Import here to avoid circular imports
    from gammapy.catalog import SourceCatalogObject3PC

    key_names = [
        "50_100_WtCt",
        "100_300_WtCt",
        "300_1000_WtCt",
        "1000_3000_WtCt",
        "3000_100000_WtCt",
        "10000_100000_WtCt",
    ]

    if not isinstance(source, SourceCatalogObject3PC):
        raise TypeError(
            f"`source` must be an instance of `~gammapy.catalog.SourceCatalogObject3PC`, got {type(source)}."
        )
    if n_period > 2:
        raise ValueError(f"`n_preiod` must be either 1 or 2, got {n_period}.")

    fig, axes = plt.subplots(
        ncols=1,
        nrows=6,
        figsize=(7, 12),
        gridspec_kw={"height_ratios": [1.6, 1, 1, 1, 1, 1]},
    )

    plt.subplots_adjust(wspace=0, hspace=0)
    for ax in axes[:-1]:
        ax.tick_params(
            axis="x", direction="inout", labelbottom=False, top=True, bottom=True
        )
    axes[0].tick_params(
        axis="x", direction="inout", labeltop=True, top=True, bottom=True
    )
    axes[-1].tick_params(
        axis="x", direction="inout", labelbottom=True, top=True, bottom=True
    )

    ax_ylim = [np.inf, 0]

    try:
        radio_profile = source.pulse_profile_radio
    except KeyError:
        add_radio_profile = False
        log.warning(f"No Radio profile available for PSR {source.name}, skipping.")

    if add_radio_profile:
        radio_axis = radio_profile.geom.axes["phase"]
        axes[0].plot(
            radio_axis.as_plot_center,
            radio_profile.data.squeeze(),
            color="#FB462A",
            lw=0.5,
            label="Radio",
        )
        ax_ylim[0] = (
            radio_profile.data.squeeze().min()
            - 0.1 * radio_profile.data.squeeze().min()
        )
        ax_ylim[1] = (
            radio_profile.data.squeeze().max()
            + 0.1 * radio_profile.data.squeeze().max()
        )

    try:
        best_fit_profile = source.pulse_profile_best_fit
    except KeyError:
        best_fit_profile = False
        log.warning(f"No Best fit profile available for PSR {source.name}, skipping.")

    if add_best_fit_profile:
        best_fit_profile = source.pulse_profile_best_fit
        best_fit_axis = best_fit_profile.geom.axes["phase"]
        axes[0].plot(
            best_fit_axis.as_plot_center,
            best_fit_profile.data.squeeze(),
            color="teal",
            lw=2,
            ls="--",
            label="Best-fit",
        )
        ax_ylim[0] = min(
            best_fit_profile.data.squeeze().min()
            - 0.1 * best_fit_profile.data.squeeze().min(),
            ax_ylim[0],
        )
        ax_ylim[1] = max(
            best_fit_profile.data.squeeze().max()
            + 0.1 * best_fit_profile.data.squeeze().max(),
            ax_ylim[1],
        )

    profiles = source.pulse_profiles
    axis = profiles["GT100_WtCnt"].geom.axes["phase"]
    data = profiles["GT100_WtCnt"].data.squeeze()
    axes[0].hist(
        axis.as_plot_center,
        color="#202449",
        bins=axis.as_plot_edges,
        weights=data,
        histtype="step",
        lw=1,
        label="> 100 MeV",
    )
    ax_ylim[0] = min(data.min() - 0.1 * data.min(), ax_ylim[0])
    ax_ylim[1] = max(data.max() + 0.1 * data.max(), ax_ylim[1])
    if add_error:
        errors = profiles["Unc_GT100_WtCnt"].data.squeeze()
        axes[0].errorbar(
            axis.as_plot_center,
            data,
            yerr=errors,
            fmt="none",
            color="#202449",
            elinewidth=0.5,
        )
        ax_ylim[0] = min(
            (data - errors).min() - 0.1 * (data - errors).min(), ax_ylim[0]
        )
        ax_ylim[1] = max(
            (data + errors).max() + 0.1 * (data + errors).max(), ax_ylim[1]
        )

    axes[0].set_ylim(ax_ylim)
    axes[0].set_xlim(0, int(n_period))
    axes[0].legend(loc=1)

    for i, (ax, name) in enumerate(
        zip(np.concatenate([axes[1:2], axes[1:]]), reversed(key_names))
    ):
        kwargs = {}
        kwargs["color"] = "#202449"
        kwargs["histtype"] = "step"
        if i == 0:
            kwargs["color"] = "grey"
            kwargs["histtype"] = "stepfilled"
        axis = profiles[name].geom.axes["phase"]
        data = profiles[name].data.squeeze()
        label = f"{int(name.split('_')[0])/1000} - {int(name.split('_')[1])/1000} GeV"
        ax.hist(
            axis.as_plot_center,
            bins=axis.as_plot_edges,
            weights=data,
            label=label,
            lw=1,
            **kwargs,
        )
        ylim = [data.min() - 0.1 * data.min(), data.max() + 0.1 * data.max()]
        if add_error:
            errors = profiles[f"Unc_{name}"].data.squeeze()
            ax.errorbar(
                axis.as_plot_center,
                data,
                yerr=errors,
                fmt="none",
                color=kwargs["color"],
                elinewidth=0.5,
            )
            ylim[0] = min((data - errors).min() - 0.1 * (data - errors).min(), ylim[0])
            ylim[1] = max((data + errors).max() + 0.1 * (data + errors).max(), ylim[1])
        if i != 0:
            ax.set_ylim(ylim)
        ax.set_xlim(0, int(n_period))
        ax.legend(loc=1)

    for ax in axes:
        ax.set_ylabel("Weighted Counts")
        ax.set_xlabel("Phase")

    return axes
