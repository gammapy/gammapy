import numpy as np

__all__ = ["plot_spectrum_datasets_off_regions", "plot_contour_line"]


def plot_spectrum_datasets_off_regions(datasets, ax=None, legend=None, **kwargs):
    """Plot spectrum datasets' off regions.

    Parameters
    ----------
    datasets : list of `SpectrumDatasetOnOff`
        List of spectrum on-off datasets.
    ax : `~`
        .
    legend : bool
        Whether to display the legend. By default True if `len(datasets) <= 10`.
    kwargs : dict
        Keyword arguments used in `~gammapy.maps.RegionNDMap.plot_region`.
        Can contain a `cycler.Cycler` in a `prop_cycle` item.

    Notes
    -----
    Properties from the `prop_cycle` have maximum priority except `edgecolor`.
    `edgecolor` is selected from the sources below in this order:
        `kwargs["edgecolor"]`
        `kwargs["prop_cycle"]`
        `~matplotlib.RcParams["axes.prop_cycle"]`
        `~matplotlib.RcParams["patch.edgecolor"]`
    `~matplotlib.RcParams["patch.facecolor"]` is never used.

    Examples
    --------
    >>> plot_spectrum_datasets_off_regions(datasets, ax, legend=False, lw=2.5)
    >>> plot_spectrum_datasets_off_regions(datasets, ax, alpha=.3, facecolor='k')
    >>> plot_spectrum_datasets_off_regions(
            datasets, ax, ls='--', prop_cycle=plt.cycler('color', list('rgb'))
        )
    >>> plt.rc('legend', fontsize=9)
    >>> plt.rc('patch', edgecolor='blue')
    >>> plot_spectrum_datasets_off_regions(
            datasets, ax, legend=True, prop_cycle=plt.cycler('ls', ['-', '-.'])
        )
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    ax = ax or plt.gca(projection=datasets[0].counts_off.geom.wcs)
    handles = []

    kwargs.setdefault("facecolor", "none")
    prop_cycle = kwargs.pop("prop_cycle", plt.rcParams["axes.prop_cycle"])
    plot_kwargs = kwargs.copy()

    for props, dataset in zip(prop_cycle(), datasets):
        props = props.copy()	# not sure why this is necessary
        color = props.pop("color", plt.rcParams["patch.edgecolor"])
        plot_kwargs["edgecolor"] = kwargs.get("edgecolor", color)
        plot_kwargs.update(props)
        dataset.counts_off.plot_region(ax, **plot_kwargs)

        # create proxy artist for the custom legend
        handle = mpatches.Patch(label=dataset.name, **plot_kwargs)
        handles.append(handle)

    if legend or legend == None and len(datasets) <= 10:
        plt.legend(handles=handles)


def plot_contour_line(ax, x, y, **kwargs):
    """Plot smooth curve from contour points"""
    from scipy.interpolate import CubicSpline

    # close countour
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
    ax.plot(xf, yf, linestyle='', marker=marker, color=color)
