import numpy as np

__all__ = ["plot_spectrum_datasets_off_regions", "plot_contour_line"]


def plot_spectrum_datasets_off_regions(datasets, ax=None, legend=None, legend_kwargs=None, **kwargs):
    """Plot the off regions of spectrum datasets.

    Parameters
    ----------
    datasets : `~gammapy.datasets.Datasets` of or sequence of
    `~gammapy.datasets.SpectrumDatasetOnOff`
        List of spectrum on-off datasets.
    ax : `~astropy.visualization.wcsaxes.WCSAxes`
        Axes object to plot on.
    legend : bool
        Whether to add/display the labels of the off regions in a legend. By default True if
        ``len(datasets) <= 10``.
    legend_kwargs : dict
        Keyword arguments used in `matplotlib.axes.Axes.legend`. The ``handler_map`` cannot be
        overridden.
    **kwargs : dict
        Keyword arguments used in `gammapy.maps.RegionNDMap.plot_region`. Can contain a
        `~cycler.Cycler` in a ``prop_cycle`` argument.

    Notes
    -----
    Properties from the ``prop_cycle`` have maximum priority, except ``color``.
    ``edgecolor``/``color`` is selected from the sources below in this order:
        ``kwargs["edgecolor"]``

        ``kwargs["prop_cycle"]``

        ``matplotlib.rcParams["axes.prop_cycle"]``

        ``matplotlib.rcParams["patch.edgecolor"]``

    ``matplotlib.rcParams["patch.facecolor"]`` is never used.

    Examples
    --------
    Plot forcibly without legend and with thick circles:
    >>> plot_spectrum_datasets_off_regions(datasets, ax, legend=False, linewidth=2.5)

    Plot that quantifies the overlap of off regions:
    >>> plot_spectrum_datasets_off_regions(datasets, ax, alpha=0.3, facecolor='black')

    Plot that cycles through colors (``edgecolor``) and line styles together:
    >>> plot_spectrum_datasets_off_regions(datasets, ax,
        prop_cycle=plt.cycler(color=list('rgb'), ls=['--', '-', ':'])
        )

    Plot that uses a modified `~matplotlib.rcParams`, has two legend columns, static and
    dynamic colors, but only shows labels for ``datasets1`` and ``datasets2``. Note that
    ``legend_kwargs`` only applies if it's given in the last function call with ``legend=True``:
    >>> plt.rc('legend', columnspacing=1, fontsize=9)
    >>> plot_spectrum_datasets_off_regions(datasets1, ax, legend=True, edgecolor='cyan')
    >>> plot_spectrum_datasets_off_regions(datasets2, ax, legend=True, legend_kwargs=dict(ncol=2))
    >>> plot_spectrum_datasets_off_regions(datasets3, ax, legend=False, edgecolor='magenta')
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, CirclePolygon
    from matplotlib.legend_handler import HandlerTuple, HandlerPatch

    ax = ax or plt.gca(projection=datasets[0].counts_off.geom.wcs)
    legend = legend or legend is None and len(datasets) <= 10
    legend_kwargs = legend_kwargs or {}
    handles, labels = [], []

    kwargs.setdefault("facecolor", "none")
    prop_cycle = kwargs.pop("prop_cycle", plt.rcParams["axes.prop_cycle"])
    plot_kwargs = kwargs.copy()

    for props, dataset in zip(prop_cycle(), datasets):
        props = props.copy()
        color = props.pop("color", plt.rcParams["patch.edgecolor"])
        plot_kwargs["edgecolor"] = kwargs.get("edgecolor", color)
        plot_kwargs.update(props)
        dataset.counts_off.plot_region(ax, **plot_kwargs)

        # create proxy artist for the custom legend
        if legend:
            handle = Patch(**plot_kwargs)
            handles.append(handle)
            labels.append(dataset.name)

    if legend:
        legend = ax.get_legend()
        if legend:
            handles = legend.legendHandles + handles
            labels = [text.get_text() for text in legend.texts] + labels

        handles = [(handle,handle) for handle in handles]
        tuple_handler = HandlerTuple(ndivide=None, pad=0)

        def patch_func(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
            radius = width / 2
            return CirclePolygon((radius - xdescent, height / 2 - ydescent), radius)
        patch_handler = HandlerPatch(patch_func)

        legend_kwargs.setdefault("handletextpad", 0.5)
        legend_kwargs["handler_map"] = {Patch: patch_handler, tuple: tuple_handler}
        ax.legend(handles, labels, **legend_kwargs)


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
