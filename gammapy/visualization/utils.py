import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

__all__ = [
    "plot_contour_line",
    "plot_spectrum_datasets_off_regions",
    "plot_theta_squared_table",
]


ARTIST_TO_LINE_PROPERTIES = {
    "color": "markeredgecolor",
    "edgecolor": "markeredgecolor",
    "ec": "markeredgecolor",
    "facecolor": "markerfacecolor",
    "fc": "markerfacecolor",
    "linewidth": "markerwidth",
    "lw": "markerwidth",
}


def plot_spectrum_datasets_off_regions(
    datasets, ax=None, legend=None, legend_kwargs=None, **kwargs
):
    """Plot the off regions of spectrum datasets.

    Parameters
    ----------
    datasets : `~gammapy.datasets.Datasets` or list of `~gammapy.datasets.SpectrumDatasetOnOff`
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
    Properties from the ``prop_cycle`` have maximum priority, except ``color``,
    ``edgecolor``/``color`` is selected from the sources below in this order:
    ``kwargs["edgecolor"]``, ``kwargs["prop_cycle"]``, ``matplotlib.rcParams["axes.prop_cycle"]``
    ``matplotlib.rcParams["patch.edgecolor"]``, ``matplotlib.rcParams["patch.facecolor"]``
    is never used.

    Examples
    --------
    Plot forcibly without legend and with thick circles::

        plot_spectrum_datasets_off_regions(datasets, ax, legend=False, linewidth=2.5)

    Plot that quantifies the overlap of off regions::

        plot_spectrum_datasets_off_regions(datasets, ax, alpha=0.3, facecolor='black')

    Plot that cycles through colors (``edgecolor``) and line styles together::

        plot_spectrum_datasets_off_regions(datasets, ax, prop_cycle=plt.cycler(color=list('rgb'), ls=['--', '-', ':']))  # noqa: E501

    Plot that uses a modified `~matplotlib.rcParams`, has two legend columns, static and
    dynamic colors, but only shows labels for ``datasets1`` and ``datasets2``. Note that
    ``legend_kwargs`` only applies if it's given in the last function call with ``legend=True``::

        plt.rc('legend', columnspacing=1, fontsize=9)
        plot_spectrum_datasets_off_regions(datasets1, ax, legend=True, edgecolor='cyan')
        plot_spectrum_datasets_off_regions(datasets2, ax, legend=True, legend_kwargs=dict(ncol=2))
        plot_spectrum_datasets_off_regions(datasets3, ax, legend=False, edgecolor='magenta')
    """
    from matplotlib.legend_handler import HandlerPatch, HandlerTuple
    from matplotlib.patches import CirclePolygon, Patch

    if ax is None:
        ax = plt.subplot(projection=datasets[0].counts_off.geom.wcs)

    legend = legend or legend is None and len(datasets) <= 10
    legend_kwargs = legend_kwargs or {}
    handles, labels = [], []

    prop_cycle = kwargs.pop("prop_cycle", plt.rcParams["axes.prop_cycle"])

    for props, dataset in zip(prop_cycle(), datasets):
        plot_kwargs = kwargs.copy()
        plot_kwargs["facecolor"] = "None"
        plot_kwargs.setdefault("edgecolor", props.pop("color"))
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

        handles = [(handle, handle) for handle in handles]
        tuple_handler = HandlerTuple(ndivide=None, pad=0)

        def patch_func(
            legend, orig_handle, xdescent, ydescent, width, height, fontsize
        ):
            radius = width / 2
            return CirclePolygon((radius - xdescent, height / 2 - ydescent), radius)

        patch_handler = HandlerPatch(patch_func)

        legend_kwargs.setdefault("handletextpad", 0.5)
        legend_kwargs["handler_map"] = {Patch: patch_handler, tuple: tuple_handler}
        ax.legend(handles, labels, **legend_kwargs)

    return ax


def plot_contour_line(ax, x, y, **kwargs):
    """Plot smooth curve from contour points"""
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
    """Plot the theta2 distribution of counts, excess and signifiance.

    Take the table containing the ON counts, the OFF counts, the acceptance,
    the off acceptance and the alpha (normalisation between ON and OFF)
    for each theta2 bin.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Required columns: theta2_min, theta2_max, counts, counts_off and alpha
    """
    from gammapy.maps import MapAxis
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
    ax1.set_xlabel(f"Theta  [{theta2_axis.unit}]")
    ax1.set_ylabel("Significance")
