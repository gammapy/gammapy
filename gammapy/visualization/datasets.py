# Licensed under a 3-clause BSD style license - see LICENSE.rst
import matplotlib.pyplot as plt

__all__ = [
    "plot_spectrum_datasets_off_regions",
    "plot_npred_signal",
]


def plot_spectrum_datasets_off_regions(
    datasets, ax=None, legend=None, legend_kwargs=None, **kwargs
):
    """Plot the off regions of spectrum datasets.

    Parameters
    ----------
    datasets : `~gammapy.datasets.Datasets` or list of `~gammapy.datasets.SpectrumDatasetOnOff`
        List of spectrum on-off datasets.
    ax : `~astropy.visualization.wcsaxes.WCSAxes`, optional
        Axes object to plot on. Default is None.
    legend : bool, optional
        Whether to add/display the labels of the off regions in a legend. By default True if
        ``len(datasets) <= 10``. Default is None.
    legend_kwargs : dict, optional
        Keyword arguments used in `matplotlib.axes.Axes.legend`. The ``handler_map`` cannot be
        overridden. Default is None.
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
    Plot without legend and with thick circles::

        plot_spectrum_datasets_off_regions(datasets, ax, legend=False, linewidth=2.5)

    Plot to visualise the overlap of off regions::

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
        plot_kwargs.setdefault("edgecolor")
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


def plot_npred_signal(
    dataset,
    ax=None,
    model_names=None,
    region=None,
    **kwargs,
):
    """
    Plot the energy distribution of predicted counts of a selection of models assigned to a dataset.

    The background and the sum af all the considered models predicted counts are plotted on top of individual
    contributions of the considered models.

    Parameters
    ----------
    dataset : an instance of `~gammapy.datasets.MapDataset`
        The dataset from which to plot the npred_signal.
    ax : `~matplotlib.axes.Axes`, optional
        Axis object to plot on. Default is None.
    model_names : list of str, optional
        The list of models for which the npred_signal is plotted. Default is None.
        If None, all models are considered.
    region : `~regions.Region` or `~astropy.coordinates.SkyCoord`, optional
        Region used to reproject predicted counts. Default is None.
        If None, use the full dataset geometry.
    **kwargs : dict
        Keyword arguments to pass to `~gammapy.maps.RegionNDMap.plot`.

    Returns
    -------
    axes : `~matplotlib.axes.Axes`
        Axis object.
    """

    npred_not_stack = dataset.npred_signal(
        model_names=model_names, stack=False
    ).to_region_nd_map(region)
    npred_background = dataset.npred_background().to_region_nd_map(region)

    if ax is None:
        ax = plt.gca()

    npred_not_stack.plot(ax=ax, axis_name="energy", **kwargs)
    if npred_not_stack.geom.axes["models"].nbin > 1:
        npred_stack = npred_not_stack.sum_over_axes(["models"])
        npred_stack.plot(ax=ax, label="stacked models")
    npred_background.plot(ax=ax, label="background", **kwargs)

    ax.set_ylabel("Predicted counts")
    ax.legend()

    return ax
