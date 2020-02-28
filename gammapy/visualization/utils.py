

__all__ = ["plot_spectrum_datasets_off_regions"]


def plot_spectrum_datasets_off_regions(datasets, ax=None):
    """Plot spectrum datasets of regions.

    Parameters
    ----------
    datasets : list of `SpectrumDatasetOnOff`
        List of spectrum on-off datasets
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    ax = plt.gca() or ax

    color_cycle = plt.rcParams["axes.prop_cycle"]
    colors = color_cycle.by_key()["color"]
    handles = []

    for color, dataset in zip(colors, datasets):
        kwargs = {"edgecolor": color, "facecolor": "none"}
        dataset.counts_off.plot_region(ax=ax, **kwargs)

        # create proxy artist for the custom legend
        handle = mpatches.Patch(label=dataset.name, **kwargs)
        handles.append(handle)

    plt.legend(handles=handles)
