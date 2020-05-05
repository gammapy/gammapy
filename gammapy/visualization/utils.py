import numpy as np

__all__ = ["plot_spectrum_datasets_off_regions", "plot_contour_line"]


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
