def get_figure(ax, fig, fig_kwargs):
    import matplotlib.pyplot as plt

    if ax:
        return ax.figure
    elif fig:
        return fig
    elif fig_kwargs:
        return plt.figure(**fig_kwargs)
    return plt.gcf()


def get_axes(ax, fig, ax_kwargs, fig_kwargs):
    fig = get_figure(ax, fig, fig_kwargs)

    if ax:
        return ax, fig
    elif ax_kwargs:
        fig.clf()
        return fig.add_subplot(**ax_kwargs), fig
    return fig.gca(), fig