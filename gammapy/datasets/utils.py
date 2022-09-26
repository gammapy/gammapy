import numpy as np


def get_figure(fig, width, height):
    import matplotlib.pyplot as plt

    if plt.get_fignums():
        if not fig:
            fig = plt.gcf()
        fig.clf()
    else:
        fig = plt.figure(figsize=(width, height))

    return fig


def get_axes(ax1, ax2, width, height, args1, args2, kwargs1=None, kwargs2=None):
    if not ax1 and not ax2:
        kwargs1 = kwargs1 or {}
        kwargs2 = kwargs2 or {}

        fig = get_figure(None, width, height)
        ax1 = fig.add_subplot(*args1, **kwargs1)
        ax2 = fig.add_subplot(*args2, **kwargs2)
    elif not ax1 or not ax2:
        raise ValueError("Either both or no Axes must be provided")

    return ax1, ax2


def get_nearest_valid_exposure_position(exposure, position=None):
    mask_exposure = exposure > 0.0 * exposure.unit
    mask_exposure = mask_exposure.reduce_over_axes(func=np.logical_or)
    if not position:
        position = mask_exposure.geom.center_skydir
    return mask_exposure.mask_nearest_position(position)
