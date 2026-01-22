import matplotlib.pyplot as plt
import numpy as np

import logging

log = logging.getLogger(__name__)


__all__ = ["plot_pulse_profile_3PC"]


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
