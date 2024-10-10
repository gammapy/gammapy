# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.maps import Map
from gammapy.modeling.models.utils import cutout_template_models
from . import Datasets

__all__ = ["apply_edisp", "split_dataset"]

log = logging.getLogger(__name__)


def apply_edisp(input_map, edisp):
    """Apply energy dispersion to map. Requires "energy_true" axis.

    Parameters
    ----------
    input_map : `~gammapy.maps.Map`
        The map to be convolved with the energy dispersion.
        It must have an axis named "energy_true".
    edisp : `~gammapy.irf.EDispKernel`
        Energy dispersion matrix.

    Returns
    -------
    map : `~gammapy.maps.Map`
        Map with energy dispersion applied.

    Examples
    --------
    >>> from gammapy.irf.edisp import EDispKernel
    >>> from gammapy.datasets.utils import apply_edisp
    >>> from gammapy.maps import MapAxis, Map
    >>> import numpy as np
    >>>
    >>> axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=6, name="energy_true")
    >>> m = Map.create(
    ...     skydir=(0.8, 0.8),
    ...     width=(1, 1),
    ...     binsz=0.02,
    ...     axes=[axis],
    ...     frame="galactic"
    ... )
    >>> e_true = m.geom.axes[0]
    >>> e_reco = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=3)
    >>> edisp = EDispKernel.from_diagonal_response(energy_axis_true=e_true, energy_axis=e_reco)
    >>> map_edisp = apply_edisp(m, edisp)
    >>> print(map_edisp)
    WcsNDMap
    <BLANKLINE>
        geom  : WcsGeom
        axes  : ['lon', 'lat', 'energy']
        shape : (50, 50, 3)
        ndim  : 3
        unit  :
        dtype : float64
    """
    # TODO: either use sparse matrix multiplication or something like edisp.is_diagonal
    if edisp is not None:
        loc = input_map.geom.axes.index("energy_true")
        data = np.rollaxis(input_map.data, loc, len(input_map.data.shape))
        data = np.dot(data, edisp.pdf_matrix)
        data = np.rollaxis(data, -1, loc)
        energy_axis = edisp.axes["energy"].copy(name="energy")
    else:
        data = input_map.data
        energy_axis = input_map.geom.axes["energy_true"].copy(name="energy")

    geom = input_map.geom.to_image().to_cube(axes=[energy_axis])
    return Map.from_geom(geom=geom, data=data, unit=input_map.unit)


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


def split_dataset(dataset, width, margin, split_template_models=True):
    """Split dataset in multiple non-overlapping analysis regions.

    Parameters
    ----------
    dataset : `~gammapy.datasets.Dataset`
        Dataset to split.
    width : `~astropy.coordinates.Angle`
        Angular size of each sub-region.
    margin : `~astropy.coordinates.Angle`
        Angular size to be added to the `width`.
        The margin should be defined such as sources outside the region of interest
        that contributes inside are well-defined.
        The mask_fit in the margin region is False and unchanged elsewhere.
    split_template_models : bool, optional
        Apply cutout to template models or not. Default is True.

    Returns
    -------
    datasets : `~gammapy.datasets.Datasets`
        Split datasets.

    Examples
    --------
    >>> from gammapy.datasets import MapDataset
    >>> from gammapy.datasets.utils import split_dataset
    >>> from gammapy.modeling.models import GaussianSpatialModel, PowerLawSpectralModel, SkyModel
    >>> import astropy.units as u
    >>> dataset = MapDataset.read("$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz")
    >>> # Split the dataset
    >>> width = 4 * u.deg
    >>> margin = 1 * u.deg
    >>> split_datasets = split_dataset(dataset, width, margin, split_template_models=False)
    >>> # Apply a model and split the dataset
    >>> spatial_model = GaussianSpatialModel()
    >>> spectral_model = PowerLawSpectralModel()
    >>> sky_model = SkyModel(
    ...     spatial_model=spatial_model, spectral_model=spectral_model, name="test-model"
    ... )
    >>> dataset.models = [sky_model]
    >>> split_datasets = split_dataset(
    ...     dataset, width=width, margin=margin, split_template_models=True
    ... )
    """

    if margin >= width / 2.0:
        raise ValueError("margin should be lower than width/2.")

    geom = dataset.counts.geom.to_image()
    pixel_width = np.ceil((width / geom.pixel_scales).to_value("")).astype(int)

    ilon = range(0, geom.data_shape[1], pixel_width[1])
    ilat = range(0, geom.data_shape[0], pixel_width[0])

    datasets = Datasets()
    for il in ilon:
        for ib in ilat:
            l, b = geom.pix_to_coord((il, ib))
            cutout_kwargs = dict(
                position=SkyCoord(l, b, frame=geom.frame), width=width + 2 * margin
            )
            d = dataset.cutout(**cutout_kwargs)

            # apply mask
            geom_cut = d.counts.geom
            geom_cut_image = geom_cut.to_image()
            ilgrid, ibgrid = np.meshgrid(
                range(geom_cut_image.data_shape[1]), range(geom_cut_image.data_shape[0])
            )
            il_cut, ib_cut = geom_cut_image.coord_to_pix((l, b))
            mask = (ilgrid >= il_cut - pixel_width[1] / 2.0) & (
                ibgrid >= ib_cut - pixel_width[0] / 2.0
            )
            if il == ilon[-1]:
                mask &= ilgrid <= il_cut + pixel_width[1] / 2.0
            else:
                mask &= ilgrid < il_cut + pixel_width[1] / 2.0

            if ib == ilat[-1]:
                mask &= ibgrid <= ib_cut + pixel_width[0] / 2.0
            else:
                mask &= ibgrid < ib_cut + pixel_width[0] / 2.0
            mask = np.expand_dims(mask, 0)
            mask = np.repeat(mask, geom_cut.data_shape[0], axis=0)
            d.mask_fit = Map.from_geom(geom_cut, data=mask)
            if dataset.mask_fit is not None:
                d.mask_fit &= dataset.mask_fit.interp_to_geom(
                    geom_cut, method="nearest"
                )

            # template models cutout (should limit memory usage in parallel)
            if split_template_models:
                d.models = cutout_template_models(
                    dataset.models, cutout_kwargs, [d.name]
                )
            else:
                d.models = dataset.models
            datasets.append(d)
    return datasets


def create_energy_mask(dataset, energy_min=None, energy_max=None):
    """Build a mask for the dataset restricting its mask_fit to a given energy range.

    Parameters
    ----------
    dataset : `~gammapy.datasets.Dataset`
        the dataset to compute a mask for.
    energy_min : `~astropy.units.Quantity`
        minimum energy.
    energy_max : `~astropy.units.Quantity`
        maximum energy.
    """
    energy_axis = dataset._geom.axes["energy"]

    if energy_min is None:
        energy_min = energy_axis.bounds[0]

    if energy_max is None:
        energy_max = energy_axis.bounds[1]

    energy_min, energy_max = u.Quantity(energy_min), u.Quantity(energy_max)

    group = energy_axis.group_table(edges=[energy_min, energy_max])

    is_normal = group["bin_type"] == "normal   "
    group = group[is_normal]

    mask_fit = dataset._geom.energy_mask(
        group["energy_min"][0] * energy_axis.unit,
        group["energy_max"][0] * energy_axis.unit,
    )

    if dataset.mask_fit is not None:
        mask_fit *= dataset.mask_fit

    return mask_fit


class set_and_restore_mask_fit:
    """Context manager to set a `mask_fit` oa dataset and restore the initial mask.

    Parameters
    ----------
    datasets : `~gammapy.datasets.datasets`
        the Datasets to apply the energy mask to.
    energy_min : `~astropy.units.Quantity`
        minimum energy.
    energy_min : `~astropy.units.Quantity`
        maximum energy.
    """

    def __init__(self, datasets, energy_min, energy_max):
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.datasets = datasets
        self.mask_fits = [dataset.mask_fit for dataset in datasets]

    def __enter__(self):
        datasets = Datasets()
        for dataset in self.datasets:
            try:
                mask_fit = create_energy_mask(dataset, self.energy_min, self.energy_max)
                if dataset.mask_fit is not None:
                    mask_fit *= dataset.mask_fit
                dataset.mask_fit = mask_fit
            except ValueError:
                log.info(
                    f"Dataset {dataset.name} does not contribute in the energy range"
                )
                continue

            datasets.append(dataset)
        return datasets

    def __exit__(self, type, value, traceback):
        for dataset, mask_fit in zip(self.datasets, self.mask_fits):
            dataset.mask_fit = mask_fit
