# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy.coordinates import SkyCoord
from gammapy.data import Observation
import astropy.units as u
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.maps.utils import _check_width, _check_binsz
from gammapy.modeling.models.utils import cutout_template_models
from . import Datasets

__all__ = [
    "apply_edisp",
    "split_dataset",
    "create_map_dataset_from_dl4",
    "set_and_restore_mask_fit",
]

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
    <BLANKLINE>
    """
    # TODO: either use sparse matrix multiplication or something like edisp.is_diagonal
    if edisp is not None:
        loc = input_map.geom.axes.index("energy_true")
        data = np.rollaxis(input_map.data, loc, len(input_map.data.shape))
        data = np.matmul(data, edisp.pdf_matrix)
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


def create_map_dataset_from_dl4(data, geom=None, energy_axis_true=None, name=None):
    """Create a map dataset from a map dataset or an observation containing DL4 IRFs.

    Parameters
    ----------
    data : `~gammapy.dataset.MapDataset` or `~gammapy.data.Observation`
        MapDataset or Observation containing DL4 IRFs
    geom : `~gammapy.maps.WcsGeom`, optional
        Output dataset maps geometry. The default is None, and it is derived from IRFs
    energy_axis_true : `~gammapy.maps.MapAxis`, optional
        True energy axis used for IRF maps. The default is None, and it is derived from IRFs
    name : str, optional
        Dataset name. The default is None, and the name is randomly generated.

    Returns
    -------
    dataset : `~gammapy.datasets.MapDataset`
        Map dataset.
    """
    from gammapy.makers import MapDatasetMaker
    from gammapy.datasets import MapDataset

    # define target geom
    if geom is None:
        if isinstance(data, Observation):
            geom_image = data.aeff.geom.to_image()
        elif isinstance(data, MapDataset):
            geom_image = data.exposure.geom.to_image()

        geom = geom_image.to_cube([data.edisp.edisp_map.geom.axes["energy"]])

    energy_axis = geom.axes["energy"]

    if energy_axis_true is None:
        energy_axis_true = data.edisp.edisp_map.geom.axes["energy_true"]

    # ensure that DL4 IRFs have the axes
    rad_axis = data.psf.psf_map.geom.axes["rad"]
    if data.psf.energy_name == "energy":
        geom_psf = data.psf.psf_map.geom.to_image().to_cube([rad_axis, energy_axis])
    else:
        geom_psf = data.psf.psf_map.geom.to_image().to_cube(
            [rad_axis, energy_axis_true]
        )

    geom_edisp = data.edisp.edisp_map.geom.to_image().to_cube(
        [energy_axis, energy_axis_true]
    )
    geom_exposure = geom.to_image().to_cube([energy_axis_true])

    # create dataset and run data reduction / irfs interpolation
    dataset = MapDataset.from_geoms(
        geom,
        geom_exposure=geom_exposure,
        geom_psf=geom_psf,
        geom_edisp=geom_edisp,
        name=name,
    )

    selection = ["exposure", "edisp", "psf"]
    if isinstance(data, Observation) and data.events:
        selection.append("counts")
    if isinstance(data, Observation) and data.bkg:
        selection.append("background")

    maker = MapDatasetMaker(selection=selection)
    dataset = maker.run(dataset, data)

    if isinstance(data, MapDataset) and data.counts:
        if dataset.counts.geom == data.counts.geom:
            dataset.counts.data = data.counts.data
        else:
            raise ValueError(
                "Counts geom of input MapDataset and target geom must be identical"
            )

    if not dataset.background:
        dataset.background = Map.from_geom(geom, data=0.0)

    if dataset.edisp.exposure_map and np.all(
        dataset.edisp.exposure_map.data == 0.0  # NOSONAR
        # (S1244): explicit check for exactly representable zeros
    ):
        dataset.edisp.exposure_map.quantity = dataset.psf.exposure_map.quantity

    return dataset


def create_global_dataset(
    datasets,
    name=None,
    position=None,
    binsz=None,
    width=None,
    energy_min=None,
    energy_max=None,
    energy_true_min=None,
    energy_true_max=None,
    nbin_per_decade=None,
):
    """Create an empty dataset encompassing the input datasets.

    Parameters
    ----------
    datasets : `~gammapy.datasets.Datasets`
        Datasets
    name : str, optional
        Name of the output dataset. Default is None.
    position : `~astropy.coordinates.SkyCoord`
        Center position of the output dataset.
        Default is None, and the average position is taken.
    binsz : float or tuple or list, optional
        Map pixel size in degrees.
        Default is None, the minimum bin size is taken.
    width : float or tuple or list or string, optional
        Width of the map in degrees.
        Default is None, in that case it is derived as
        the maximal width + twice the maximal separation between datasets and position.
    energy_min :  `~astropy.units.Quantity`
        Energy range.
        Default is None, the minimum energy is taken.
    energy_max :  `~astropy.units.Quantity`
        Energy range.
        Default is None, the maximum energy is taken.
    energy_true_min, energy_true_max :  `~astropy.units.Quantity`,  `~astropy.units.Quantity`
        True energy range. If None, minimum and maximum energies of all geometries is used.
        Default is None.
    nbin_per_decade : int
        number of energy bins per decade.
        Default is None, the maximum is taken.

    Returns
    -------
    datasets : `~gammapy.datasets.MapDatset`
        Empty global dataset.
    """
    from gammapy.datasets import MapDataset

    if position is None:
        frame = datasets[0].counts.geom.frame
        positions = SkyCoord(
            [d.counts.geom.center_skydir.transform_to(frame) for d in datasets]
        )
        position = SkyCoord(positions.cartesian.mean(), frame="icrs")
        position = SkyCoord(position.ra, position.dec, frame="icrs").transform_to(
            frame
        )  # drop fake distance

    binsz_list = []
    width_list = []
    energy_min_list = []
    energy_max_list = []
    energy_true_min_list = []
    energy_true_max_list = []
    nbin_per_decade_list = []
    for d in datasets:
        binsz_list.append(np.abs(d.counts.geom.pixel_scales).min())
        width_list.append(
            d.counts.geom.width.max()
            + 2 * d.counts.geom.center_skydir.separation(position)
        )
        energy_min_list.append(d.counts.geom.axes["energy"].edges.min())
        energy_max_list.append(d.counts.geom.axes["energy"].edges.max())
        energy_true_min_list.append(d.exposure.geom.axes["energy_true"].edges.min())
        energy_true_max_list.append(d.exposure.geom.axes["energy_true"].edges.max())
        ndecade = np.log10(energy_true_max_list[-1].value) - np.log10(
            energy_true_min_list[-1].value
        )
        nbins = len(d.exposure.geom.axes[0].center)
        nbin_per_decade_list.append(np.ceil(nbins / ndecade))

    if binsz is None:
        binsz = np.min(u.Quantity(binsz_list))
    binsz = _check_binsz(binsz)
    if width is None:
        width = np.max(u.Quantity(width_list))
    width = _check_width(width)
    energy_true_min = (
        energy_true_min
        if energy_true_min is not None
        else u.Quantity(energy_true_min_list).min()
    )
    if energy_true_max is None:
        energy_true_max = np.max(u.Quantity(energy_true_max_list))
    if energy_min is None:
        energy_min = np.min(u.Quantity(energy_min_list))
    if energy_max is None:
        energy_max = np.max(u.Quantity(energy_max_list))

    nbin_per_decade = (
        nbin_per_decade if nbin_per_decade is not None else np.max(nbin_per_decade_list)
    )
    energy_axis = MapAxis.from_energy_bounds(
        energy_min, energy_max, nbin_per_decade, unit="TeV", per_decade=True
    )

    geom = WcsGeom.create(
        skydir=position,
        binsz=binsz,
        width=width,
        frame=position.frame,
        proj=datasets[0].counts.geom.projection,
        axes=[energy_axis],
    )

    energy_axis_true = MapAxis.from_energy_bounds(
        energy_true_min,
        energy_true_max,
        nbin_per_decade,
        unit="TeV",
        name="energy_true",
        per_decade=True,
    )
    return MapDataset.create(geom=geom, energy_axis_true=energy_axis_true, name=name)


class set_and_restore_mask_fit:
    """Context manager to set a `mask_fit` on dataset and restore the initial mask.

    Parameters
    ----------
    datasets : `~gammapy.datasets.datasets`
        the Datasets to apply the energy mask to.
    mask_fit : `~gammapy.maps.Map`, optional
        New mask to apply.
        Default if  None.
    energy_min : `~astropy.units.Quantity`, optional
        minimum energy.
    energy_min : `~astropy.units.Quantity`, optional
        maximum energy.
    round_to_edge: bool, optional
        Whether to round `energy_min` and `energy_max` to the closest axis bin value.
        See `~gammapy.maps.MapAxis.round`. Default is False.
    operator : {`np.logical_and`, `np.logical_or`, None}, optional
        Operator to apply between to existing dataset.mask_fit and the new one.
        Default is `np.logical_and`. If None the existing mask_fit is ignored.
    """

    def __init__(
        self,
        datasets,
        mask_fit=None,
        energy_min=None,
        energy_max=None,
        round_to_edge=False,
        operator=np.logical_and,
    ):
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.round_to_edge = round_to_edge

        self.operator = operator
        self.mask_fit = mask_fit
        self.datasets = datasets
        self.mask_fits = [dataset.mask_fit for dataset in datasets]

    def __enter__(self):
        datasets = Datasets()
        for dataset in self.datasets:
            mask_fit = dataset._geom.energy_mask(
                self.energy_min, self.energy_max, self.round_to_edge
            )
            if self.mask_fit is not None:
                mask_fit &= self.mask_fit.interp_to_geom(
                    mask_fit.geom, method="nearest"
                )

            if not (self.operator is None or dataset.mask_fit is None):
                mask_fit = Map.from_geom(
                    mask_fit.geom, data=self.operator(dataset.mask_fit, mask_fit)
                )
            dataset.mask_fit = mask_fit

            if np.any(mask_fit.data):
                datasets.append(dataset)
            else:
                log.info(
                    f"Dataset {dataset.name} does not contribute in the energy range"
                )
        return datasets

    def __exit__(self, type, value, traceback):
        for dataset, mask_fit in zip(self.datasets, self.mask_fits):
            dataset.mask_fit = mask_fit
