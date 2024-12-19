# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.coordinates import SkyCoord
from gammapy.maps import Map
from gammapy.data.observations import create_observation_from_fermi_files
from gammapy.modeling.models.utils import cutout_template_models
from gammapy.modeling.models import (
    create_fermi_isotropic_diffuse_model,
    Models,
    FoVBackgroundModel,
)
from . import Datasets

__all__ = [
    "apply_edisp",
    "split_dataset",
    "create_map_dataset_from_dl4_irfs",
    "create_map_dataset_from_fermi_files",
]


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
        shape : (np.int64(50), np.int64(50), 3)
        ndim  : 3
        unit  :
        dtype : float64
    <BLANKLINE>
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


def create_map_dataset_from_dl4_irfs(obs, geom=None, energy_axis_true=None, name=None):
    """Create a map dataset from an observation containing DL4 IRFs


    Parameters
    ----------
    obs : `~gammapy.data.Observation`
        Observation containing DL4 IRFs
    geom : `~gammapy.maps.WcsGeom`, optional
        Output dataset maps geometry. The default is None, and it is derived from IRFS
    energy_axis_true : `~gammapy.maps.MapAxis`, optional
        True energy axis used for IRF maps. The default is None, and it is derived from IRFS
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
        geom_image = obs.aeff.geom.to_image()
        geom = geom_image.to_cube([obs.edisp.edisp_map.geom.axes["energy"]])

    energy_axis = geom.axes["energy"]

    if energy_axis_true is None:
        # energy_axis_true = obs.edisp.edisp_map.geom.axes["energy_true"]
        energy_axis_true = obs.aeff.geom.axes["energy_true"]

    # ensure that DL4 IRFs have the axes
    rad_axis = obs.psf.psf_map.geom.axes["rad"]
    geom_psf = obs.psf.psf_map.geom.to_image().to_cube([rad_axis, energy_axis_true])
    geom_edisp = obs.edisp.edisp_map.geom.to_image().to_cube(
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
    if obs.events:
        selection.append("counts")
    if obs.bkg:
        selection.append("background")

    maker = MapDatasetMaker(selection=selection)
    dataset = maker.run(dataset, obs)

    if "background" not in selection:
        dataset.background = Map.from_geom(geom, data=0.0)

    return dataset


def add_fermi_isotropic_model(dataset, isotropic_filepath, as_bkg=True):
    """Add Fermi-LAT isotropic model to dataset


    Parameters
    ----------
    dataset : `~gammapy.datasets.MapDataset`
        Map dataset.
    isotropic_filepath : str
        Isotropic file path.
    as_bkg : bool, optional
        If True set the isotropic as background model instead of a SkyModel.
        The default is True.
    """

    if dataset.models is None:
        models = Models()

    diffuse_iso = create_fermi_isotropic_diffuse_model(
        filename=isotropic_filepath, interp_kwargs={"extrapolate": True}
    )
    diffuse_iso.apply_irf["edisp"] = False

    dataset.models = models + [diffuse_iso]

    if as_bkg:
        dataset.background = dataset.npred()
        dataset.models = models + [FoVBackgroundModel(dataset_name=dataset.name)]


def create_map_dataset_from_fermi_files(
    path,
    isotropic_filename,
    events_filename="ft1_00.fits",
    counts_filename="ccube_00.fits",
    exposure_filename="bexpmap_roi_00.fits",
    psf_filename="psf_00.fits",
    drm_filename="drm_00.fits",
    geom=None,
    energy_axis_true=None,
    name=None,
    isotropic_as_bkg=True,
):
    """


    Parameters
    ----------
    path : TYPE
        DESCRIPTION.
    isotropic_filename : str
        Isotropic file name.
    events_filename : str
        Events filename. Default is ft1_00.fits
    counts_filename : str, optional
        Counts filename. Default is ccube_00.fits
    exposure_filename : str, optional
        exposure filename. Default is bexpmap_roi_00.fits
    psf_filename : str, optional
        PSF filename. Default is psf_00.fits
    drm_filename : str, optional
        DRM filename. Default is drm_00.fits
    obs : `~gammapy.data.Observation`
        Observation with DL4 IRFs
    geom : `~gammapy.maps.WcsGeom`, optional
        Output dataset maps geometry. The default is None, and it is derived from IRFS
    energy_axis_true : `~gammapy.maps.MapAxis`, optional
        True energy axis used for IRF maps. The default is None, and it is derived from IRFS
    name : str, optional
        Dataset name. The default is None, and the name is randomly generated.
    isotropic_as_bkg : bool, optional
        If True set the isotropic as background model instead of a SkyModel.
        The default is True.

    Returns
    -------
    dataset : `~gammapy.datasets.MapDataset`
        Map dataset.

    """

    obs = create_observation_from_fermi_files(
        path=path,
        events_filename=events_filename,
        counts_filename=counts_filename,
        exposure_filename=exposure_filename,
        psf_filename=psf_filename,
        drm_filename=drm_filename,
    )

    dataset = create_map_dataset_from_dl4_irfs(obs, geom, energy_axis_true, name)
    add_fermi_isotropic_model(dataset, isotropic_filename, isotropic_as_bkg)
    return dataset
