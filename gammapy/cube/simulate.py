# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Simulate observations"""
import astropy.units as u
from ..cube import MapDataset, PSFKernel
from ..cube import make_map_exposure_true_energy, make_map_background_irf
from ..maps import WcsNDMap
from ..cube.models import BackgroundModel
from ..utils.random import get_random_state

__all__ = ["simulate_dataset"]


def simulate_dataset(
    skymodel,
    geom,
    pointing,
    irfs,
    livetime=1 * u.h,
    offset=0 * u.deg,
    max_radius=0.8 * u.deg,
    random_state="random-seed",
):
    """Simulate a 3D dataset.
    
    Simulate a source defined with a sky model for a given pointing,
    geometry and irfs for a given exposure time.
    This will return a dataset object which includes the counts cube,
    the exposure cube, the psf cube, the background model and the sky model.

    Parameters
    ----------
    skymodel : `~gammapy.cube.models.SkyModel`
        Background model map
    geom : `~gammapy.maps.WcsGeom`
        Geometry object for the observation
    pointing : `~astropy.coordinates.SkyCoord`
        Pointing position
    irfs : dict
        Irfs used for simulating the observation
    livetime : `~astropy.units.Quantity`
        Livetime exposure of the simulated observation
    offset : `~astropy.units.Quantity`
        Offset from the center of the pointing position.
        This is used for the PSF and Edisp estimation
    max_radius : `~astropy.coordinates.Angle`
        The maximum radius of the PSF kernel.
    random_state: {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.

    Returns
    -------
    dataset : `~gammapy.cube.MapDataset`
        A dataset of the simulated observation.
    """
    background = make_map_background_irf(
        pointing=pointing, ontime=livetime, bkg=irfs["bkg"], geom=geom
    )

    background_model = BackgroundModel(background)

    psf = irfs["psf"].to_energy_dependent_table_psf(theta=offset)
    psf_kernel = PSFKernel.from_table_psf(psf, geom, max_radius=max_radius)

    exposure = make_map_exposure_true_energy(
        pointing=pointing, livetime=livetime, aeff=irfs["aeff"], geom=geom
    )

    if "edisp" in irfs:
        energy = geom.axes[0].edges
        edisp = irfs["edisp"].to_energy_dispersion(offset, e_reco=energy, e_true=energy)
    else:
        edisp = None

    dataset = MapDataset(
        model=skymodel,
        exposure=exposure,
        background_model=background_model,
        psf=psf_kernel,
        edisp=edisp,
    )

    npred_map = dataset.npred()
    rng = get_random_state(random_state)
    counts = rng.poisson(npred_map.data)
    dataset.counts = WcsNDMap(geom, counts)

    return dataset
