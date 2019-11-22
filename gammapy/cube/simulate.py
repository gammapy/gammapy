# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Simulate observations"""
import astropy.units as u
from gammapy.data import Observation
from gammapy.cube import MapDatasetMaker, MapDataset
from gammapy.maps import WcsNDMap
from gammapy.modeling.models import BackgroundModel
from gammapy.utils.random import get_random_state

__all__ = ["simulate_dataset"]


def simulate_dataset(
    skymodel,
    geom,
    pointing,
    irfs,
    livetime=1 * u.h,
    offset_max=2.0*u.deg,
    selection = ["exposure", "background", "psf", "edisp"],
    random_state="random-seed",
):
    """Simulate a 3D dataset.

    Simulate a source defined with a sky model for a given pointing,
    geometry and irfs for a given exposure time.
    This will return a MapDataset object which includes the counts cube,
    exposure cube, psf cube, energy dispersion cube, background model and the sky model.

    Parameters
    ----------
    skymodel : `~gammapy.modeling.models.SkyModel`
        The skymodel to simulate the data
    geom : `~gammapy.maps.WcsGeom`
        Geometry object for the observation
    pointing : `~astropy.coordinates.SkyCoord`
        Pointing position
    irfs : dict
        Irfs used for simulating the observation
    livetime : `~astropy.units.Quantity`
        Livetime exposure of the simulated observation
    offset_max : `~astropy.units.Quantity`
        FoV offset cut
    selection: list
        List of str, selecting which IRFs to use.
        Available: 'exposure', 'background', 'psf', 'edisp'
        By default, all are made.
    random_state: {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.

    Returns
    -------
    dataset : `~gammapy.cube.MapDataset`
        A dataset of the simulated observation.
    """

    obs = Observation.create(pointing=pointing, livetime=livetime, irfs=irfs)
    empty = MapDataset.create(geom)
    maker = MapDatasetMaker(offset_max=offset_max)
    dataset = maker.run(empty, obs, selection=selection)
    dataset.model = skymodel
    dataset.fake(random_state=random_state)
    return dataset