# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Simulate observations"""
import astropy.units as u
from gammapy.data import Observation
from gammapy.cube import MapDatasetMaker, MapDataset


__all__ = ["simulate_map"]


def simulate_map(
    skymodel,
    geom,
    pointing,
    irfs,
    livetime=1 * u.h,
    offset_max=2.0 * u.deg,
    selection=["exposure", "background", "psf", "edisp"],
    background_pars={"norm": 1.0, "tilt": 0.0},
    random_state="random-seed",
    **kwargs,
):
    """Simulate a 3D dataset.

    Simulate a source defined with a sky model for a given pointing,
    geometry and irfs for a given exposure time.
    This will return a MapDataset object which includes the counts cube,
    exposure cube, psf cube, energy dispersion cube, background model and the sky model.
    By default, a background model will be made from the IRFs. The user can also pass some
    other model if desired.

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
    selection: list, optional
        List of str, selecting which IRFs to use.
        Available: 'exposure', 'background', 'psf', 'edisp'
        By default, all are made.
    random_state: {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
    background pars: dict
        Dictionary specifying an additional norm and tilt in the background
    **kwargs: additional default `MapAxis` to pass to `MapDataset.create`

    Returns
    -------
    dataset : `~gammapy.cube.MapDataset`
        A dataset of the simulated observation.
    """

    obs = Observation.create(pointing=pointing, livetime=livetime, irfs=irfs)
    empty = MapDataset.create(geom, **kwargs)
    maker = MapDatasetMaker(offset_max=offset_max)
    dataset = maker.run(empty, obs, selection=selection)
    dataset.psf = dataset.psf.get_psf_kernel(
        position=pointing, geom=geom, max_radius="0.3 deg",
    )
    dataset.background_model.parameters["norm"].value = background_pars["norm"]
    dataset.background_model.parameters["tilt"].value = background_pars["tilt"]
    dataset.model = skymodel
    dataset.fake(random_state=random_state)
    return dataset
