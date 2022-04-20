# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Simulate observations"""
import multiprocessing as mp
import numpy as np
import astropy.units as u
from astropy.io import fits
from gammapy.maps import MapAxis, WcsGeom
from gammapy.utils.scripts import make_path
from gammapy.modeling.models import Models, FoVBackgroundModel


class ObservationsEventsSampler:
    """Run event sampling for an emsemble of observations

    Parameters
    ----------
    models : `~gammapy.modeling.models.Models`
        Sky-models to simulate
    binsz_min : `~astropy.quantity.Quantity`
        Minimal spatial bin size for the reference dataset
    width_max : `~astropy.quantity.Quantity`
        Minimal spatial width for the reference dataset
    nbin_per_decade_max : int
        Maximal number of bin per decade in energy for the reference dataset
    outdir : str, Path
        path of the output files created
    prefix : str
        prefix of the output files names
    n_jobs : int
        Number of processes to run in parallel
        Default is None
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`.
    overwrite : bool
        Overwrite the output files or not
    """

    def __init__(
        self,
        models=None,
        binsz_min=0.01 * u.deg,
        width_max=12 * u.deg,
        nbin_per_decade_max=30,
        outdir="./data/",
        prefix=None,
        n_jobs=1,
        random_state="random-seed",
        overwrite=True,
    ):
        if models is None:
            models = Models([])
        self.models = models
        outdir = make_path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        self.binsz_min = binsz_min
        self.width_max = width_max
        self.nbin_per_decade_max = nbin_per_decade_max
        self.outdir = outdir
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.overwrite = overwrite
        if prefix is None:
            prefix = "events"
        self.prefix = prefix

    def create_dataset(self, observation):
        """ create datasets for an emsemble of observation
    
            Parameters
            ----------
            observation : `gammapy.data.Observation`
                Observation
    
        """
        from gammapy.datasets import MapDataset

        # number of bins per decade estimated from the energy resolution
        # such as diff(ereco.edges)/ereco.center ~ min(eres)
        etrue = observation.psf.axes["energy_true"].edges  # only where psf is defined
        eres = observation.edisp.to_edisp_kernel(0 * u.deg).get_resolution(etrue)
        eres = eres[np.isfinite(eres)]
        nbin_per_decade = np.nan_to_num(
            int(np.rint(2.0 / np.min(eres.value))), nan=np.inf
        )
        nbin_per_decade = np.minimum(nbin_per_decade, self.nbin_per_decade_max)

        etrue_axis = MapAxis.from_energy_bounds(
            etrue[0],
            etrue[-1],
            nbin=nbin_per_decade,
            per_decade=True,
            name="energy_true",
        )
        energy_axis = MapAxis.from_energy_bounds(
            etrue[0], etrue[-1], nbin=nbin_per_decade
        )
        migra_axis = observation.edisp.axes["migra"]

        # bin size estimated from the minimal r68 of the psf
        psf_r68 = observation.psf.containment_radius(
            0.68, energy_true=etrue[-1], offset=0.0 * u.deg
        )
        binsz = np.nan_to_num(psf_r68, nan=-np.inf)
        binsz = np.maximum(binsz, self.binsz_min)
        # width estimated from the rad_max or the offset_max
        if observation.rad_max is not None:
            width = 2.0 * np.max(observation.rad_max)
        else:
            width = 2.0 * observation.psf.axes["offset"].edges[-1]
        binsz = np.minimum(width, self.width_max)

        geom = WcsGeom.create(
            skydir=observation.pointing_radec,
            width=(width, width),
            binsz=binsz,
            frame="icrs",
            axes=[energy_axis],
        )

        dataset = MapDataset.create(
            geom,
            energy_axis_true=etrue_axis,
            migra_axis=migra_axis,
            name=str(observation.obs_id),
        )
        return dataset

    def simulate_observation(self, observation):
        """Simulate a  single observation.

        Parameters
        ----------
        observation : `gammapy.data.Observation`
            Observation object or table
        """
        from gammapy.datasets import MapDatasetEventSampler
        from gammapy.makers import MapDatasetMaker

        dataset = self.create_dataset(observation)

        maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
        dataset = maker.run(dataset, observation)
        bkg_model = FoVBackgroundModel(dataset_name=dataset.name)

        dataset.models = Models(list(self.models) + [bkg_model])

        sampler = MapDatasetEventSampler(random_state=self.random_state)
        events = sampler.run(dataset, observation)

        primary_hdu = fits.PrimaryHDU()
        hdu_evt = fits.BinTableHDU(events.table)
        hdu_gti = fits.BinTableHDU(dataset.gti.table, name="GTI")
        hdu_all = fits.HDUList([primary_hdu, hdu_evt, hdu_gti])
        hdu_all.writeto(
            self.outdir / f"{self.prefix}_{observation.obs_id}.fits",
            overwrite=self.overwrite,
        )

    def run(self, observations):
        """Run event sampling for an ensemble of onservations
    
        Parameters
        ----------
        observations : `~gammapy.data.Observations`
            Observations object or table

        """

        if self.n_jobs > 1:
            with mp.Pool(processes=self.n_jobs) as pool:
                args = [(observation,) for observation in observations]
                pool.starmap(self.simulate_observation, args)
            pool.join()
        else:
            for observation in observations:
                self.simulate_observation(observation)
