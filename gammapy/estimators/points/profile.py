# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tools to create profiles (i.e. 1D "slices" from 2D images)."""
from astropy import units as u
from regions import CircleAnnulusSkyRegion
from gammapy.datasets import Datasets
from gammapy.maps import MapAxis
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
from .core import FluxPoints
from .sed import FluxPointsEstimator

__all__ = ["FluxProfileEstimator"]


class FluxProfileEstimator(FluxPointsEstimator):
    """Estimate flux profiles

    Parameters
    ----------
    regions : list of `~regions.SkyRegion`
        regions to use
    spectrum : `~gammapy.modeling.models.SpectralModel` (optional)
        Spectral model to compute the fluxes or brightness.
        Default is power-law with spectral index of 2.
    **kwargs : dict
        Keywords forwarded to the `FluxPointsEstimator` (see documentation
        there for further description of valid keywords)

    Examples
    --------
    This example shows how to compute a counts profile for the Fermi galactic
    center region::

    >>> from astropy import units as u
    >>> from astropy.coordinates import SkyCoord
    >>> from gammapy.data import GTI
    >>> from gammapy.estimators import FluxProfileEstimator
    >>> from gammapy.utils.regions import make_orthogonal_rectangle_sky_regions
    >>> from gammapy.datasets import MapDataset
    >>> from gammapy.maps import RegionGeom

    >>> # load example data
    >>> filename = "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc.fits.gz"
    >>> dataset = MapDataset.read(filename, name="fermi-dataset")

    >>> # configuration
    >>> dataset.gti = GTI.create("0s", "1e7s", "2010-01-01")

    >>> # creation of the boxes and axis
    >>> start_pos = SkyCoord("-1d", "0d", frame='galactic')
    >>> end_pos = SkyCoord("1d", "0d", frame='galactic')

    >>> regions = make_orthogonal_rectangle_sky_regions(
                start_pos=start_pos,
                end_pos=end_pos,
                wcs=dataset.counts.geom.wcs,
                height=2 * u.deg,
                nbin=21
            )

    >>> # set up profile estimator and run
    >>> prof_maker = FluxProfileEstimator(regions=regions, energy_edges=[10, 2000] * u.GeV)
    >>> fermi_prof = prof_maker.run(dataset)
    >>> print(fermi_prof)
    FluxPoints
    ----------
    <BLANKLINE>
      geom                   : RegionGeom
      axes                   : ['lon', 'lat', 'energy', 'projected-distance']
      shape                  : (1, 1, 1, 21)
      quantities             : ['norm', 'norm_err', 'ts', 'npred', 'npred_excess', 'stat', 'counts', 'success']  # noqa: E501
      ref. model             : pl
      n_sigma                : 1
      n_sigma_ul             : 2
      sqrt_ts_threshold_ul   : 2
      sed type init          : likelihood

    """

    tag = "FluxProfileEstimator"

    def __init__(self, regions, spectrum=None, **kwargs):
        if len(regions) <= 1:
            raise ValueError(
                "Please provide at least two regions for flux profile estimation."
            )

        self.regions = regions

        if spectrum is None:
            spectrum = PowerLawSpectralModel()

        self.spectrum = spectrum
        super().__init__(**kwargs)

    @property
    def projected_distance_axis(self):
        """Get projected distance from the first region.

        For normal region this is defined as the distance from the
        center of the region. For annulus shaped regions it is the
        mean between the inner and outer radius.

        Returns
        -------
        axis : `MapAxis`
            Projected distance axis
        """
        distances = []
        center = self.regions[0].center

        for idx, region in enumerate(self.regions):
            if isinstance(region, CircleAnnulusSkyRegion):
                distance = (region.inner_radius + region.outer_radius) / 2.0
            else:
                distance = center.separation(region.center)

            distances.append(distance)

        return MapAxis.from_nodes(
            u.Quantity(distances, "deg"), name="projected-distance"
        )

    def run(self, datasets):
        """Run flux profile estimation

        Parameters
        ----------
        datasets : list of `~gammapy.datasets.MapDataset`
            Map datasets.

        Returns
        -------
        profile : `~gammapy.estimators.FluxPoints`
            Profile flux points.
        """
        datasets = Datasets(datasets=datasets)

        maps = []

        for region in self.regions:
            datasets_to_fit = datasets.to_spectrum_datasets(region=region)
            datasets_to_fit.models = SkyModel(self.spectrum, name="test-source")
            fp = super().run(datasets_to_fit)
            maps.append(fp)

        return FluxPoints.from_stack(
            maps=maps,
            axis=self.projected_distance_axis,
        )

    @property
    def config_parameters(self):
        """Config parameters"""
        pars = self.__dict__.copy()
        pars = {key.strip("_"): value for key, value in pars.items()}
        pars.pop("regions")
        return pars
