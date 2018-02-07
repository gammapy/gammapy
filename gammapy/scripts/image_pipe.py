# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.units import Quantity
from astropy.table import Table
from astropy.coordinates import Angle
from astropy.convolution import Tophat2DKernel
from ..utils.energy import EnergyBounds
from ..stats import significance
from ..background import fill_acceptance_image
from ..image import SkyImage, SkyImageList

__all__ = ['SingleObsImageMaker', 'StackedObsImageMaker']


class SingleObsImageMaker(object):
    """Compute images for one observation.

    The computed images are stored in a ``images`` attribute of
    type `~gammapy.image.SkyImageList` with the following keys:

    * ``counts`` : Counts
    * ``bkg`` : Background model
    * ``exposure`` : Exposure
    * ``excess`` : Excess
    * ``significance`` : Significance

    Parameters
    ----------
    obs : `~gammapy.data.DataStoreObservation`
        Observation data
    empty_image : `~gammapy.image.SkyImage`
        Reference image
    energy_band : `~gammapy.utils.energy.Energy`
        Energy band selection
    offset_band : `astropy.coordinates.Angle`
        Offset band selection
    exclusion_mask : `~gammapy.image.SkyImage`
        Exclusion mask
    ncounts_min : int
        Minimum counts required for the observation (TODO: used how?)
    save_bkg_scale: bool
        True if you want to save the normalisation of the bkg computed outside the exlusion region in a Table
    """

    def __init__(self, obs, empty_image,
                 energy_band, offset_band, exclusion_mask=None, ncounts_min=0, save_bkg_scale=True):
        # Select the events in the given energy and offset range
        self.energy_band = energy_band
        self.offset_band = offset_band
        events = obs.events
        self.obs_id = events.table.meta["OBS_ID"]
        events = events.select_energy(self.energy_band)
        self.events = events.select_offset(self.offset_band)

        self.images = SkyImageList()
        self.empty_image = empty_image
        self.header = self.empty_image.to_image_hdu().header
        if exclusion_mask:
            exclusion_mask.name = 'exclusion'
            self.images['exclusion'] = exclusion_mask

        self.ncounts_min = ncounts_min
        self.aeff = obs.aeff
        self.edisp = obs.edisp
        self.psf = obs.psf
        self.bkg = obs.bkg
        self.obs_center = obs.pointing_radec
        self.livetime = obs.observation_live_time_duration
        self.save_bkg_scale = save_bkg_scale
        if self.save_bkg_scale:
            self.table_bkg_scale = Table(names=["OBS_ID", "bkg_scale", "N_counts"])

    def counts_image(self):
        """Fill the counts image for the events of one observation."""
        self.images['counts'] = SkyImage.empty_like(self.empty_image, name='counts')

        if len(self.events.table) > self.ncounts_min:
            self.images['counts'].fill_events(self.events)

    def bkg_image(self, bkg_norm=True):
        """
        Make the background image for one observation from a bkg model.

        Parameters
        ----------
        bkg_norm : bool
            If true, apply the scaling factor from the number of counts
            outside the exclusion region to the bkg image
        """
        bkg_image = SkyImage.empty_like(self.empty_image)
        table = self.bkg.acceptance_curve_in_energy_band(energy_band=self.energy_band)
        center = self.obs_center.galactic
        bkg_hdu = fill_acceptance_image(self.header, center, table["offset"], table["Acceptance"], self.offset_band[1], self.offset_band[0])
        bkg_image.data = Quantity(bkg_hdu.data, table["Acceptance"].unit) * bkg_image.solid_angle() * self.livetime
        bkg_image.data = bkg_image.data.decompose()
        bkg_image.data = bkg_image.data.value
        if bkg_norm:
            scale, counts = self.background_norm_factor(self.images["counts"], bkg_image)
            bkg_image.data = scale * bkg_image.data
            if self.save_bkg_scale:
                self.table_bkg_scale.add_row([self.obs_id, scale, counts])

        self.images["bkg"] = bkg_image

    def make_1d_expected_counts(self, spectral_index=2.3, for_integral_flux=False, eref=None):
        """Compute the 1D exposure table for one observation for an offset table.

        Parameters
        ----------
        spectral_index : float
            Assumed power-law spectral index
        for_integral_flux : bool
            True if you want that the total excess / exposure gives the integrated flux
        eref: `~gammapy.utils.energy.Energy`
            Reference energy at which you want to compute the exposure. Default is the log center of the energy band of
             the image.
        Returns
        -------
        table : `astropy.table.Table`
            Two columns: offset in the FOV "theta" and expected counts "npred"
        """
        energy = EnergyBounds.equal_log_spacing(self.energy_band[0].value, self.energy_band[1].value, 100,
                                                self.energy_band.unit)
        energy_band = energy.bands
        energy_bin = energy.log_centers
        if not eref:
            eref = EnergyBounds(self.energy_band).log_centers
        spectrum = (energy_bin / eref) ** (-spectral_index)
        offset = Angle(np.linspace(self.offset_band[0].value, self.offset_band[1].value, 10), self.offset_band.unit)
        arf = self.aeff.data.evaluate(offset=offset, energy=energy_bin).T
        npred = np.sum(arf * spectrum * energy_band, axis=1)
        npred *= self.livetime

        if for_integral_flux:
            norm = np.sum(spectrum * energy_band)
            npred /= norm

        table = Table()
        table['theta'] = offset
        table['npred'] = npred

        return table

    def exposure_image(self, spectral_index=2.3, for_integral_flux=False):
        r"""Compute the exposure image for one observation.

        Excess/exposure will give the differential flux at the energy Eref at the middle of the ``self.energy_band``

        If ``for_integral_flux`` is true, it will give the integrated flux over the ``self.energy_band``

        Exposure is define as follow:

        .. math ::

            EXPOSURE = \int_{E_1}^{E_2} A(E) \phi(E) * T \, dE

        with ``T`` the observation livetime, ``A(E)`` the effective area,
        the energy integration range :math:`[E_1,E_2]` given by ``self.energy_range``
        and assuming a power law for the flux :math:`\phi(E) = \phi_{Eref} \times \frac{E}{E_{ref}}^{\gamma}`
        with :math:`\gamma` the spectral index of the assumed power law.

        If ``for_integral_flux`` is true,
        :math:`EXPOSURE = \int_{E_1}^{E_2} A(E) \phi_{E} * T \, dE / \int \phi_{E} \, dE`

        Parameters
        ----------
        spectral_index : float
            Assumed power-law spectral index
        for_integral_flux : bool
            True if you want that the total excess / exposure gives the integrated flux
        """
        from scipy.interpolate import interp1d
        # TODO: should be re-implemented using the make_exposure_cube function
        table = self.make_1d_expected_counts(spectral_index, for_integral_flux)
        exposure = SkyImage.empty_like(self.empty_image, unit=table["npred"].unit)

        # calculate pixel offset from center (in world coordinates)
        coord = exposure.coordinates()
        offset = coord.separation(self.obs_center)

        # Interpolate for the offset of each pixel
        f = interp1d(table["theta"], table["npred"], bounds_error=False, fill_value=0)
        exposure.data = f(offset)
        exposure.data[offset >= self.offset_band[1]] = 0
        self.images["exposure"] = exposure

    def background_norm_factor(self, counts, bkg):
        """Determine the scaling factor to apply to the background image.

        Compares the events in the counts images and the bkg image outside the exclusion images.

        Parameters
        ----------
        counts : `~gammapy.image.SkyImage`
            counts image
        bkg : `~gammapy.image.SkyImage`
            bkg image

        Returns
        -------
        scale : float
            scaling factor between the counts and the bkg images outside the exclusion region.
        """
        counts_sum = np.sum(counts.data * self.images['exclusion'].data)
        bkg_sum = np.sum(bkg.data * self.images['exclusion'].data)
        scale = counts_sum / bkg_sum

        return scale, counts_sum

    def significance_image(self, radius):
        """Make the significance image from the counts and bkg images.

        Parameters
        ----------
        radius : float
            Disk radius in pixels.
        """
        image = SkyImage.empty_like(self.empty_image)
        disk = Tophat2DKernel(radius)
        disk.normalize('peak')
        counts = self.images["counts"].convolve(disk.array)
        bkg = self.images["bkg"].convolve(disk.array)
        image.data = significance(counts.data, bkg.data)
        self.images["significance"] = image

    def excess_image(self):
        """Compute excess between counts and bkg image."""
        total_excess = SkyImage.empty_like(self.empty_image)
        total_excess.data = self.images["counts"].data - self.images["bkg"].data
        self.images["excess"] = total_excess


class StackedObsImageMaker(object):
    """Compute stacked images for many observations.

    The computed images are stored in a ``images`` attribute of
    type `~gammapy.image.SkyImageList` with the following keys:

    * ``counts`` : Counts
    * ``bkg`` : Background model
    * ``exposure`` : Exposure
    * ``excess`` : Excess
    * ``significance`` : Significance

    Parameters
    ----------
    empty_image : `~gammapy.image.SkyImage`
        Reference image
    energy_band : `~gammapy.utils.energy.Energy`
        Energy band selection
    offset_band : `astropy.coordinates.Angle`
        Offset band selection
    data_store : `~gammapy.data.DataStore`
        Data store
    obs_table : `~astropy.table.Table`
        Required columns: OBS_ID
    exclusion_mask : `~gammapy.image.SkyImage`
        Exclusion mask
    ncounts_min : int
        Minimum counts required for the observation
    save_bkg_scale: bool
        True if you want to save the normalisation of the bkg for each run in a `Table` table_bkg_norm with two columns:
         "OBS_ID" and "bkg_scale"
    """

    def __init__(self, empty_image=None, energy_band=None, offset_band=None,
                 data_store=None, obs_table=None, exclusion_mask=None, ncounts_min=0, save_bkg_scale=True):

        self.images = SkyImageList()

        self.data_store = data_store
        self.obs_table = obs_table
        self.energy_band = energy_band
        self.offset_band = offset_band

        self.empty_image = empty_image
        self.header = self.empty_image.to_image_hdu().header
        self.exclusion_mask = exclusion_mask
        if exclusion_mask:
            exclusion_mask.name = 'exclusion'
            self.images['exclusion'] = exclusion_mask
        self.ncounts_min = ncounts_min
        self.psfmeantab = None
        self.thetapsf = None
        self.save_bkg_scale = save_bkg_scale
        if self.save_bkg_scale:
            self.table_bkg_scale = Table(names=["OBS_ID", "bkg_scale", "N_counts"])

    def make_images(self, make_background_image=False, bkg_norm=True,
                    spectral_index=2.3, for_integral_flux=False, radius=10):
        """Compute the counts, bkg, exposure, excess and significance images for a set of observation.

        Parameters
        ----------
        make_background_image : bool
            True if you want to compute the background and exposure images
        bkg_norm : bool
            If true, apply the scaling factor to the bkg image
        spectral_index : float
            Assumed power-law spectral index
        for_integral_flux : bool
            True if you want that the total excess / exposure gives the integrated flux
        radius : float
            Disk radius in pixels for the significance image
        """
        total_counts = SkyImage.empty_like(self.empty_image, name='counts')

        if make_background_image:
            total_bkg = SkyImage.empty_like(self.empty_image, name='bkg')
            total_exposure = SkyImage.empty_like(self.empty_image, name='exposure')

        for obs_id in self.obs_table['OBS_ID']:
            obs = self.data_store.obs(obs_id)
            obs_image = SingleObsImageMaker(obs, self.empty_image, self.energy_band, self.offset_band,
                                            self.exclusion_mask, self.ncounts_min)
            if len(obs_image.events.table) <= self.ncounts_min:
                continue
            else:
                obs_image.counts_image()
                total_counts.data += obs_image.images['counts'].data
                if make_background_image:
                    obs_image.bkg_image(bkg_norm)
                    if self.save_bkg_scale:
                        self.table_bkg_scale.add_row(obs_image.table_bkg_scale[0])
                    obs_image.exposure_image(spectral_index, for_integral_flux)
                    total_bkg.data += obs_image.images['bkg'].data
                    total_exposure.data += obs_image.images['exposure'].data

        self.images['counts'] = total_counts

        if make_background_image:
            self.images['bkg'] = total_bkg
            self.images['exposure'] = total_exposure
            self.significance_image(radius)
            self.excess_image()

    def significance_image(self, radius):
        """Make the significance image from the counts and bkg images.

        Parameters
        ----------
        radius : float
            Disk radius in pixels.
        """
        image = SkyImage.empty_like(self.empty_image, name='significance')

        disk = Tophat2DKernel(radius)
        disk.normalize('peak')
        counts = self.images["counts"].convolve(disk.array)
        bkg = self.images["bkg"].convolve(disk.array)
        image.data = significance(counts.data, bkg.data)

        self.images['significance'] = image

    def excess_image(self):
        """Compute excess between counts and bkg image."""
        total_excess = SkyImage.empty_like(self.empty_image, name='excess')
        total_excess.data = self.images['counts'].data - self.images['bkg'].data
        self.images['excess'] = total_excess
