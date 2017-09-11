# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from astropy.units import Quantity
from astropy import units as u
from astropy.table import Table
from astropy.convolution import Tophat2DKernel
from ..utils.energy import Energy, EnergyBounds
from ..stats import significance
from ..irf import TablePSF
from ..background import fill_acceptance_image
from ..cube import SkyCube
from .exposure import make_exposure_cube

__all__ = [
    'SingleObsCubeMaker',
    'StackedObsCubeMaker',
]

log = logging.getLogger(__name__)


class SingleObsCubeMaker(object):
    """Compute `~gammapy.cube.SkyCube` images for one observation.

    The computed cubes are stored in a `~gammapy.cube.SkyCube` with the following name:

    * ``counts_cube`` : Counts
    * ``bkg_cube`` : Background model
    * ``exposure_cube`` : Exposure
    * ``excess_cube`` : Excess
    * ``significance_cube`` : Significance

    Parameters
    ----------
    obs : `~gammapy.data.DataStoreObservation`
        Observation data
    empty_cube_images : `~gammapy.cube.SkyCube`
        Reference Cube for images in reco energy
    empty_exposure_cube : `~gammapy.cube.SkyCube`
        Reference Cube for exposure in true energy
    offset_band : `~astropy.coordinates.Angle`
        Offset band selection
    exclusion_mask : `~gammapy.cube.SkyCube`
        Exclusion mask
    save_bkg_scale: bool
        True if you want to save the normalisation of the bkg
        computed outside the exclusion region in a Table
    """

    def __init__(self, obs, empty_cube_images, empty_exposure_cube,
                 offset_band, exclusion_mask=None, save_bkg_scale=True):
        self.energy_reco = empty_cube_images.energies(mode="edges")
        self.offset_band = offset_band
        self.counts_cube = SkyCube.empty_like(empty_cube_images)
        self.bkg_cube = SkyCube.empty_like(empty_cube_images)
        self.significance_cube = SkyCube.empty_like(empty_cube_images)
        self.excess_cube = SkyCube.empty_like(empty_cube_images)
        self.exposure_cube = SkyCube.empty_like(empty_exposure_cube, unit="m2 s")

        self.obs_id = obs.obs_id
        events = obs.events
        self.events = events.select_offset(self.offset_band)

        self.header = empty_cube_images.sky_image_ref.to_image_hdu().header
        self.cube_exclusion_mask = exclusion_mask
        self.aeff = obs.aeff
        self.edisp = obs.edisp
        self.psf = obs.psf
        self.bkg = obs.bkg
        self.obs_center = obs.pointing_radec
        self.livetime = obs.observation_live_time_duration
        self.save_bkg_scale = save_bkg_scale
        if self.save_bkg_scale:
            self.table_bkg_scale = Table(names=["OBS_ID", "bkg_scale"])

    def make_counts_cube(self):
        """Fill the counts cube for the events of one observation."""
        self.counts_cube.fill_events(self.events)

    def make_bkg_cube(self, bkg_norm=True):
        """Make the background image for one observation from a bkg model.

        Parameters
        ----------
        bkg_norm : bool
            If true, apply the scaling factor from the number of counts
            outside the exclusion region to the bkg image
        """
        for idx_energy in range(len(self.energy_reco) - 1):
            energy_band = Energy(
                [self.energy_reco[idx_energy].value, self.energy_reco[idx_energy + 1].value],
                self.energy_reco.unit)
            table = self.bkg.acceptance_curve_in_energy_band(
                energy_band=energy_band)
            center = self.obs_center.galactic
            bkg_hdu = fill_acceptance_image(self.header, center,
                                            table["offset"],
                                            table["Acceptance"],
                                            self.offset_band[1])
            bkg_image = (
                Quantity(bkg_hdu.data, table["Acceptance"].unit) *
                self.bkg_cube.sky_image_ref.solid_angle() *
                self.livetime
            )
            self.bkg_cube.data[idx_energy, :, :] = bkg_image.decompose().value

        if bkg_norm:
            scale = self.background_norm_factor()
            self.bkg_cube.data = scale * self.bkg_cube.data
            if self.save_bkg_scale:
                self.table_bkg_scale.add_row([self.obs_id, scale])

    def background_norm_factor(self):
        """Determine background scaling factor.

        Compares the events in the counts cube and the bkg cube
        outside the exclusion regions.

        Returns
        -------
        scale : float
            scaling factor between the counts and the bkg images outside the exclusion region.
        """
        counts_sum = np.sum(self.counts_cube.data * self.cube_exclusion_mask.data)
        bkg_sum = np.sum(self.bkg_cube.data * self.cube_exclusion_mask.data)
        scale = counts_sum / bkg_sum

        return scale

    def make_exposure_cube(self):
        """Compute the exposure cube."""
        self.exposure_cube = make_exposure_cube(
            pointing=self.obs_center,
            livetime=self.livetime,
            aeff=self.aeff,
            ref_cube=self.exposure_cube,
            offset_max=self.offset_band[1],
        )

    def make_significance_cube(self, radius):
        """Make the significance cube from the counts and bkg cubes.

        Parameters
        ----------
        radius : float
            Disk radius in pixels.
        """
        disk = Tophat2DKernel(radius)
        disk.normalize('peak')
        list_kernel = [disk.array] * (len(self.significance_cube.energies()))
        counts = self.counts_cube.convolve(list_kernel)
        bkg = self.bkg_cube.convolve(list_kernel)
        self.significance_cube.data = significance(counts.data, bkg.data)

    def make_excess_cube(self):
        """Compute excess cube between counts and bkg cubes."""
        self.excess_cube.data = self.counts_cube - self.bkg_cube


class StackedObsCubeMaker(object):
    """Compute stacked cubes for many observations.

    The computed cubes are stored in a `~gammapy.cube.SkyCube`
    with the following name:

    * ``counts_cube`` : Counts
    * ``bkg_cube`` : Background model
    * ``exposure_cube`` : Exposure
    * ``excess_cube`` : Excess
    * ``significance_cube`` : Significance

    Parameters
    ----------
    empty_cube_images : `~gammapy.cube.SkyCube`
        Reference Cube for images in reco energy
    empty_exposure_cube : `~gammapy.cube.SkyCube`
        Reference Cube for exposure in true energy
    offset_band : `astropy.coordinates.Angle`
        Offset band selection of the events to fill the cubes
    data_store : `~gammapy.data.DataStore`
        Data store
    obs_table : `~astropy.table.Table`
        Required columns: OBS_ID
    exclusion_mask : `~gammapy.cube.SkyCube`
        Exclusion mask
    save_bkg_scale : bool
        True if you want to save the normalisation of the bkg for each run
        in a table table_bkg_norm with two columns:
        "OBS_ID" and "bkg_scale"
    """

    def __init__(self, empty_cube_images, empty_exposure_cube=None,
                 offset_band=None, data_store=None, obs_table=None,
                 exclusion_mask=None, save_bkg_scale=True):

        self.empty_cube_images = empty_cube_images
        if not empty_exposure_cube:
            self.empty_exposure_cube = SkyCube.empty_like(empty_cube_images)
        else:
            self.empty_exposure_cube = empty_exposure_cube

        self.counts_cube = SkyCube.empty_like(empty_cube_images)
        self.bkg_cube = SkyCube.empty_like(empty_cube_images)
        self.significance_cube = SkyCube.empty_like(empty_cube_images)
        self.excess_cube = SkyCube.empty_like(empty_cube_images)
        self.exposure_cube = SkyCube.empty_like(empty_exposure_cube, unit="m2 s")

        self.data_store = data_store
        self.obs_table = obs_table
        self.offset_band = offset_band

        self.header = empty_cube_images.sky_image_ref.to_image_hdu().header
        self.cube_exclusion_mask = exclusion_mask

        self.save_bkg_scale = save_bkg_scale
        if self.save_bkg_scale:
            self.table_bkg_scale = Table(names=["OBS_ID", "bkg_scale"])

    def make_cubes(self, make_background_image=False, bkg_norm=True, radius=10):
        """Compute all cubes for a given set of observation.

        Cubes computed: total counts, bkg, exposure, excess, significance.

        Parameters
        ----------
        make_background_image : bool
            True if you want to compute the background and exposure images
        bkg_norm : bool
            If true, apply the scaling factor to the bkg image
        radius : float
            Disk radius in pixels for the significance image
        """
        for obs_id in self.obs_table['OBS_ID']:
            obs = self.data_store.obs(obs_id)
            cube_images = SingleObsCubeMaker(obs=obs,
                                             empty_cube_images=self.empty_cube_images,
                                             empty_exposure_cube=self.empty_exposure_cube,
                                             offset_band=self.offset_band,
                                             exclusion_mask=self.cube_exclusion_mask,
                                             save_bkg_scale=self.save_bkg_scale)
            cube_images.make_counts_cube()
            self.counts_cube.data += cube_images.counts_cube.data

            if make_background_image:
                cube_images.make_bkg_cube(bkg_norm)
                if self.save_bkg_scale:
                    self.table_bkg_scale.add_row(
                        cube_images.table_bkg_scale[0])
                cube_images.make_exposure_cube()
                self.bkg_cube.data += cube_images.bkg_cube.data
                self.exposure_cube.data += cube_images.exposure_cube.data.to("m2 s")

        if make_background_image:
            self.make_significance_cube(radius)
            self.make_excess_cube()

    def make_significance_cube(self, radius):
        """Make the significance cube from the counts and bkg cubes.

        Parameters
        ----------
        radius : float
            Disk radius in pixels.
        """
        disk = Tophat2DKernel(radius)
        disk.normalize('peak')
        list_kernel = [disk.array] * (len(self.significance_cube.energies()))
        counts = self.counts_cube.convolve(list_kernel)
        bkg = self.bkg_cube.convolve(list_kernel)
        self.significance_cube.data = significance(counts.data, bkg.data)

    def make_excess_cube(self):
        """Compute excess cube between counts and bkg cube."""
        self.excess_cube.data = self.counts_cube.data - self.bkg_cube.data

    # Define a method for the mean psf from a list of observation
    def make_mean_psf_cube(self, ref_cube, spectral_index=2.3):
        """Compute the mean PSF for a set of observation for different energy bands.

        Parameters
        ----------
        ref_cube : `~gammapy.cube.SkyCube`
            Reference sky cube to evaluate PSF on.
        spectral_index : float
            Assumed spectral index to compute mean PSF in energy band.

        Returns
        -------
        ref_cube : `~gammapy.cube.SkyCube`
            PSF mean cube
        """
        ref_cube = SkyCube.empty_like(ref_cube)
        obslist = self.data_store.obs_list(self.data_store.obs_table["OBS_ID"])

        header = ref_cube.sky_image_ref.to_image_hdu().header
        energy_bins = ref_cube.energies()
        center = ref_cube.sky_image_ref.center

        for idx_energy, E in enumerate(energy_bins[0:-1]):
            energy_band = Energy([energy_bins[idx_energy].value, energy_bins[idx_energy + 1].value], energy_bins.unit)
            energy = EnergyBounds.equal_log_spacing(energy_band[0].value, energy_band[1].value, 100, energy_band.unit)
            psf_energydependent = obslist.make_mean_psf(center, energy, rad=None)
            try:
                psf_table = psf_energydependent.table_psf_in_energy_band(
                    energy_band, spectral_index=spectral_index)
            except:
                rad = psf_energydependent.rad
                dp_domega = u.Quantity(np.zeros(len(psf_energydependent.rad)), 'sr-1')
                psf_table = TablePSF(rad, dp_domega)

            image = fill_acceptance_image(
                header, center,
                psf_table._rad.to("deg"),
                psf_table._dp_domega,
                psf_table._rad.to("deg")[-1],
            )
            ref_cube.data[idx_energy, :, :] = image.data / image.data.sum()

        return ref_cube
