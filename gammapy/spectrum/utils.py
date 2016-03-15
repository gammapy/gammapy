# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
from astropy.units import Quantity

from gammapy.region import SkyCircleRegion, SkyRegionList
from ..utils.scripts import read_yaml

__all__ = [
    'LogEnergyAxis',
    'plot_npred_vs_excess',
]


class LogEnergyAxis(object):
    """Log10 energy axis.

    Defines a transformation between:

    * ``energy = 10 ** x``
    * ``x = log10(energy)``
    * ``pix`` in the range [0, ..., len(x)] via linear interpolation of the ``x`` array,
      e.g. ``pix=0`` corresponds to ``x[0]``
      and ``pix=0.3`` is ``0.5 * (0.3 * x[0] + 0.7 * x[1])``

    .. note::
        The `specutils.Spectrum1DLookupWCS <http://specutils.readthedocs.org/en/latest/api/specutils.wcs.specwcs.Spectrum1DLookupWCS.html>`__
        class is similar (only that it doesn't include the ``log`` transformation and the API is different.
        Also see this Astropy feature request: https://github.com/astropy/astropy/issues/2362

    Parameters
    ----------
    energy : `~astropy.units.Quantity`
        Energy array
    """

    def __init__(self, energy):
        self.energy = energy
        self.x = np.log10(energy.value)
        self.pix = np.arange(len(self.x))

    def world2pix(self, energy):
        """TODO: document.
        """
        # Convert `energy` to `x = log10(energy)`
        x = np.log10(energy.to(self.energy.unit).value)

        # Interpolate in `x`
        pix = np.interp(x, self.x, self.pix)

        return pix

    def pix2world(self, pix):
        """TODO: document.
        """
        # Interpolate in `x = log10(energy)`
        x = np.interp(pix, self.pix, self.x)

        # Convert `x` to `energy`
        energy = Quantity(10 ** x, self.energy.unit)

        return energy

    def closest_point(self, energy):
        """TODO: document
        """
        x = np.log10(energy.value)
        # TODO: I'm not sure which is faster / better here?
        index = np.argmin(np.abs(self.x - x))
        # np.searchsorted(self.x, x)
        return index

    def bin_edges(self, energy):
        """TODO: document.

        Parameters
        ----------
        TODO

        Returns
        -------
        TODO
        """
        try:
            pix = np.where(energy >= self.energy)[0][-1]
        except ValueError:
            # Loop over es by hand
            pix1 = np.empty_like(energy, dtype=int)
            for ii in range(energy.size):
                # print ii, e[ii], np.where(e[ii] >= self.e)
                pix1[ii] = np.where(energy[ii] >= self.energy)[0][-1]
        pix2 = pix1 + 1
        energy1 = self.energy[pix1]
        energy2 = self.energy[pix2]

        return pix1, pix2, energy1, energy2


# Todo: find a better place for these functions (Spectrum analysis class?)
def get_empty_skymap(**kwargs):
    """Plot empty sky map

    It will have the same WCS specification as the exclusion mask.
    The plot will be centered at the target position
    """
    from gammapy.image import SkyMap
    from gammapy.spectrum import SpectrumExtraction

    exclusion = SkyMap.read(SpectrumExtraction.EXCLUDEDREGIONS_FILE, 1)
    skymap = SkyMap.empty_like(exclusion)
    return skymap


def plot_exclusion_contours(ax, **kwargs):
    """Plot exclusion contours"""
    from gammapy.image import SkyMap
    from gammapy.spectrum import SpectrumExtraction
    import matplotlib.pyplot as plt

    exclusion = SkyMap.read(SpectrumExtraction.EXCLUDEDREGIONS_FILE)
    plt.contour(exclusion.data, [0.5], **kwargs)

    return ax


def plot_on_region(ax, **kwargs):
    """Plot target regions"""
    from gammapy.spectrum import SpectrumExtraction

    val = read_yaml(SpectrumExtraction.REGIONS_FILE)
    on_region = SkyCircleRegion.from_dict(val['on_region'])
    on_region.plot(ax, **kwargs)
    return ax


def plot_off_region(ax, **kwargs):
    """Plot off regions for all observations"""
    from gammapy.spectrum import SpectrumExtraction

    val = read_yaml(SpectrumExtraction.REGIONS_FILE)
    all_regions = SkyRegionList()
    for regions in val['off_region'].values():
        all_regions.append(SkyRegionList.from_dict(regions))
    all_regions.plot(ax, **kwargs)
    return ax


def plot_observations_positions(ax, **kwargs):
    from gammapy.data import ObservationTable
    from gammapy.spectrum import SpectrumExtraction

    kwargs.setdefault('marker', 'x')
    kwargs.setdefault('s', 150)
    obs_table = ObservationTable.read(SpectrumExtraction.OBSTABLE_FILE)
    ra = obs_table['RA_PNT']
    dec = obs_table['DEC_PNT']
    ax.scatter(ra, dec, transform=ax.get_transform('icrs'), **kwargs)


def fill_off_events_in_skymap(skymap=None):
    from gammapy.data import EventList
    from gammapy.spectrum import SpectrumExtraction

    skymap = get_empty_skymap() if skymap is None else skymap
    off_events = EventList.read(SpectrumExtraction.OFFLIST_FILE)
    filled_map = off_events.fill_skymap(skymap)
    return filled_map

def plot_npred_vs_excess(ogip_dir='ogip_data', npred_dir='n_pred', ax=None):
    """Plot predicted and measured excess counts

    Parameters
    ----------
    npred_dir : str, Path
        Directory holding npred fits files
    ogip_dir : str, Path
        Directory holding OGIP data
    """
    from ..spectrum.spectrum_extraction import SpectrumObservationList
    from ..spectrum import CountsSpectrum
    from ..utils.scripts import make_path

    import matplotlib.pyplot as plt
    ax = plt.gca() if ax is None else ax

    ogip_dir = make_path(ogip_dir)
    n_pred_dir = make_path(npred_dir)

    obs = SpectrumObservationList.read_ogip(ogip_dir)
    excess = np.sum([o.excess_vector for o in obs])

    # Need to give RMF file for reco energy binning
    id = obs[0].meta.obs_id
    rmf = str(ogip_dir/ 'rmf_run{}.fits'.format(id))
    val = [CountsSpectrum.read_bkg(_, rmf) for _ in n_pred_dir.glob('*.fits')]
    npred = np.sum(val)

    npred.plot(ax=ax, color='red', alpha=0.7, label='Predicted counts')
    excess.plot(ax=ax, color='green', alpha=0.7, label='Excess counts')
    ax.legend(numpoints=1)
    plt.xscale('log')

    return ax
