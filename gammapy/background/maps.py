# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import logging
import numpy as np
from astropy.io import fits
from ..image import disk_correlate
from .. import stats

__all__ = ['Maps', 'BASIC_MAP_NAMES', 'DERIVED_MAP_NAMES']

BASIC_MAP_NAMES = ['n_on', 'a_on', 'n_off', 'a_off',
                   'exclusion', 'exposure']

basic_map_defaults = [0, 1, 0, 1, 1, 1]

DERIVED_MAP_NAMES = ['alpha', 'area_factor', 'background',
                     'excess', 'significance', 'flux']


class Maps(fits.HDUList):
    """Maps container for basic maps and methods to compute derived maps.

    It is simply a list of HDUs containing the maps, plus methods to
    compute the derived maps.

    These maps allow implementing all background estimation methods.
    Not all maps are used for each method, unused maps are typically
    filled with zeros or ones as appropriate.

    TODO: Correlation of basic maps is done repeatedly when
          computing all derived maps.
          Is it worth speeding things up by writing the steps explicitly?


    Parameters
    ----------
    hdus : `~astropy.io.fits.HDUList` containing `~astropy.io.fits.ImageHDU` objects
        Must contain at least one of the basic maps
    file : str
        Passed right on to HDUList constructor
    rename_hdus : dict
        Dictionary of HDUs to rename, e.g. rename_hdus=dict(n_on=3, exclusion=2).
    is_off_correlated : bool
        Flag whether the off map is already correlated
    theta : float
        Correlation radius (deg)
    theta_pix : float
        Correlation radius (pix)
    """
    def __init__(self, hdus=[], file=None, rename_hdus=None,
                 is_off_correlated=True, theta=None, theta_pix=0):
        super(Maps, self).__init__(hdus, file)

        #import IPython; IPython.embed()
        #if rename_hdus is not None:
        #    for name, number in rename_hdus.items():
        #        self[number].name = name

        hdu_names = [hdu.name.lower() for hdu in self]
        print(hdu_names)

        # Check that there is at least one of the basic_maps present.
        # This is required so that the map geometry is defined.
        existing_basic_maps = [name for name in BASIC_MAP_NAMES
                               if name in hdu_names]
        nonexisting_basic_maps = [name for name in BASIC_MAP_NAMES
                                  if name not in hdu_names]
        if not existing_basic_maps:
            logging.error('hdu_names =', hdu_names)
            logging.error('BASIC_MAP_NAMES = ', BASIC_MAP_NAMES)
            raise IndexError('hdus must contain at least one of the BASIC_MAP_NAMES')
        # Declare any one of the existing basic maps the reference map.
        # This HDU will be used as the template when adding other hdus.
        self.ref_hdu = self[existing_basic_maps[0]]
        # If the HDUList doesn't contain a PrimaryHDU at index x,
        # add an empty one because this is required by the FITS standard
        if not isinstance(self[0], fits.PrimaryHDU):
            self.insert(0, fits.PrimaryHDU())
        # Add missing BASIC_MAP_NAMES with default value and
        # same shape and type as existing reference basic map
        logging.debug('Adding missing basic maps: {0}'
                      ''.format(nonexisting_basic_maps))
        for name in nonexisting_basic_maps:
            value = basic_map_defaults[BASIC_MAP_NAMES.index(name)]
            data = np.ones_like(self.ref_hdu.data) * value
            header = self.ref_hdu.header
            hdu = fits.ImageHDU(data, header, name)
            self.append(hdu)
        self.is_off_correlated = is_off_correlated
        logging.debug('is_off_correlated: {0}'.format(self.is_off_correlated))
        # Set the correlation radius in pix
        if theta and 'CDELT2' in self.ref_hdu.header:
            self.theta = theta / self.ref_hdu.header['CDELT2']
        else:
            self.theta = theta_pix
        logging.debug('theta: {0}'.format(self.theta))

    def get_basic(self, name):
        """Gets the data of a basic map and disk-correlates if required.

        Parameters
        ----------
        name : str
            Map name

        Returns
        -------
        image : `numpy.array`
            Map data
        """
        # Build a list of maps requiring correlation
        requires_correlation = ['n_on', 'a_on', 'exposure']
        if not self.is_off_correlated:
            requires_correlation.extend(['n_off', 'a_off'])
        data = self[name].data
        if name in requires_correlation:
            # Makes a copy
            logging.debug('Correlating and returning map: {0}'.format(name))
            return disk_correlate(data, self.theta)
        else:
            # Doesn't make a copy, which is ok since
            # we only read from this array
            logging.debug('Returning map: {0}'.format(name))
            return data

    def get_derived(self, name):
        """Gets the data if it exists or makes it if not.

        Parameters
        ----------
        name : str
            Map name

        Returns
        -------
        image : `numpy.array`
            Map data
        """
        try:
            data = self[name].data
            logging.debug('Returning already existing derived map {0}'
                          ''.format(name))
            return data
        except KeyError:
            return eval('self.{0}.data'.format(name))

    def _make_hdu(self, data, name):
        """Helper function to make an image HDU.

        Parameters
        ----------
        data : array_like
            Image data
        name : str
            FITS extension name

        Returns
        -------
        hdu : `~astropy.io.fits.ImageHDU`
            Map HDU
        """
        return fits.ImageHDU(data, self.ref_hdu.header, name)

    @property
    def alpha(self):
        """Alpha map HDU."""
        a_on = self.get_basic('a_on')
        a_off = self.get_basic('a_off')
        alpha = a_on / a_off

        return self._make_hdu(alpha, 'alpha')

    @property
    def area_factor(self):
        """Area factor map HDU."""
        alpha = self.get_derived('alpha')
        area_factor = 1. / alpha

        return self._make_hdu(area_factor, 'area_factor')

    @property
    def background(self):
        """Background map HDU."""
        n_off = self.get_basic('n_off')
        alpha = self.get_derived('alpha')
        background = stats.background(n_off, alpha)

        return self._make_hdu(background, 'background')

    @property
    def make_excess(self):
        """Excess map HDU."""
        n_on = self.get_basic('n_on')
        background = self.get_derived('background')
        excess = n_on - background

        return self._make_hdu(excess, 'excess')

    @property
    def significance(self, method='lima', neglect_background_uncertainty=False):
        """Significance map HDU."""
        n_on = self.get_basic('n_on')
        n_off = self.get_basic('n_off')
        alpha = self.get_derived('alpha')

        significance = stats.significance_on_off(n_on, n_off, alpha, method,
                                                 neglect_background_uncertainty)
        return self._make_hdu(significance, 'significance')

    @property
    def flux(self):
        """Flux map HDU."""
        exposure = self.get_basic('exposure')
        excess = self.get_derived('excess')

        flux = excess / exposure
        return self._make_hdu(flux, 'flux')

    def make_derived_maps(self):
        """Make all the derived maps."""
        logging.debug('Making derived maps.')
        for name in DERIVED_MAP_NAMES:
            # Compute the derived map
            hdu = eval('self.make_{0}()'.format(name))
            # Put it in the HDUList, removing an older version
            # of the derived map should it exist.
            try:
                index = self.index_of(name)
                self[index] = hdu
                logging.info('Replacing existing extension {0}'.format(name))
            except KeyError:
                self.append(hdu)
                logging.info('Added extension {0}'.format(name))

    def make_correlated_basic_maps(self):
        """Make correlated versions of all the basic maps.

        @note This is mainly useful for debugging.
        @note All maps are disk-correlated, even e.g. the off map
        if it had been ring-correlated before already.
        """
        logging.debug('Making correlated basic maps.')
        for name in BASIC_MAP_NAMES:
            # Compute the derived map
            data = eval('self["{0}"].data'.format(name))
            data_corr = disk_correlate(data, self.theta)
            name = '{0}_corr'.format(name)
            hdu = self._make_hdu(data_corr, name)
            # Put it in the HDUList, removing an older version
            # of the derived map should it exist.
            try:
                index = self.index_of(name)
                self[index] = hdu
            except KeyError:
                self.append(hdu)
