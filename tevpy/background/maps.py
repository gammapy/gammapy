"""
Maps container class

These maps allow implementing all background estimation methods.
Not all maps are used for each method, unused maps are typically
filled with zeros or ones as appropriate.

- Correlation of basic maps is done repeatedly when
  computing all derived maps.
  Is it worth speeding things up by writing the steps explicitly?

"""
import logging
import numpy as np
from astropy.io import fits
from ..utils.image import tophat_correlate
from .. import stats

__all__ = ['Maps']

basic_maps = ['n_on', 'a_on', 'n_off', 'a_off',
              'exclusion', 'exposure']
basic_map_defaults = [0, 1, 0, 1, 1, 1]
derived_maps = ['alpha', 'area_factor', 'background',
                'excess', 'significance',
                'flux']


class Maps(fits.HDUList):
    """Maps container for basic maps and methods to compute derived maps

    It is simply a list of HDUs containing the maps, plus methods to
    compute the derived maps."""
    def __init__(self, hdus=[], file=None,
                 is_off_correlated=True, theta=None, theta_pix=0):
        """Initialize the Maps object.
        @param hdus: HDUList of ImageHDUs containing at least one of the basic maps
        @param file: passed right on to HDUList constructor
        @param is_off_correlated: flag whether the off map is already correlated
        @param theta: correlation radius (deg)
        @param theta_pix: correlation radius (pix)"""
        super(Maps, self).__init__(hdus, file)
        # Check that there is at least one of the basic_maps present.
        # This is required so that the map geometry is defined.
        hdu_names = [hdu.name.lower() for hdu in self]
        existing_basic_maps = [name for name in basic_maps
                               if name in hdu_names]
        nonexisting_basic_maps = [name for name in basic_maps
                                  if name not in hdu_names]
        if not existing_basic_maps:
            logging.error('hdu_names =', hdu_names)
            logging.error('basic_maps = ', basic_maps)
            raise IndexError('hdus must contain at least one of the basic_maps')
        # Declare any one of the existing basic maps the reference map.
        # This HDU will be used as the template when adding other hdus.
        self.ref_hdu = self[existing_basic_maps[0]]
        # If the HDUList doesn't contain a PrimaryHDU at index x,
        # add an empty one because this is required by the FITS standard
        if not isinstance(self[0], fits.PrimaryHDU):
            self.insert(0, fits.PrimaryHDU())
        # Add missing basic_maps with default value and
        # same shape and type as existing reference basic map
        logging.debug('Adding missing basic maps: {0}'
                     ''.format(nonexisting_basic_maps))
        for name in nonexisting_basic_maps:
            value = basic_map_defaults[basic_maps.index(name)]
            data = np.ones_like(self.ref_hdu.data) * value
            header = self.ref_hdu.header
            hdu = fits.ImageHDU(data, header, name)
            self.append(hdu)
        self.is_off_correlated = is_off_correlated
        logging.info('is_off_correlated: {0}'.format(self.is_off_correlated))
        # Set the correlation radius in pix
        if theta and 'CDELT2' in self.ref_hdu.header:
            self.theta = theta / self.ref_hdu.header['CDELT2']
        else:
            self.theta = theta_pix
        logging.debug('theta: {0}'.format(self.theta))

    def get_basic(self, name):
        """Gets the data of a basic map and tophat correlates if required.
        @param name: basic map name"""
        # Build a list of maps requiring correlation
        requires_correlation = ['n_on', 'a_on', 'exposure']
        if not self.is_off_correlated:
            requires_correlation.extend(['n_off', 'a_off'])
        data = self[name].data
        if name in requires_correlation:
            # Makes a copy
            logging.debug('Correlating and returning map: {0}'.format(name))
            return tophat_correlate(data, self.theta)
        else:
            # Doesn't make a copy, which is ok since
            # we only read from this array
            logging.debug('Returning map: {0}'.format(name))
            return data

    def get_derived(self, name):
        """Gets the data if it exists or makes it if not.
        @param name: derived map name"""
        try:
            data = self[name].data
            logging.debug('Returning already existing derived map {0}'
                         ''.format(name))
            return data
        except KeyError:
            return eval('self.make_{0}().data'.format(name))

    def _make_hdu(self, data, name):
        """Helper function to make a FITSimage.
        @param data: numpy array
        @param name: FITS extension name"""
        return fits.ImageHDU(data, self.ref_hdu.header, name)

    def make_alpha(self):
        """Make the alpha map."""

        a_on = self.get_basic('a_on')
        a_off = self.get_basic('a_off')
        alpha = a_on / a_off

        return self._make_hdu(alpha, 'alpha')

    def make_area_factor(self):
        """Make the area factor map."""
        alpha = self.get_derived('alpha')
        area_factor = 1. / alpha

        return self._make_hdu(area_factor, 'area_factor')

    def make_background(self):
        """Make the background map."""
        n_off = self.get_basic('n_off')
        alpha = self.get_derived('alpha')
        background = stats.background(n_off, alpha)

        return self._make_hdu(background, 'background')

    def make_excess(self):
        """Make the excess map."""
        n_on = self.get_basic('n_on')
        background = self.get_derived('background')
        excess = n_on - background

        return self._make_hdu(excess, 'excess')

    def make_significance(self, method='lima', neglect_background_uncertainty=False):
        """Make the significance map using the Li & Ma formula."""
        n_on = self.get_basic('n_on')
        n_off = self.get_basic('n_off')
        alpha = self.get_derived('alpha')

        significance = stats.significance_on_off(n_on, n_off, alpha, method,
                                                      neglect_background_uncertainty)
        return self._make_hdu(significance, 'significance')

    def make_flux(self):
        """Make the flux map."""
        exposure = self.get_basic('exposure')
        excess = self.get_derived('excess')

        flux = excess / exposure
        return self._make_hdu(flux, 'flux')

    def make_derived_maps(self):
        """Make all the derived maps."""
        logging.debug('Making derived maps.')
        for name in derived_maps:
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
        @note All maps are tophat-correlated, even e.g. the off map
        if it had been ring-correlated before already."""
        logging.debug('Making correlated basic maps.')
        for name in basic_maps:
            # Compute the derived map
            data = eval('self["{0}"].data'.format(name))
            data_corr = tophat_correlate(data, self.theta)
            name = '{0}_corr'.format(name)
            hdu = self._make_hdu(data_corr, name)
            # Put it in the HDUList, removing an older version
            # of the derived map should it exist.
            try:
                index = self.index_of(name)
                self[index] = hdu
            except KeyError:
                self.append(hdu)
