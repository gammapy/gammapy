# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
from astropy.wcs import WCS
from .core import SkyImage
from ..utils.scripts import make_path

__all__ = ['SkyImageHealpix']


class SkyImageHealpix(object):
    """
    Sky image object with healpix pixelisation.

    Parameters
    ----------
    name : str
        Name of the image.
    data : `~numpy.ndarray`
        Data array.
    wcs : `WCSHealpix`
        Healpix WCS transformation object.
    unit : str
        String specifying the data units.
    meta : `~collections.OrderedDict`
        Dictionary to store meta data.
    """

    def __init__(self, name=None, data=None, wcs=None, meta=None, unit=None):
        self.name = name
        if not len(data) == wcs.npix:
            raise ValueError("'data' must have length of {}".format(wcs.npix))
        self.data = data
        self.wcs = wcs

        if meta is None:
            self.meta = OrderedDict()
        else:
            self.meta = OrderedDict(meta)

        self.unit = unit

    @classmethod
    def read(cls, filename):
        """Read image from FITS file.

        Parameters
        ----------
        filename : str
            FITS file name
        **kwargs : dict
            Keyword arguments passed `~healpy.read_map`.
        """
        import healpy as hp
        filename = make_path(filename)
        data, header = hp.read_map(str(filename), h=True)
        header = OrderedDict(header)
        nside = header['NSIDE']
        scheme = header.get('ORDERING', 'ring').lower()
        wcs = WCSHealpix(nside, scheme=scheme)
        return cls(data=data, wcs=wcs)

    def reproject(self, reference, **kwargs):
        """
        Reproject healpix image to given reference.

        Parameters
        ----------
        reference : `~astropy.io.fits.Header`, or `~gammapy.image.SkyImage`
            Reference image specification to reproject the data on.
        **kwargs : dict
            Keyword arguments passed to `~reproject.reproject_from_healpix`.

        Returns
        -------
        image : `~gammapy.image.SkyImage`
            Image reprojected onto ``reference``.
        """
        from reproject import reproject_from_healpix

        if isinstance(reference, SkyImage):
            wcs_reference = reference.wcs.deepcopy()
            shape_out = reference.data.shape
        elif isinstance(reference, fits.Header):
            wcs_reference = WCS(reference)
            shape_out = (reference['NAXIS2'], reference['NAXIS1'])
        else:
            raise TypeError("Invalid reference image. Must be either instance"
                            "of `Header`, `WCS` or `SkyImage`.")

        out = reproject_from_healpix((self.data, self.wcs.coordsys), wcs_reference,
                                     nested=self.wcs.nested, shape_out=shape_out)
        return SkyImage(name=self.name, data=out[0], wcs=wcs_reference)

    def plot(self, ax=None, projection='mollweide', **kwargs):
        """
        Plot healpix sky image using a global projection.

        Parameters
        ----------
        projection : ['mollweide', 'cartesian']
            Which projection to use for plotting.
        """
        import healpy as hp
        # TODO: add other projections
        if projection == 'mollweide':
            hp.mollview(map=self.data, nest=self.wcs.nested, **kwargs)
        elif projection == 'cartesian':
            hp.cartview(map=self.data, nest=self.wcs.nested, **kwargs)
        else:
            raise ValueError("Projection must be 'cartesian' or 'mollweide'")


class WCSHealpix(object):
    """
    Healpix transformation that behave WCS object like.

    TODO: check if this can be handled by `~astropy.wcs.WCS` as well and if this
    class is needed at all.
    """

    def __init__(self, nside, scheme='nested', coordsys='galactic'):
        self.coordsys = coordsys

        # check if nside is power of 2, taken from:
        # http://code.activestate.com/recipes/577514-chek-if-a-number-is-a-power-of-two/
        if not ((nside & (nside - 1)) == 0) and nside > 0:
            raise ValueError("'nside' must be power of 2!")
        self.nside = nside
        if not scheme in ['nested', 'ring']:
            raise ValueError("Scheme must be 'nested' or 'ring'")
        self.scheme = scheme

    def wcs_pix2world(self, ipix):
        import healpy as hp
        theta, phi = hp.pix2ang(self.nside, ipix, nest=self.nested)
        return theta, phi

    @property
    def nested(self):
        """Whether pixel ordering scheme is nested"""
        return self.scheme == 'nested'

    @property
    def npix(self):
        """Number of pixels corresponding to nside"""
        import healpy as hp
        return hp.nside2npix(self.nside)

    def __str__(self):
        info = 'WCSHealpix\n'
        info += '==========\n'
        info += '  coordsys: {}\n'.format(self.coordsys)
        info += '  nside   : {}\n'.format(self.nside)
        info += '  npix    : {}\n'.format(self.npix)
        info += '  nested  : {}\n'.format(self.nested)
        return info
