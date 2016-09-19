# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import logging
from astropy.extern import six
from astropy.io import fits
from astropy.io.fits import HDUList
from astropy.units import Quantity
from ..image import SkyImage
from ..utils.scripts import make_path

__all__ = ['SkyImageCollection', 'SkyImageList']

log = logging.getLogger(__name__)


class SkyImageCollection(list):
    """List of `~gammapy.image.SkyImage` objects.

    This is a simple class that provides

    * FITS I/O
    * Dict-like access by string image name keys
      in addition to list-like access by integer index.

    Examples
    --------

    Load the image collection from a FITS file:

    >>> from gammapy.image import SkyImage, SkyImageCollection
    >>> images = SkyImageCollection.read('$GAMMAPY_EXTRA/datasets/fermi_survey/all.fits.gz')

    Which images are available?

    >>> images.names

    Access one image by list index or image name string key:

    >>> images[0]
    >>> images['counts']
    >>> images['counts'].show('ds9')

    Print some summary info about the images:

    >>> print(images)

    Remove and append an image:

    >>> del images['background']
    >>> images.

    """

    # TODO: implement delitem by name
    # TODO: implement copy?
    # TODO: implement convenience constructors for many images with the same WCS?

    def __init__(self, images=None, meta=None):
        if images is None:
            images = []
        super(SkyImageCollection, self).__init__(images)

        if meta is not None:
            self.meta = OrderedDict(meta)
        else:
            self.meta = OrderedDict()

    @property
    def names(self):
        """List of image names."""
        return [_.name for _ in self]

    def __getitem__(self, key):
        """Add dict-like access by string image name as key.
        """
        # Special lookup by image name for string key
        if isinstance(key, six.string_types):
            if key in self.names:
                idx = self.names.index(key)
                return self[idx]
            else:
                fmt = 'No image with name: {}.\nAvailable image names: {}'
                raise KeyError(fmt.format(key, self.names))

        # Normal list lookup (for int key)
        return super(SkyImageCollection, self).__getitem__(key)

    def __setitem__(self, key, image):
        if isinstance(key, six.string_types):
            if image.name and image.name != key:
                fmt = "SkyImage(name='{}') doesn't match assigned key='{}'"
                raise KeyError(fmt.format(image.name, key))

            if key in self.names:
                idx = self.names.index(key)
                super(SkyImageCollection, self).__setitem__(idx, image)
            else:
                if image.name is None:
                    image.name = key
                self.append(image)
        else:
            super(SkyImageCollection, self).__setitem__(key, image)

    @classmethod
    def from_hdu_list(cls, hdu_list):
        """Construct from `~astropy.io.fits.HDUList`.
        """
        images = []
        for hdu in hdu_list:
            image = SkyImage.from_image_hdu(hdu)
            images.append(image)
        return cls(images)

    @classmethod
    def read(cls, filename, **kwargs):
        """Write to FITS file.

        ``kwargs`` are passed to `astropy.io.fits.open`.
        """
        filename = make_path(filename)
        hdu_list = fits.open(str(filename), **kwargs)
        return cls.from_hdu_list(hdu_list)

    def to_hdu_list(self):
        """Convert to `~astropy.io.fits.HDUList`.
        """
        hdu_list = HDUList()
        for image in self:
            hdu = image.to_image_hdu()
            hdu_list.append(hdu)
        return hdu_list

    def write(self, filename, **kwargs):
        """Write to FITS file.

        ``kwargs`` are passed to `astropy.io.fits.HDUList.writeto`.
        """
        filename = make_path(filename)
        hdu_list = self.to_hdu_list()
        hdu_list.writeto(str(filename), **kwargs)

    def __str__(self):
        s = 'SkyImageCollection:\n'
        s += 'Number of images: {}\n'.format(len(self))
        for idx, image in enumerate(self):
            s += 'Image(index={}, name={}) properties:'.format(idx, image.name)
            s += str(image)
        return s


class SkyImageList(object):
    """
    Class to represent connection between `~gammapy.image.SkyImage` and `~gammapy.cube.SkyCube`.

    Keeps list of images and has methods to convert between them and SkyCube.

    Parameters
    ----------
    name : str
        Name of the sky image list.
    images : list of `~gammapy.image.SkyImage`
        Data array as list of images.
    wcs : `~astropy.wcs.WCS`
        Word coordinate system transformation
    energy : `~astropy.units.Quantity`
        Energy array
    meta : dict
        Dictionary to store meta data.
    """

    def __init__(self, name=None, images=None, wcs=None, energy=None, meta=None):
        self.name = name
        self.images = images
        self.wcs = wcs
        self.energy = energy
        self.meta = meta

    def to_cube(self):
        """Convert this list of images to a `~gammapy.cube.SkyCube`.
        """
        from ..cube import SkyCube
        if hasattr(self.images[0].data, 'unit'):
            unit = self.images[0].data.unit
        else:
            unit = None
        data = Quantity([image.data for image in self.images],
                        unit)
        return SkyCube(name=self.name, data=data, wcs=self.wcs, energy=self.energy, meta=self.meta)
