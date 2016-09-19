# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import logging
import numpy as np
from astropy.io import fits
from astropy.units import Quantity
from ..extern.bunch import Bunch
from ..image import SkyImage
from ..utils.scripts import make_path

__all__ = ['SkyImageCollection', 'SkyImageList']

log = logging.getLogger(__name__)


class SkyImageCollection(Bunch):
    """
    Container for a collection of `~gammapy.image.SkyImage` objects.

    This class bundles as set of `SkyImage` objects in single data container and provides
    convenience methods for FITS I/O and `~gammapy.extern.bunch.Bunch` like
    handling of the data members.

    TODO: maybe `SkyImageList` should be a sub-class of `SkyImageCollection`?
    `SkyImageList` is a special case, for a collection of images in energy bands.

    Parameters
    ----------
    name : str
        Name of the collection
    meta : `~collections.OrderedDict`
        Dictionary to store meta data for the collection.

    Examples
    --------
    Load the image collection from a FITS file:

    >>> from gammapy.image import SkyImageCollection
    >>> images = SkyImageCollection.read('$GAMMAPY_EXTRA/datasets/fermi_survey/all.fits.gz')

    Then try tab completion on the ``images`` object to access the images.
    E.g. to show the counts image::

    >>> images.counts.show('ds9')
    """

    def __init__(self, name=None, meta=None, **kwargs):
        # Set real class attributes
        self._image_names = []
        self.name = name
        if meta:
            self.meta = meta
        else:
            self.meta = OrderedDict()

        # Everything else is stored as dict entries
        for key in kwargs:
            self[key] = kwargs[key]

    def __setitem__(self, key, item):
        """
        Overwrite __setitem__ operator to remember order the images are added
        to the collection, by storing it in the _image_names list.
        """
        if isinstance(item, np.ndarray):
            item = SkyImage(name=key, data=item)
        if isinstance(item, SkyImage):
            self._image_names.append(key)

        super(SkyImageCollection, self).__setitem__(key, item)

    @classmethod
    def read(cls, filename):
        """
        Create collection of images from FITS file.

        Parameters
        ----------
        filename : str
            FITS file name.
        """
        hdulist = fits.open(str(make_path(filename)))
        kwargs = {}
        _image_names = []  # list of image names to save order in FITS file

        for hdu in hdulist:
            image = SkyImage.from_image_hdu(hdu)

            # This forces lower case image names, but only on the collection object
            # When writing to FITS again the image.name attribute is used.
            name = image.name.lower()
            kwargs[name] = image
            _image_names.append(name)
        _ = cls(**kwargs)
        _._map_names = _image_names
        return _

    def write(self, filename=None, **kwargs):
        """
        Write images to FITS file.

        Parameters
        ----------
        filename : str
            FITS file name.
        """
        hdulist = fits.HDUList()

        for name in self.get('_image_names', sorted(self)):
            if isinstance(self[name], SkyImage):
                hdu = self[name].to_image_hdu()

                # For now add common collection meta info to the single image headers
                hdu.header.update(self.meta)
                hdu.name = name
                hdulist.append(hdu)
            else:
                log.warn("Can't save {} to file, not a image.".format(name))

        hdulist.writeto(filename, **kwargs)

    def info(self):
        """
        Print summary info about the image collection.
        """
        print(str(self))

    def __str__(self):
        info = ''
        for name in self.get('_image_names', sorted(self)):
            info += self[name].__str__()
            info += '\n'
        return info


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
