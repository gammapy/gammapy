# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import logging
from ..extern import six
from ..extern.six.moves import UserList
from astropy.io import fits
from astropy.io.fits import HDUList
from .core import SkyImage
from ..utils.scripts import make_path

__all__ = ['SkyImageList']

log = logging.getLogger(__name__)


class SkyImageList(UserList):
    """List of `~gammapy.image.SkyImage` objects.

    This is a simple class that provides

    * FITS I/O
    * Dict-like access by string image name keys
      in addition to list-like access by integer index.

    Examples
    --------

    Load the image collection from a FITS file:

    >>> from gammapy.image import SkyImage, SkyImageList
    >>> images = SkyImageList.read('$GAMMAPY_EXTRA/datasets/fermi_survey/all.fits.gz')

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
        super(SkyImageList, self).__init__(images)

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
                fmt = "No image with name: '{}'. Available image names: {}"
                raise KeyError(fmt.format(key, self.names))

        # Normal list lookup (for int key)
        return super(SkyImageList, self).__getitem__(key)

    def __setitem__(self, key, image):
        if isinstance(key, six.string_types):
            if image.name and image.name != key:
                fmt = "SkyImage(name='{}') doesn't match assigned key='{}'"
                raise KeyError(fmt.format(image.name, key))

            if key in self.names:
                idx = self.names.index(key)
                super(SkyImageList, self).__setitem__(idx, image)
            else:
                if image.name is None:
                    image.name = key
                self.append(image)
        else:
            super(SkyImageList, self).__setitem__(key, image)

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
        with fits.open(str(filename), **kwargs) as hdu_list:
            images = cls.from_hdu_list(hdu_list)
        return images

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
        s = 'SkyImageList:\n'
        s += 'Number of images: {}\n\n'.format(len(self))
        for idx, image in enumerate(self):
            s += 'Image(index={}, name={}) properties: \n'.format(idx, image.name)
            s += str(image)
            s += '\n'
        return s

    @staticmethod
    def assert_allclose(images1, images2, check_wcs=True):
        """Assert all-close for `SkyImageList`.

        A useful helper function to implement tests.
        """
        assert len(images1) == len(images2)

        for image1, image2 in zip(images1, images2):
            SkyImage.assert_allclose(image1, image2, check_wcs=check_wcs)

    def check_required(self, required_images):
        """Check if required images are present in the sky image list.

        Parameters
        ----------
        required_images : list
            List of names of required sky images.
        """
        for image in required_images:
            if image not in self.names:
                raise ValueError("Algorithm requires '{}' image to run.".format(image))
