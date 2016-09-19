# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from ...image import SkyImageList, SkyImage


def assert_sky_image_collection_isclose(images1, images2, check_wcs=True):
    """Assert that two sky image collections are almost the same.
    """
    assert len(images1) == len(images2)
    for image1, image2 in zip(images1, images2):
        assert_sky_image_isclose(image1, image2, check_wcs=check_wcs)


def assert_sky_image_isclose(image1, image2, check_wcs=True):
    """Assert that two sky images are almost the same.

    TODO: move this to `test_core.py` and use there also.
    """
    assert image1.name == image2.name

    if (image1.data is None) and (image2.data is None):
        pass
    elif (image1.data is not None) and (image2.data is not None):
        assert_allclose(image1.data, image2.data)
    else:
        raise ValueError('One image has `data==None` and the other does not.')

    if check_wcs is False:
        pass
    elif (image1.wcs is None) and (image2.wcs is None):
        pass
    elif (image1.wcs is not None) and (image2.wcs is not None):
        assert_wcs_isclose(image1.wcs, image2.wcs)
    else:
        raise ValueError('One image has `wcs==None` and the other does not.')


def assert_wcs_isclose(wcs1, wcs2):
    """Assert that two WCS objects are almost the same.

    TODO: move this to `test_core.py` and use there also.
    """
    # TODO: implement properly
    assert_allclose(wcs1.wcs.cdelt, wcs2.wcs.cdelt)


class TestSkyImageList:
    @staticmethod
    def assert_hdu_list_roundtrips(images, check_wcs=True):
        images2 = SkyImageList.from_hdu_list(images.to_hdu_list())
        assert_sky_image_collection_isclose(images, images2, check_wcs=check_wcs)
        return images2

    @staticmethod
    def assert_read_write_roundtrips(filename, images, check_wcs=True):
        images.write(filename)
        # Make sure clobber works as expected by writing again.
        # Before there was a bug where this appended and duplicated HDUs
        images.write(filename, clobber=True)
        images2 = SkyImageList.read(filename)
        assert_sky_image_collection_isclose(images, images2, check_wcs=check_wcs)
        return images2

    @staticmethod
    def make_test_images():
        image1 = SkyImage.empty(name='image1')
        image2 = SkyImage.empty(name='image 2', nxpix=3, nypix=2)
        return SkyImageList(images=[image1, image2])

    def test_empty(self):
        """Test operations with an empty example."""
        images = SkyImageList()

        assert len(images) == 0
        assert images.names == []
        assert len(images.to_hdu_list()) == 0
        assert 'Number of images: 0' in str(images)

    def test_one(self, tmpdir):
        """Test with a one-element example."""
        image1 = SkyImage(name='image1')
        images = SkyImageList(images=[image1])

        assert len(images) == 1
        assert images.names == ['image1']
        assert len(images.to_hdu_list()) == 1
        assert 'Number of images: 1' in str(images)

        self.assert_hdu_list_roundtrips(images, check_wcs=False)
        self.assert_read_write_roundtrips(tmpdir / 'test.fits', images, check_wcs=False)

    def test_two(self, tmpdir):
        """Test with a two-element example."""
        image1 = SkyImage(name='image1')
        image2 = SkyImage(name='image2')
        images = SkyImageList(images=[image1, image2])

        assert len(images) == 2
        assert images.names == ['image1', 'image2']
        assert len(images.to_hdu_list()) == 2
        assert 'Number of images: 2' in str(images)

        self.assert_hdu_list_roundtrips(images, check_wcs=False)
        self.assert_read_write_roundtrips(tmpdir / 'test.fits', images, check_wcs=False)

    def test_two_with_data_and_wcs(self, tmpdir):
        """Test with a two-element example where the images have data and wcs."""
        images = self.make_test_images()

        assert len(images) == 2
        assert images.names == ['image1', 'image 2']
        assert len(images.to_hdu_list()) == 2
        assert 'Number of images: 2' in str(images)

        self.assert_hdu_list_roundtrips(images, check_wcs=True)
        self.assert_read_write_roundtrips(tmpdir / 'test.fits', images)

    def test_getitem(self):
        """Test that `__getitem__` works as expected.
        """
        images = self.make_test_images()

        assert images[0].name == 'image1'
        assert images['image1'].name == 'image1'

        assert images[1].name == 'image 2'
        assert images['image 2'].name == 'image 2'

        with pytest.raises(KeyError):
            images['not available']
        with pytest.raises(IndexError):
            images[2]

    def test_setitem(self):
        """Test that `__setitem__` works as expected.
        """
        images = self.make_test_images()

        images.append(SkyImage(name='append'))
        assert images['append'].name == 'append'

        images[1] = SkyImage(name='index')
        assert images['index'].name == images[1].name == 'index'

        # Image with new key gets appended at the end
        images['new key'] = SkyImage(name='new key')
        assert images['new key'].name == images[-1].name == 'new key'

        # Image with existing key gets overwritten
        images['image1'] = SkyImage(name='image1')
        assert images['image1'].name == images[0].name == 'image1'

        # Image without a name gets the key set as name
        images['aaa'] = SkyImage(name=None)
        assert images['aaa'].name == images[-1].name == 'aaa'

        with pytest.raises(KeyError):
            images['aaa'] = SkyImage(name='bbb')

            # TODO: test more error cases for setitem
            # TODO: test delitem by index and name

    def test_meta(self):
        images = SkyImageList(meta=dict(a=42))
        assert images.meta['a'] == 42
