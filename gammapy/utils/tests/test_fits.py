# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pytest
from astropy.io import fits
from ..fits import (
    SmartHDUList,
)


def make_test_hdu_list():
    return fits.HDUList([
        fits.PrimaryHDU(),
        fits.BinTableHDU(name='TABLE1'),
        fits.ImageHDU(name='IMAGE1', data=np.zeros(shape=(1, 2, 3))),
        fits.BinTableHDU(name='TABLE2'),
        fits.ImageHDU(name='IMAGE2', data=np.zeros(shape=(4, 5))),
    ])


class TestSmartHDUList:
    def setup(self):
        self.hdus = SmartHDUList(hdu_list=make_test_hdu_list())

        self.names = ['PRIMARY', 'TABLE1', 'IMAGE1', 'TABLE2', 'IMAGE2']
        self.numbers = list(range(5))

    def test_names(self):
        assert self.hdus.names == self.names

    def test_fits_get_hdu(self):
        def g(hdu=None, hdu_type=None):
            """Short helper function, to save some typing."""
            return self.hdus.get_hdu(hdu, hdu_type).name

        # Make a few valid queries, and assert that the right result comes back

        for number, name in zip(self.numbers, self.names):
            assert g(hdu=name) == name
            assert g(hdu=name.lower()) == name
            assert g(hdu=number) == name

        g(hdu_type='image') == 'IMAGE1'
        g(hdu_type='table') == 'TABLE1'

        # Call the method incorrectly, and assert that ValueError is raised:

        with pytest.raises(ValueError) as exc:
            g()
        assert str(exc.value) == 'Must give either `hdu` or `hdu_type`. Got `None` for both.'

        # with pytest.raises(ValueError) as exc:
        #     g(hdu='TABLE1', hdu_type='table')
        # assert str(exc.value) == (
        #     "Must give either `hdu` or `hdu_type`."
        #     " Got a value for both: hdu=TABLE1 and hdu_type=table"
        # )

        with pytest.raises(ValueError) as exc:
            g(hdu_type='bad value')
        assert str(exc.value) == "Invalid hdu_type=bad value"

        # Query for non-existent HDUs, and assert that KeyError is raised:

        with pytest.raises(KeyError) as exc:
            g(hdu=['bad', 'type'])

        with pytest.raises(KeyError) as exc:
            g(hdu='kronka lonka')

        with pytest.raises(KeyError) as exc:
            g(hdu=42)
        # assert str(exc.value) == 'HDU not found: hdu=42. Index out of range.'

    def test_fits_get_hdu_index(self):
        # We test almost everything above via `test_fits_get_hdu`
        # Here we just add a single test for `get_hdu_index` to
        # make sure it returns an int index all right.
        assert self.hdus.get_hdu_index(hdu='TABLE2') == 3

    def test_read_write(self, tmpdir):
        filename = str(tmpdir / 'data.fits')
        self.hdus.write(filename)
        hdus2 = SmartHDUList.open(filename)
        assert self.hdus.names == hdus2.names
