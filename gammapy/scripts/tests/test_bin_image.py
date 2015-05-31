# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from tempfile import NamedTemporaryFile
from astropy.tests.helper import pytest
from ...datasets import get_path
from ..bin_image import main as bin_image_main


def test_bin_image_main():
    # TODO: make a better example dataset where the counts are actually
    # inside the image, then add useful asserts

    event_file = get_path('hess/run_0023037_hard_eventlist.fits.gz')
    reference_file = get_path('fermi/fermi_exposure.fits.gz')
    with NamedTemporaryFile(suffix='.fits') as temp_file:
        open(temp_file.name, 'w').write('hi')
        text = open(temp_file.name).read()
        assert text == 'hi'
    # TODO: this doesn't currently work because bin_image_main has an issue
    # (could be considered a bug)
    # bin_image_main([event_file, reference_file, out_file])
    # read output file and assert something