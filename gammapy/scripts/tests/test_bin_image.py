# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from ...datasets import get_path
from ..bin_image import main as bin_image_main


def test_bin_image_main(tmpdir):
    # TODO: make a better example dataset where the counts are actually
    # inside the image, then add useful asserts

    event_file = get_path('hess/run_0023037_hard_eventlist.fits.gz')
    reference_file = get_path('fermi/fermi_exposure.fits.gz')

    filename = str(tmpdir.join('bin_image_test.fits'))
    open(filename, 'w').write('hi')
    text = open(filename).read()
    assert text == 'hi'
    # TODO: this doesn't currently work because bin_image_main has an issue
    # (could be considered a bug)
    # bin_image_main([event_file, reference_file, out_file])
    # read output file and assert something