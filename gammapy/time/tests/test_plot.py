# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from ...utils.testing import requires_dependency, requires_data
from ..plot import plot_fermi_3fgl_light_curve


# Some change in our dependencies (maybe matplotlib, maybe astropy)
# makes this now fail with this error:
# https://travis-ci.org/gammapy/gammapy/jobs/140330531#L1607
@pytest.mark.xfail
@requires_dependency('matplotlib')
@requires_data('gammapy-extra')
def test_plot_fermi_3fgl_light_curve():
    plot_fermi_3fgl_light_curve('3FGL J0349.9-2102',
                                time_start='2010-01-01',
                                time_end='2015-02-02')
