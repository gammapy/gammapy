# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from ...utils.testing import requires_dependency, requires_data
from ..plot import plot_fermi_3fgl_light_curve


@requires_dependency('matplotlib')
def test_plot_time_difference_distribution():
    pass


@requires_dependency('matplotlib')
@requires_data('gammapy-extra')
def test_plot_fermi_3fgl_light_curve():
    plot_fermi_3fgl_light_curve('3FGL J0349.9-2102',
                                time_start='2010-01-01',
                                time_end='2015-02-02')
