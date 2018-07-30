# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from ...testing import requires_dependency
from ....maps import Map
from ....spectrum.models import PowerLaw



@requires_dependency('sherpa')
def test_sherpa_wrapper():
    from sherpa.fit import Fit
    from sherpa.data import DataSimulFit
    from ..sherpa import (SherpaDataWrapper, SherpaStatWrapper,
                          SherpaModelWrapper, SHERPA_OPTMETHODS)

    map_wcs = Map.create(binsz=1, width=(3, 4), map_type='wcs')
    data = SherpaDataWrapper(map_wcs)
    model = PowerLaw()

    def gp_stat(gp_data, gp_model):
        assert gp_data is map_wcs
        assert gp_model is model
        return 0, 0

    stat = SherpaStatWrapper(gp_stat)
    data = DataSimulFit(name='gp_data', datasets=[data])
    method = SHERPA_OPTMETHODS['simplex']
    models = SherpaModelWrapper(model)

    fitter = Fit(data=data, model=models, stat=stat, method=method)
    fitter.calc_stat()





