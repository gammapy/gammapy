# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

from astropy.tests.helper import pytest
from ...datasets import gammapy_extra
from ...spectrum.spectrum_pipe import SpectrumPipe
from ...utils.scripts import read_yaml
from ...utils.testing import requires_dependency, requires_data, SHERPA_LT_4_8

@requires_dependency('scipy')
@requires_dependency('sherpa')
@pytest.mark.skipif('SHERPA_LT_4_8')
@requires_data('gammapy-extra')
def test_spectrum_pipe(tmpdir):
    configfile = gammapy_extra.filename('test_datasets/spectrum/spectrum_pipe_example.yaml')
    config = read_yaml(configfile)
    config['base_config']['extraction']['results']['outdir'] = str(tmpdir)
    config['base_config']['fit']['outdir'] = str(tmpdir)
    config['base_config']['fit']['observation_table'] = str(tmpdir / 'observations.fits')
    pipe = SpectrumPipe.from_config(config, auto_outdir=False)
    pipe.run()

