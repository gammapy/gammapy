# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

from gammapy.datasets import gammapy_extra
from gammapy.scripts import SpectrumPipe
from gammapy.utils.scripts import read_yaml
from gammapy.utils.testing import requires_dependency, requires_data


@requires_dependency('scipy')
@requires_dependency('sherpa')
@requires_data('gammapy-extra')
def test_spectrum_pipe(tmpdir):
    configfile = gammapy_extra.filename('test_datasets/spectrum/spectrum_pipe_example.yaml')
    config = read_yaml(configfile)
    config['base_config']['general']['outdir'] = str(tmpdir)
    pipe = SpectrumPipe.from_config(config, auto_outdir=False)
    pipe.run()
