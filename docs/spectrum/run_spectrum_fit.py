"""
This script illustrates how to use `~gammapy.spectrum.SpectralFit`
in order to fit a model to set of OGIP data
"""


from gammapy.datasets import gammapy_extra
from gammapy.spectrum.spectrum_analysis import SpectralFit

pha23592 = gammapy_extra.filename("datasets/hess-crab4_pha/pha_run23592.pha")
pha23526 = gammapy_extra.filename("datasets/hess-crab4_pha/pha_run23526.pha")
pha_list = [pha23592, pha23426]
fit = SpectralFit(pha_list)
fit.model = 'PL'
fit.low_threshold = '100 GeV'
fit.high_threshold = '10 TeV'
fit.run(method='sherpa')
