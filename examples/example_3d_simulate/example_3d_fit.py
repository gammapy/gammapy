import numpy as np
import yaml
from astropy import log

from gammapy.image.models import SkyGaussian2D
from gammapy.spectrum.models import PowerLaw
from gammapy.maps import WcsNDMap
from gammapy.cube import SkyModel, SkyModelMapEvaluator
from gammapy.stats import cash
from gammapy.utils.fitting import fit_minuit
from example_3d_simulate import get_sky_model


def load_cubes():
    npred_cube = WcsNDMap.read('npred.fits')
    exposure_cube = WcsNDMap.read('exposure.fits')
    return dict(counts=npred_cube, exposure=exposure_cube)

def get_fit_model():
    spatial_model = SkyGaussian2D(
        lon_0='0 deg',
        lat_0='0 deg',
        sigma='1 deg',
    )
    spectral_model = PowerLaw(
        index=2,
        amplitude='1e-11 cm-2 s-1 TeV-1',
        reference='1 TeV',
    )
    model = SkyModel(
        spatial_model=spatial_model,
        spectral_model=spectral_model,
    )

    model.parameters.set_parameter_errors(
        {'lon_0': '0.1 deg',
        'lat_0': '0.1 deg',
        'sigma': '0.1 deg',
         'index': '0.1',
         'amplitude': '1e-12 cm-2 s-1 TeV-1'
        })

    model.parameters['sigma'].parmin = 0

    return model


def main():
    log.setLevel('INFO')
    log.info('Starting ...')

    cubes = load_cubes()
    log.info('Loaded cubes: {}'.format(cubes))

    model = get_fit_model()
    log.info('Loaded model: {}'.format(model))

    fit = CubeFit(model=model.copy(), **cubes)
    log.info('Created analysis: {}'.format(fit))

    fit.fit()
    log.info('Starting values\n{}'.format(model.parameters))
    log.info('Best fit values\n{}'.format(fit.model.parameters))
    log.info('True values\n{}'.format(get_sky_model().parameters))
    

class CubeFit(object):
    """Perform 3D likelihood fit

    This is my first go at such a class. It's geared to the
    `~gammapy.spectrum.SpectrumFit` class which does the 1D spectrum fit.

    Parameters
    ----------
    counts : `~gammapy.maps.WcsNDMap`
        Counts cube
    exposure : `~gammapy.maps.WcsNDMap`
        Exposure cube
    model : `~gammapy.cube.SkyModel`
        Fit model
    """
    def __init__(self, model, counts, exposure):
        self.model = model
        self.counts = counts
        self.exposure = exposure
        self._init_evaluator()
        
        self._npred = None
        self._stat = None
        self._minuit = None

    @property
    def npred(self):
        """Predicted counts cube"""
        return self._npred

    @property
    def stat(self):
        """Fit statistic per bin"""
        return self._stat

    @property
    def minuit(self):
        """`~iminuit.Minuit` object"""
        return self._npred

    def _init_evaluator(self):
        """Initialize SkyModelEvaluator"""
        self.evaluator = SkyModelMapEvaluator(self.model,
                                              self.exposure)
        
    def compute_npred(self):
        self._npred = self.evaluator.compute_npred()

    def compute_stat(self):
        self._stat = cash(
            n_on=self.counts.data,
            mu_on=self.npred
        )

    def total_stat(self, parameters):
        self.model.parameters = parameters
        self.compute_npred()
        self.compute_stat()
        total_stat = np.sum(self.stat, dtype=np.float64)
        log.debug('\n-----------------\n\n')
        log.debug(parameters)
        log.debug('STAT: {}'.format(total_stat))
        return total_stat

    def fit(self):
        """Run the fit"""
        parameters, minuit = fit_minuit(parameters=self.model.parameters,
                                        function=self.total_stat)
        self._minuit = minuit

    

if __name__ == '__main__':
    main()
