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
    i_nan = np.where(np.isnan(exposure_cube.data))
    exposure_cube.data[i_nan] = 0
    return dict(counts=npred_cube, exposure=exposure_cube)

def get_fit_model():
    spatial_model = SkyGaussian2D(
        lon_0='0 deg',
        lat_0='0 deg',
        sigma='1 deg',
    )
    spectral_model = PowerLaw(
        index=2,
        amplitude='1e-10 cm-2 s-1 TeV-1',
        reference='1 TeV',
    )
    return SkyModel(
        spatial_model=spatial_model,
        spectral_model=spectral_model,
    )

def main():
    log.setLevel('INFO')
    log.info('Starting ...')

    cubes = load_cubes()
    log.info('Loaded cubes: {}'.format(cubes))

    model = get_fit_model()
    log.info('Loaded model: {}'.format(model))

    # NOTE: Without this the fitter set the amplitude to 0
    # This result in npred = 0 and thus cash = 0 everywhere
    model.parameters['amplitude'].parmin = 1e-12

    fit = CubeFit(cubes=cubes, model=model.copy())
    log.info('Created analysis: {}'.format(fit))

    fit.fit()
    log.info('Starting values\n{}'.format(get_fit_model().parameters))
    log.info('Best fit values\n{}'.format(fit.model.parameters))
    log.info('True values\n{}'.format(get_sky_model().parameters))
    

class CubeFit(object):
    """Perform 3D likelihood fit

    This is my first go at such a class. It's geared to the SpectrumFit class
    which does the 1D spectrum fit.

    Parameters
    ----------
    cubes : dict
        Dict containting tow WcsNDMaps: 'counts' and 'exposure'
    model : `~gammapy.cube.SkyModel`
        Fit model
    """
    def __init__(self, cubes, model):
        self.cubes = cubes
        self.model = model
        self._init_evaluator()
        
        self.npred = None
        self.stat = None

    def _init_evaluator(self):
        """Initialize SkyModelEvaluator"""
        self.evaluator = SkyModelMapEvaluator(self.model,
                                              self.cubes['exposure'])
        
    def compute_npred(self):
        self.npred = self.evaluator.compute_npred()

    def compute_stat(self):
        self.stat = cash(
            n_on=self.cubes['counts'].data,
            mu_on=self.npred
        )

    def total_stat(self, parameters):
        log.debug('\n-----------------\n\n')
        log.debug(parameters)
        self.model.parameters = parameters
        self.compute_npred()
        self.compute_stat()
        total_stat = np.sum(self.stat, dtype=np.float64)
        #if(total_stat == 0):
        #    import IPython; IPython.embed()
        log.debug('STAT: {}'.format(total_stat))
        return total_stat

    def fit(self):
        """Run the fit"""
        parameters, minuit = fit_minuit(parameters=self.model.parameters,
                                        function=self.total_stat)
        log.info(minuit.get_param_states())

    

if __name__ == '__main__':
    main()
