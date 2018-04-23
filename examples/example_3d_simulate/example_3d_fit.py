from astropy import log
from gammapy.maps import WcsNDMap
from gammapy.spectrum.models import PowerLaw
from gammapy.image.models import SkyGaussian
from gammapy.cube import SkyModel, SkyModelMapFit
from example_3d_simulate import get_sky_model


def load_cubes():
    npred_cube = WcsNDMap.read('npred.fits')
    exposure_cube = WcsNDMap.read('exposure.fits')
    return dict(counts=npred_cube, exposure=exposure_cube)


def get_fit_model():
    spatial_model = SkyGaussian(
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

    fit = SkyModelMapFit(model=model.copy(), **cubes)
    log.info('Created analysis: {}'.format(fit))

    fit.fit()
    log.info('Starting values\n{}'.format(model.parameters))
    log.info('Best fit values\n{}'.format(fit.model.parameters))
    log.info('True values\n{}'.format(get_sky_model().parameters))


if __name__ == '__main__':
    main()
