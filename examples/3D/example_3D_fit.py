import numpy as np
import yaml
from configuration import get_model_gammapy
from sherpa.estmethods import Covariance
from sherpa.fit import Fit
from sherpa.optmethods import NelderMead, LevMar
from sherpa.stats import Cash, Chi2ConstVar
from sherpa.models import TableModel

from gammapy.cube.sherpa_ import CombinedModel3D, CombinedModel3DInt
from gammapy.cube import SkyCube
from gammapy.extern.pathlib import Path


def load_cubes(config):
    cube_dir = Path(config['logging']['working_dir'])
    npred_cube = SkyCube.read(cube_dir / 'npred_cube.fits.gz')
    exposure_cube = SkyCube.read(cube_dir / 'exposure_cube.fits', format='fermi-exposure')
    # print(exposure_cube)
    # print('exposure sum: {}'.format(np.nansum(exposure_cube.data)))
    i_nan = np.where(np.isnan(exposure_cube.data))
    exposure_cube.data[i_nan] = 0

    # npred_cube_convolved = SkyCube.read(cube_dir / 'npred_cube_convolved.fits.gz')

    return dict(counts=npred_cube, exposure=exposure_cube)


def read_config(filename):
    with open(filename) as fh:
        config = yaml.load(fh)

    # TODO: fix the following issue in a better way (e.g. raise an error or fix somehow)
    # apparently this gets returned as string, but we want float!?
    # prefactor1: 2e-12
    config['model']['prefactor1'] = float(config['model']['prefactor1'])

    return config


def main():
    config = read_config('config.yaml')
    # ref_cube = make_ref_cube(config)
    # target_position = SkyCoord(config['model']['ra1'], config['model']['dec1'], unit="deg").galactic

    cubes = load_cubes(config)
    print('which available cubes:', cubes)

    # converting data SkyCube to sherpa-format cube
    counts = cubes['counts'].to_sherpa_data3d() #dstype='Data3DInt')
    print('counts: ', counts)
    # Define a 2D gaussian for the spatial model
    # spatial_model = NormGauss2DInt('spatial-model')

    # Define a power law for the spectral model
    # spectral_model = PowLaw1D('spectral-model')

    coord = cubes['counts'].sky_image_ref.coordinates(mode="edges")
    energies = cubes['counts'].energies(mode='edges').to("TeV")
    print('my energy bins: ', energies)
    # import IPython; IPython.embed();

    # Set up exposure table model
    exposure = TableModel('exposure')
    exposure.load(None, cubes['exposure'].data.ravel())
    exposure.ampl.freeze()

    use_psf = config['model']['use_psf']

    model_gammapy = get_model_gammapy(config)

    spectral_model_sherpa = model_gammapy.spectral_model.to_sherpa()
    spectral_model_sherpa.ampl.thaw()

    spatial_model_sherpa = model_gammapy.spatial_model.to_sherpa()
    spatial_model_sherpa.xpos.freeze()
    spatial_model_sherpa.ypos.freeze()
    spatial_model_sherpa.r0.freeze()
    spatial_model_sherpa.width.freeze()

    source_model = CombinedModel3D(
        spatial_model=spatial_model_sherpa,
        spectral_model=spectral_model_sherpa,
    )

    # source_model = CombinedModel3DInt(
    #     spatial_model=spatial_model_sherpa,
    #     spectral_model=spectral_model_sherpa,
    #     exposure=exposure,
    #     coord=coord,
    #     energies=energies,
    #     use_psf=use_psf,
    #     psf=None,
    # )

    print(source_model)
    # source_model_cube = source_model.evaluate_cube(ref_cube)
    # model = source_model  # + background_model

    # source_model2 = CombinedModel3DInt(
    #     coord=coord,
    #     energies=energies,
    #     use_psf=False,
    #     exposure=cubes['exposure'],
    #     psf=None,
    #     spatial_model=spatial_model,
    #     spectral_model=spectral_model,
    # )

    model = 1e-9 * exposure * source_model  # 1e-9 flux factor

    fit = Fit(
        data=counts,
        model=model,
        stat=Chi2ConstVar(),
        method=LevMar(),
        # estmethod=Covariance(),
    )


    fit_results = fit.fit()
    print(fit_results.format())

    print('------------------------------------ end fitting')

    # import IPython; IPython.embed()


if __name__ == '__main__':
    main()
