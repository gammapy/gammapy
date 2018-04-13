"""Example how to simulate and fit a 3D map using CTA IRFs.

"""
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from gammapy.irf import EffectiveAreaTable2D, EnergyDispersion2D, EnergyDependentMultiGaussPSF, Background3D
from gammapy.maps import WcsGeom, MapAxis
from gammapy.spectrum.models import PowerLaw
from gammapy.image.models import Shell2D
from gammapy.cube import make_map_exposure_true_energy
from gammapy.cube import compute_npred_cube, compute_npred_cube_simple, CombinedModel3D


def get_irfs():
    filename = '$GAMMAPY_EXTRA/datasets/cta-1dc/caldb/data/cta//1dc/bcf/South_z20_50h/irf_file.fits'
    psf = EnergyDependentMultiGaussPSF.read(filename, hdu='POINT SPREAD FUNCTION')
    aeff = EffectiveAreaTable2D.read(filename, hdu='EFFECTIVE AREA')
    edisp = EnergyDispersion2D.read(filename, hdu='ENERGY DISPERSION')
    bkg = Background3D.read(filename, hdu='BACKGROUND')
    # TODO: probably this should be done later, not here
    # e.g. we might have more precise analysis later
    # with spatially varying responses
    # offset = Angle('2 deg')
    # psf = psf_fov.to_energy_dependent_table_psf(theta=offset)
    # edisp = edisp_fov.to_energy_dispersion(offset=offset)
    return dict(psf=psf, aeff=aeff, edisp=edisp, bkg=bkg)


def get_model():
    spatial_model = Shell2D(
        amplitude=1,
        x_0=0.2,
        y_0=0.1,
        r_in=0.3,
        width=0.2,
        # Note: for now we need spatial models that are normalised
        # to integrate to 1 or results will be incorrect!!!
        normed=True,
    )
    spectral_model = PowerLaw(index=2, amplitude='1e-11 cm-2 s-1 TeV-1', reference='1 TeV')
    model = CombinedModel3D(
        spatial_model=spatial_model,
        spectral_model=spectral_model,
    )
    return model


def get_geom():
    axis = MapAxis.from_edges(np.logspace(-1., 1., 10), unit=u.TeV)
    geom = WcsGeom.create(skydir=(0, 0), binsz=0.02, width=(8, 3), coordsys='GAL', axes=[axis])
    return geom


def main():
    # Define some parameters
    pointing = SkyCoord(1, 0.5, unit='deg', frame='galactic')
    livetime = 1 * u.hour
    offset_max = 3 * u.deg

    irfs = get_irfs()
    geom = get_geom()
    model = get_model()

    # Compute maps
    exposure_map = make_map_exposure_true_energy(
        pointing=pointing, livetime=livetime, aeff=irfs['aeff'],
        ref_geom=geom, offset_max=offset_max,
    )
    exposure_map.write('exposure.fits')
    print('exposure sum: {}'.format(np.nansum(exposure_map.data)))


    # Compute PSF-convolved npred in a few steps
    # 1. flux cube
    # 2. npred_cube
    # 3. apply PSF
    # 4. apply EDISP

    flux_cube = model.evaluate_cube(ref_cube)

    from time import time
    t0 = time()
    npred_cube = compute_npred_cube(
        flux_cube, exposure_cube,
        ebounds=flux_cube.energies('edges'),
        integral_resolution=2,
    )
    t1 = time()
    npred_cube_simple = compute_npred_cube_simple(flux_cube, exposure_cube)
    t2 = time()
    print('npred_cube: ', t1 - t0)
    print('npred_cube_simple: ', t2 - t1)
    print('npred_cube sum: {}'.format(np.nansum(npred_cube.data.to('').data)))
    print('npred_cube_simple sum: {}'.format(np.nansum(npred_cube_simple.data.to('').data)))

    # Apply PSF convolution here
    kernels = irfs['psf'].kernels(npred_cube_simple)
    npred_cube_convolved = npred_cube_simple.convolve(kernels)

    # TODO: apply EDISP here!

    # Compute counts as a Poisson fluctuation
    # counts_cube = SkyCube.empty_like(ref_cube)
    # counts_cube.data = np.random.poisson(npred_cube_convolved.data)

    # Debugging output

    print('npred_cube sum: {}'.format(np.nansum(npred_cube.data.to('').data)))
    print('npred_cube_convolved sum: {}'.format(np.nansum(npred_cube_convolved.data.to('').data)))
    # TODO: check that sum after PSF convolution or applying EDISP are the same

    exposure_cube.write('exposure_cube.fits', overwrite=True, format='fermi-exposure')
    flux_cube.write('flux_cube.fits.gz', overwrite=True)
    npred_cube.write('npred_cube.fits.gz', overwrite=True)
    npred_cube_convolved.write('npred_cube_convolved.fits.gz', overwrite=True)

    # npred_cube2 = SkyCube.read('npred_cube.fits.gz')
    # print(npred_cube2)


if __name__ == '__main__':
    main()
