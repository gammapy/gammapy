# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Cube analysis utility functions.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from ..utils.energy import EnergyBounds
from .core import SkyCube

__all__ = [
    'compute_npred_cube',
]


# TODO: remove this!
# If useful, add to the new SkyModelMapEvaluator
def compute_npred_cube(flux_cube, exposure_cube, ebounds,
                       integral_resolution=10):
    """Compute predicted counts cube.

    Parameters
    ----------
    flux_cube : `SkyCube`
        Flux cube, really differential surface brightness in 'cm-2 s-1 TeV-1 sr-1'
    exposure_cube : `SkyCube`
        Exposure cube
    ebounds : `~astropy.units.Quantity`
        Energy bounds for the output cube
    integral_resolution : int (optional)
        Number of integration steps in energy bin when computing integral flux.

    Returns
    -------
    npred_cube : `SkyCube`
        Predicted counts cube with energy bounds as given by the input ``ebounds``.

    Examples
    --------
    Load an example dataset::

        from gammapy.datasets import FermiGalacticCenter
        from gammapy.utils.energy import EnergyBounds
        from gammapy.irf import EnergyDependentTablePSF
        from gammapy.cube import SkyCube, compute_npred_cube
        
        filenames = FermiGalacticCenter.filenames()
        flux_cube = SkyCube.read(filenames['diffuse_model'], format='fermi-background')
        exposure_cube = SkyCube.read(filenames['exposure_cube'], format='fermi-exposure')
        psf = EnergyDependentTablePSF.read(filenames['psf'])
        
    Compute an ``npred`` cube and a PSF-convolved version::

        flux_cube = flux_cube.reproject(exposure_cube)
        ebounds = EnergyBounds([10, 30, 100, 500], 'GeV')
        npred_cube = compute_npred_cube(flux_cube, exposure_cube, ebounds)
        
        kernels = psf.kernels(npred_cube)
        npred_cube_convolved = npred_cube.convolve(kernels)
    """
    # Make an empty cube with the requested energy binning
    sky_geom = exposure_cube.sky_image_ref
    energies = EnergyBounds(ebounds)
    npred_cube = SkyCube.empty_like(sky_geom, energies=energies, unit='', fill=np.nan)

    # Process and fill one energy bin at a time
    for idx in range(len(ebounds) - 1):
        emin, emax = ebounds[idx: idx + 2]
        ecenter = np.sqrt(emin * emax)

        flux = flux_cube.sky_image_integral(emin, emax, interpolation='linear', nbins=integral_resolution)
        exposure = exposure_cube.sky_image(ecenter, interpolation='linear')
        solid_angle = exposure.solid_angle()

        npred = flux.data * exposure.data * solid_angle

        npred_cube.data[idx] = npred.to('')

    return npred_cube
