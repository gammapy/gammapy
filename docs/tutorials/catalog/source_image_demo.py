"""Produces an image from 1FHL catalog point sources.
"""
from aplpy import FITSFigure
from gammapy.datasets import FermiGalacticCenter
from gammapy.image import make_empty_image, catalog_image
from gammapy.irf import EnergyDependentTablePSF

# Create image of defined size
reference = make_empty_image(nxpix=300, nypix=100, binsz=1)
psf_file = FermiGalacticCenter.filenames()['psf']
psf = EnergyDependentTablePSF.read(psf_file)

# Create image
image = catalog_image(reference, psf, catalog='1FHL', source_type='point',
                  total_flux='True')

# Plot
fig = FITSFigure(image.to_fits()[0])
fig.show_grayscale(stretch='linear', interpolation='none')
fig.add_colorbar()
