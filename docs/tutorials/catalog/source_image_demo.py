"""Produces an image from 1FHL catalog point sources.
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.visualization import simple_norm
from gammapy.image import CatalogImageEstimator, SkyImage
from gammapy.irf import EnergyDependentTablePSF
from gammapy.catalog import SourceCatalog3FHL

# Create reference image
reference = SkyImage.empty(nxpix=300, nypix=100, binsz=0.1)
emin, emax = [10, 500] * u.GeV

fermi_3fhl = SourceCatalog3FHL()
estimator = CatalogImageEstimator(reference=reference, emin=emin, emax=emax)

result = estimator.run(fermi_3fhl)

flux = result['flux'].smooth(radius=0.2 * u.deg)

norm = simple_norm(flux.data, stretch='log')
flux.show(norm=norm)