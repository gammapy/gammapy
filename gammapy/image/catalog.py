# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Make an image from a source catalog, or simulated catalog, e.g 1FHL 2FGL etc
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np
from astropy.wcs import WCS
from astropy.units import Quantity
from astropy import units as u
from astropy.table import Table
from .core import SkyImage
from .lists import SkyImageList

__all__ = [
    'CatalogImageEstimator',
]

BBOX_DELTA2D_PIX = 5


class CatalogImageEstimator(object):
    """Compute model image for given energy band from a catalog.

    Sources are only filled when their center lies within the image boundaries.

    Parameters
    ----------
    reference : `~gammapy.image.SkyImage`
        Reference sky image.
    emin : `~astropy.units.Quantity`
        Lower bound of energy range.
    emax : `~astropy.units.Quantity`
        Upper bound of energy range.

    Examples
    --------

    Here is an example how to compute a flux image from a catalog:

        from astropy import units as u
        from gammapy.image import SkyImage, CatalogImageEstimator
        from gammapy.catalog import SourceCatalogGammaCat

        reference = SkyImage.empty(xref=265, yref=-1.5, nxpix=201,
                                   nypix=201, binsz=0.04)

        image_estimator = CatalogImageEstimator(reference=reference,
                                                emin=1 * u.TeV,
                                                emax=10 * u.TeV)

        catalog = SourceCatalogGammaCat()
        result = image_estimator.run(catalog)
        result['flux'].show()

    Currently the `CatalogImageEstimator` class does not support to compute model
    cubes of catalogs. But this can achieved with only a little more of python code:

        from astropy import units as u
        from gammapy.image import CatalogImageEstimator, SkyImage
        from gammapy.cube import SkyCube
        from gammapy.catalog import SourceCatalogGammaCat
        from gammapy.utils.energy import EnergyBounds

        reference = SkyImage.empty(xref=265, yref=-1.5, nxpix=201,
                                   nypix=201, binsz=0.04)

        energies = EnergyBounds.equal_log_spacing(1 * u.TeV, 100 * u.TeV, 3)

        flux_cube = SkyCube.empty_like(reference=reference, energies=energies)

        catalog = SourceCatalogGammaCat()

        for idx in range(energies.size - 1):
            image_estimator = CatalogImageEstimator(reference=reference,
                                                    emin=energies[idx],
                                                    emax=energies[idx + 1])

            result = image_estimator.run(catalog)
            flux_cube.data[idx, :, :] = result['flux'].data

        flux_cube.show()

    """

    def __init__(self, reference, emin, emax):
        self.reference = reference
        self.parameters = OrderedDict(emin=emin, emax=emax)

    def flux(self, catalog):
        """Compute flux image from catalog.

        Sources are only filled when their center lies within the image boundaries.

        Parameters
        ----------
        catalog : `~gammapy.catalog.SourceCatalog`
            Source catalog instance.

        Returns
        -------
        image :  `~gammapy.image.SkyImage`
            Flux sky image.
        """
        from ..catalog.gammacat import NoDataAvailableError
        p = self.parameters
        image = SkyImage.empty_like(self.reference)

        selection = catalog.select_image_region(image)

        for source in selection:
            try:
                spatial_model = source.spatial_model(emin=p['emin'], emax=p['emax'])
            # TODO: remove this error handling and add selection to SourceCatalog
            # class
            except (NotImplementedError, NoDataAvailableError):
                continue

            if source.is_pointlike:
                # use 5 pixel bbox for point-like models
                size = BBOX_DELTA2D_PIX * image.wcs_pixel_scale().to('deg')
            else:
                height, width = np.diff(spatial_model.bounding_box)
                size = (float(height) * u.deg, float(width) * u.deg)

            cutout = image.cutout(source.position, size=size)

            if source.is_pointlike:
                solid_angle = 1.
            else:
                solid_angle = cutout.solid_angle().to('deg2').value

            # evaluate model on smaller image and paste
            c = cutout.coordinates()
            l, b = c.galactic.l.wrap_at('180d'), c.galactic.b
            cutout.data = spatial_model(l.deg, b.deg) * solid_angle
            image.paste(cutout)

        return image

    def run(self, catalog, which='flux'):
        """Run catalog image estimator.

        Parameters
        ----------
        catalog : `~gammapy.catalog.SourceCatalog`
            Source catalog instance.

        Returns
        -------
        sky_images : `~gammapy.image.SkyImageList`
            List of sky images
        """
        result = SkyImageList()

        # TODO: add input image list and computed derived quantities such as
        # excess, psf convolution etc.
        if 'flux' in which:
            result['flux'] = self.flux(catalog)

        return result
