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


def _extended_image(catalog, reference_cube):
    """Reprojects and adds extended source images to a larger survey image.
    """
    # This import is here instead of at the top to avoid an ImportError
    # due to circular dependencies
    from ..catalog import fetch_fermi_extended_sources
    from ..cube import SkyCube

    # Note that the first extended source fits file is unreadable...
    hdu_list = fetch_fermi_extended_sources(catalog)[1:]
    for source in hdu_list:
        source_wcs = WCS(source.header)
        source_spec_cube = SkyCube(data=Quantity(np.array([source.data]), ''),
                                   wcs=source_wcs, energy=energy)
        new_source_cube = source_spec_cube.reproject_to(reference_cube)
        # TODO: Fix this hack
        reference_cube.data = reference_cube.data + np.nan_to_num(new_source_cube.data * 1e-30)
    return reference_cube.data[0]


def _source_image(catalog, reference_cube, sim_table=None, total_flux=True):
    """Adds point sources to a larger survey image.
    """
    new_image = np.zeros_like(reference_cube.data, dtype=np.float64)
    if sim_table is None:
        source_table = catalog_table(catalog, energy_bands=False)
    else:
        source_table = sim_table

    energies = source_table.meta['Energy Bins']
    wcs_reference = reference_cube.wcs
    footprint = wcs_reference.calc_footprint()
    glon_max, glon_min = footprint[0][0], footprint[2][0] - 360
    glat_min, glat_max = footprint[0][1], footprint[1][1]

    for source in np.arange(len(source_table['flux'])):
        lon = source_table['GLON'][source]
        if lon >= 180:
            lon = lon - 360
        if (glon_min < lon) & (lon < glon_max):
            lat = source_table['GLAT'][source]
            if (glat_min < lat) & (lat < glat_max):
                flux = source_table['flux'][source]
                wcs = reference_cube.wcs
                origin = 0  # convention for gammapy
                x, y = wcs.wcs_world2pix(lon, lat, origin)
                xi, yi = x.astype(int), y.astype(int)
                new_image[yi, xi] = new_image[yi, xi] + flux

    if total_flux:
        factor = source_table['flux'].sum() / new_image.sum()
    else:
        factor = 1

    return new_image * factor, energies


def catalog_image(reference, psf, catalog='1FHL', source_type='point',
                  total_flux=False, sim_table=None):
    """Creates an image from a simulated catalog, or from 1FHL or 2FGL sources.

    Parameters
    ----------
    reference : `~astropy.io.fits.ImageHDU`
        Reference Image HDU. The output takes the shape and resolution of this.
    psf : `~gammapy.irf.EnergyDependentTablePSF`
        Energy dependent Table PSF object for image convolution.
    catalog : {'1FHL', '2FGL', 'simulation'}
        Flag which source catalog is to be used to create the image.
        If 'simulation' is used, sim_table must also be provided.
    source_type : {'point', 'extended', 'all'}
        Specify whether point or extended sources should be included, or both.
        TODO: Currently only 'point' is implemented.
    total_flux : bool
        Specify whether to conserve total flux.
    sim_table : `~astropy.table.Table`
        Table of simulated point sources. Only required if ``catalog='simulation'``

    Returns
    -------
    out_cube : `~gammapy.data.SkyCube`
        2D Spectral cube containing the image.

    Notes
    -----
    This is currently only implemented for a single energy band.
    """
    from scipy.ndimage import convolve
    # This import is here instead of at the top to avoid an ImportError
    # due to circular dependencies
    from ..cube import SkyCube
    from ..spectrum import LogEnergyAxis

    wcs = WCS(reference.header)
    # Uses dummy energy for now to construct spectral cube
    # TODO : Fix this hack
    reference_cube = SkyCube(data=Quantity(np.array(reference.data), ''),
                             wcs=wcs)

    if source_type == 'extended':
        raise NotImplementedError
        # TODO: Currently fluxes are not correct for extended sources.
        new_image = _extended_image(catalog, reference_cube)

    elif source_type == 'point':
        new_image, energy = _source_image(catalog, reference_cube,
                                          sim_table, total_flux)

    elif source_type == 'all':
        raise NotImplementedError
        # TODO: Currently Extended Sources do not work
        extended = _extended_image(catalog, reference_cube)
        point_source = _source_image(catalog, reference_cube, total_flux=True)[0]
        new_image = extended + point_source

    else:
        raise ValueError

    total_point_image = SkyCube(data=new_image, wcs=wcs, energy_axis=LogEnergyAxis(energy))
    convolved_cube = new_image.copy()

    psf = psf.table_psf_in_energy_band(Quantity([np.min(energy).value,
                                                 np.max(energy).value], energy.unit))

    resolution = abs(reference.header['CDELT1'])
    ref = total_point_image.sky_image_ref
    kernel_array = psf.kernel(ref, normalize=True)

    convolved_cube = convolve(new_image, kernel_array, mode='constant')

    out_cube = SkyCube(data=Quantity(convolved_cube, ''),
                       wcs=total_point_image.wcs, energy_axis=LogEnergyAxis(energy))

    return out_cube


def catalog_table(catalog, energy_bands=False):
    """Creates catalog table from published source catalog.

    This creates a table of catalog sources, positions and fluxes for an
    indicated published source catalog - either 1FHL or 2FGL. This should
    be used to in instances where a table is required, for instance as an
    input for the `~gammapy.image.catalog_image` function.

    Parameters
    ----------
    catalog : {'1FHL', '2FGL'}
        Catalog to load.
    energy_bands : bool
        Whether to return catalog in energy bands.

    Returns
    -------
    table : `~astropy.table.Table`
        Point source catalog table.
    """
    # This import is here instead of at the top to avoid an ImportError
    # due to circular dependencies
    from ..catalog import fetch_fermi_catalog

    data = []
    cat_table = fetch_fermi_catalog(catalog, 'LAT_Point_Source_Catalog')

    for source in np.arange(len(cat_table)):
        glon = cat_table['GLON'][source]
        glat = cat_table['GLAT'][source]

        # Different from here between each catalog because of different catalog header names
        if catalog in ['1FHL', 'simulation']:
            energy = Quantity([10, 30, 100, 500], 'GeV')

            if energy_bands:
                Flux_10_30 = cat_table['Flux10_30GeV'][source]
                Flux_30_100 = cat_table['Flux30_100GeV'][source]
                Flux_100_500 = cat_table['Flux100_500GeV'][source]
                row = dict(Source_Type='PointSource',
                           GLON=glon, GLAT=glat, Flux10_30=Flux_10_30,
                           Flux30_100=Flux_30_100, Flux100_500=Flux_100_500)

            else:
                flux_bol = cat_table['Flux'][source]
                row = dict(Source_Type='PointSource',
                           GLON=glon, GLAT=glat, flux=flux_bol)

        elif catalog == '2FGL':
            energy = Quantity([30, 100, 300, 1000, 3000, 10000, 100000], 'GeV')

            if not energy_bands:
                flux_bol = cat_table['Flux_Density'][source]
                row = dict(Source_Type='PointSource',
                           GLON=glon,
                           GLAT=glat,
                           flux=flux_bol)

            else:
                Flux_30_100 = cat_table['Flux30_100'][source]
                Flux_100_300 = cat_table['Flux100_300'][source]
                Flux_300_1000 = cat_table['Flux300_1000'][source]
                Flux_1000_3000 = cat_table['Flux1000_3000'][source]
                Flux_3000_10000 = cat_table['Flux3000_10000'][source]
                Flux_10000_100000 = cat_table['Flux10000_100000'][source]
                row = dict(Source_Type='PointSource',
                           Source_Name=source,
                           GLON=glon,
                           GLAT=glat,
                           Flux_30_100=Flux_30_100,
                           Flux_100_300=Flux_100_300,
                           Flux_300_1000=Flux_300_1000,
                           Flux_1000_3000=Flux_1000_3000,
                           Flux_3000_10000=Flux_3000_10000,
                           Flux_10000_100000=Flux_10000_100000)

        data.append(row)

    table = Table(data)
    table.meta['Energy Bins'] = energy

    return table
