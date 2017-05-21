# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from .core import SkyCube

__all__ = [
    'CombinedModel3D',
]


class CombinedModel3D(object):
    """Combine spatial and spectral model into a 3D model.

    TODO: give move infos and an example how spatial models
    must be normalised to integrate to 1 and caveats about
    binning effects, i.e. how too small bins or very small
    sources will lead to incorrect spectral results!

    At the moment this model has no built-in integration.
    I.e. it's left up to callers to:

    * integrate over energy bins
    * integrate over spatial pixels

    TODO: This is a prototype, and the evaluation scheme might change!
    Feedback on what you'd like to do and whether this class is working
    for you or not is highly welcome!!!

    Parameters
    ----------
    spatial_model : `~gammapy.image.models.SpatialModel`
        Spatial model (must be normalised to integrate to 1)
    spectral_model : `~gammapy.spectrum.models.SpectralModel`
        Spectral model

    Examples
    --------
    Create a `CombinedModel3D`, i.e. one that factors into a spatial
    and position-independent spectral part::

        import astropy.units as u
        from gammapy.image.models import Shell2D
        from gammapy.spectrum.models import PowerLaw
        from gammapy.cube.models import CombinedModel3D

        spatial_model = Shell2D(
            amplitude=1, x_0=3, y_0=4, r_in=5, width=6, normed=True,
        )
        spectral_model = PowerLaw(
            index=2, amplitude=1 * u.Unit('cm-2 s-1 TeV-1'), reference=1 * u.Unit('TeV'),
        )
        model = CombinedModel3D(spatial_model, spectral_model)

    Look at the model you created::

        >>> model
        CombinedModel3D(spatial_model=<Shell2D(amplitude=1.0, x_0=3.0, y_0=4.0, r_in=5.0, width=6.0)>, spectral_model=PowerLaw())
        >>> print(model.spectral_model)
        PowerLaw

        Parameters:

               name     value   error       unit      min  max  frozen
            --------- --------- ----- --------------- ---- ---- ------
                index 2.000e+00   nan                    0 None  False
            amplitude 1.000e+00   nan 1 / (cm2 s TeV)    0 None  False
            reference 1.000e+00   nan             TeV None None   True

        >>> print(model.spatial_model)
        Model: Shell2D
        Inputs: ('x', 'y')
        Outputs: ('z',)
        Model set size: 1
        Parameters:
            amplitude x_0 y_0 r_in width
            --------- --- --- ---- -----
                  1.0 3.0 4.0  5.0   6.0

    Evaluate the model at a given point::

        >>> import astropy.units as u
        >>> model.evaluate(lon=0.1 * u.deg, lat=0.2 * u.deg, energy='1 TeV')
        <Quantity [[[ 0.00334177]]] 1 / (cm2 deg2 s TeV)>

    Evaluate the model on a sky cube::

        >>> # TODO: add example.
    """

    def __init__(self, spatial_model, spectral_model):
        self.spatial_model = spatial_model
        self.spectral_model = spectral_model

    def __repr__(self):
        fmt = '{}(spatial_model={!r}, spectral_model={!r})'
        return fmt.format(self.__class__.__name__, self.spatial_model, self.spectral_model)

    # TODO: decide on coordinate order and make it uniform within Gammapy
    # see SkyCube.lookup(skycoord, energy)
    # see sherpy_.CombinedModel3D(e_lo, e_hi, x, y)
    def evaluate(self, lon, lat, energy):
        """Evaluate the model at given points.

        Return differential surface brightness cube.
        At the moment in units: ``cm-2 s-1 TeV-1 sr-1``

        TODO: currently spatial models don't support units,
        and we have hard-coded in this evaluate the assumption
        that they return their result in unit ``deg-2``

        Parameters
        ----------
        lon, lat : `~astropy.units.Quantity`
            Spatial coordinates
        energy : `~astropy.units.Quantity`
            Energy coordinate

        Returns
        -------
        value : `~astropy.units.Quantity`
            Model value at the given point.
        """
        # Evaluate the spatial and spectral models
        # TODO: change spatial models to work with quantities,
        # so that these explicit unit conversions become unnecessary.
        a = self.spatial_model(lon.to('deg').value, lat.to('deg').value) * u.Unit('deg-2')
        b = self.spectral_model(energy)

        # TODO: make this more general to support all possible use cases (in an efficient way).
        # is this a good pattern?
        # shape = SkyCube.compute_shape(lon, lat, energy)
        # val = a.reshape(tuple(shape)) * b.reshape(tuple(shape))

        # For now, we only support the case of scalar or 1-dim energy
        # where the following broadcasting works:
        val = a * np.atleast_1d(b)[:, np.newaxis, np.newaxis]

        return val

    def evaluate_cube(self, ref_cube):
        """Evaluate the model on coordinates given by a reference sky cube.

        Parameters
        ----------
        ref_cube : `~gammapy.cube.SkyCube`
            Reference sky cube

        Returns
        -------
        model_cube : `~gammapy.cube.SkyCube`
            Sky cube with data filled with evaluated model values.
            Units: ``cm-2 s-1 TeV-1 sr-1``
        """
        # Extract grid of coordinates (lon, lat, energy) from the cube
        coords = ref_cube.sky_image_ref.coordinates()
        lon = coords.data.lon
        lat = coords.data.lat
        energy = ref_cube.energies(mode='center')

        data = self.evaluate(lon, lat, energy)

        # TODO: Fix this so that explicit unit conversion here become unnecessary
        # The problem at the moment is that here we have quantities with
        # a unit scale != 1, and `.write('cube.fits')` errors out for a SkyCube with such a Quantity.
        # This is a temp quick fix:
        data = data.to('cm-2 s-1 TeV-1 sr-1')

        return SkyCube(data=data, wcs=ref_cube.wcs, energy_axis=ref_cube.energy_axis)
