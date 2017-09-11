# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Morphological models for astrophysical gamma-ray sources.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np
from astropy.modeling import Parameter, ModelDefinitionError, Fittable2DModel
from astropy.modeling.models import Gaussian2D
from astropy.utils import lazyproperty
from astropy.coordinates import SkyCoord
from ..core import SkyImage

__all__ = [
    'morph_types',
    'Delta2D',
    # TODO: we need our own model, so that we can add in stuff like XML I/O
    # Copy over the Astropy version and use that throughout Gammapy
    # 'Gaussian2D',
    'Shell2D',
    'Sphere2D',
    'Template2D',
]


class Delta2D(Fittable2DModel):
    """Two dimensional delta function .

    This model can be used for a point source morphology.

    Parameters
    ----------
    amplitude : float
        Peak value of the point source
    x_0 : float
        x position center of the point source
    y_0 : float
        y position center of the point source

    Notes
    -----
    Model formula:

    .. math::

        f(x, y) = \\cdot \\left \\{
                    \\begin{array}{ll}
                        A & :  x = x_0 \\ \\mathrm{and} \\ y = y_0 \\\\
                        0 & : else
                    \\end{array}
                \\right.
    """
    amplitude = Parameter('amplitude')
    x_0 = Parameter('x_0')
    y_0 = Parameter('y_0')

    def __init__(self, amplitude, x_0, y_0, **constraints):
        super(Delta2D, self).__init__(amplitude=amplitude, x_0=x_0,
                                      y_0=y_0, **constraints)

    @staticmethod
    def evaluate(x, y, amplitude, x_0, y_0):
        """Two dimensional delta model function using a local rectangular pixel
        approximation.
        """
        _, grad_x = np.gradient(x)
        grad_y, _ = np.gradient(y)
        x_diff = np.abs((x - x_0) / grad_x)
        y_diff = np.abs((y - y_0) / grad_y)

        x_val = np.select([x_diff < 1], [1 - x_diff], 0)
        y_val = np.select([y_diff < 1], [1 - y_diff], 0)
        return x_val * y_val * amplitude


class Shell2D(Fittable2DModel):
    """Projected homogeneous radiating shell model.

    This model can be used for a shell type SNR source morphology.

    Parameters
    ----------
    amplitude : float
        Value of the integral of the shell function.
    x_0 : float
        x position center of the shell
    y_0 : float
        y position center of the shell
    r_in : float
        Inner radius of the shell
    width : float
        Width of the shell
    r_out : float (optional)
        Outer radius of the shell
    normed : bool (True)
        If set the amplitude parameter corresponds to the integral of the
        function. If not set the 'amplitude' parameter corresponds to the
        peak value of the function (value at :math:`r = r_{in}`).

    Notes
    -----
    Model formula with integral normalization:

    .. math::

        f(r) = A \\frac{3}{2 \\pi (r_{out}^3 - r_{in}^3)} \\cdot \\left \\{
                \\begin{array}{ll}
                    \\sqrt{r_{out}^2 - r^2} - \\sqrt{r_{in}^2 - r^2} & : r < r_{in} \\\\
                    \\sqrt{r_{out}^2 - r^2} & :  r_{in} \\leq r \\leq r_{out} \\\\
                    0 & : r > r_{out}
                \\end{array}
            \\right.

    Model formula with peak normalization:

    .. math::

        f(r) = A \\frac{1}{\\sqrt{r_{out}^2 - r_{in}^2}} \\cdot \\left \\{
                \\begin{array}{ll}
                    \\sqrt{r_{out}^2 - r^2} - \\sqrt{r_{in}^2 - r^2} & : r < r_{in} \\\\
                    \\sqrt{r_{out}^2 - r^2} & :  r_{in} \\leq r \\leq r_{out} \\\\
                    0 & : r > r_{out}
                \\end{array}
            \\right.

    With :math:`r_{out} = r_{in} + \\mathrm{width}`.


    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from gammapy.image.models import Shell2D

        shell = Shell2D(amplitude=100, x_0=25, y_0=25, r_in=10, width=5)
        y, x = np.mgrid[0:50, 0:50]
        plt.imshow(shell(x, y), origin='lower', interpolation='none')
        plt.xlabel('x (pix)')
        plt.ylabel('y (pix)')
        plt.colorbar(label='Brightness (A.U.)')
        plt.grid(False)
        plt.show()
    """
    amplitude = Parameter('amplitude')
    x_0 = Parameter('x_0')
    y_0 = Parameter('y_0')
    r_in = Parameter('r_in')
    width = Parameter('width')

    def __init__(self, amplitude, x_0, y_0, r_in, width=None, r_out=None,
                 normed=True, **constraints):
        if r_out is not None:
            width = r_out - r_in
        if r_out is None and width is None:
            raise ModelDefinitionError("Either specify width or r_out.")

        if not normed:
            self.evaluate = self.evaluate_peak_norm
        super(Shell2D, self).__init__(amplitude=amplitude, x_0=x_0,
                                      y_0=y_0, r_in=r_in, width=width,
                                      **constraints)

    @staticmethod
    def evaluate(x, y, amplitude, x_0, y_0, r_in, width):
        """Two dimensional Shell model function normed to integral"""
        rr = (x - x_0) ** 2 + (y - y_0) ** 2
        rr_in = r_in ** 2
        rr_out = (r_in + width) ** 2

        # Because np.select evaluates on the whole rr array
        # we have to catch the invalid value warnings
        # Note: for r > r_out 'np.select' fills automatically zeros!
        with np.errstate(invalid='ignore'):
            values = np.select([rr <= rr_in, rr <= rr_out],
                               [np.sqrt(rr_out - rr) - np.sqrt(rr_in - rr),
                                np.sqrt(rr_out - rr)])
        return amplitude * values / (2 * np.pi / 3 *
                                     (rr_out * (r_in + width) - rr_in * r_in))

    @staticmethod
    def evaluate_peak_norm(x, y, amplitude, x_0, y_0, r_in, width):
        """Two dimensional Shell model function normed to peak value"""
        rr = (x - x_0) ** 2 + (y - y_0) ** 2
        rr_in = r_in ** 2
        rr_out = (r_in + width) ** 2

        # Because np.select evaluates on the whole rr array
        # we have to catch the invalid value warnings
        # Note: for r > r_out 'np.select' fills automatically zeros!
        with np.errstate(invalid='ignore'):
            values = np.select([rr <= rr_in, rr <= rr_out],
                               [np.sqrt(rr_out - rr) - np.sqrt(rr_in - rr),
                                np.sqrt(rr_out - rr)])
        return amplitude * values / np.sqrt(rr_out - rr_in)

    @property
    def bounding_box(self):
        """
        Tuple defining the default ``bounding_box`` limits.
        ``((y_low, y_high), (x_low, x_high))``
        """
        r_out = self.r_in + self.width
        return ((self.y_0 - r_out, self.y_0 + r_out),
                (self.x_0 - r_out, self.x_0 + r_out))

    def to_sherpa(self, name='default'):
        """Convert to a `~sherpa.models.ArithmeticModel`.

        Parameters
        ----------
        name : str, optional
            Name of the sherpa model instance
        """
        from sherpa.astro.models import Shell2D
        model = Shell2D(name=name)

        model.xpos = self.x_0.value
        model.ypos = self.y_0.value
        model.ampl = self.amplitude.value
        # Note: we checked, the Sherpa `r0` is our `r_in`.
        model.r0 = self.r_in.value
        model.width = self.width.value

        return model


class Sphere2D(Fittable2DModel):
    """Projected homogeneous radiating sphere model.

    This model can be used for a simple PWN source morphology.

    Parameters
    ----------
    amplitude : float
        Value of the integral of the sphere function
    x_0 : float
        x position center of the sphere
    y_0 : float
        y position center of the sphere
    r_0 : float
        Radius of the sphere
    normed : bool (True)
        If set the amplitude parameter corresponds to the integral of the
        function. If not set the 'amplitude' parameter corresponds to the
        peak value of the function (value at :math:`r = 0`).

    Notes
    -----
    Model formula with integral normalization:

    .. math::

        f(r) = A \\frac{3}{4 \\pi r_0^3} \\cdot \\left \\{
                \\begin{array}{ll}
                    \\sqrt{r_0^2 - r^2} & :  r \\leq r_0 \\\\
                    0 & : r > r_0
                \\end{array}
            \\right.

    Model formula with peak normalization:

    .. math::

        f(r) = A \\frac{1}{r_0} \\cdot \\left \\{
                \\begin{array}{ll}
                    \\sqrt{r_0^2 - r^2} & :  r \\leq r_0 \\\\
                    0 & : r > r_0
                \\end{array}
            \\right.

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from gammapy.image.models import Sphere2D

        sphere = Sphere2D(amplitude=100, x_0=25, y_0=25, r_0=20)
        y, x = np.mgrid[0:50, 0:50]
        plt.imshow(sphere(x, y), origin='lower', interpolation='none')
        plt.xlabel('x (pix)')
        plt.ylabel('y (pix)')
        plt.colorbar(label='Brightness (A.U.)')
        plt.grid(False)
        plt.show()
    """
    amplitude = Parameter('amplitude')
    x_0 = Parameter('x_0')
    y_0 = Parameter('y_0')
    r_0 = Parameter('r_0')

    def __init__(self, amplitude, x_0, y_0, r_0, normed=True, **constraints):
        if not normed:
            self.evaluate = self.evaluate_peak_norm
        super(Sphere2D, self).__init__(amplitude=amplitude, x_0=x_0,
                                       y_0=y_0, r_0=r_0, **constraints)

    @staticmethod
    def evaluate(x, y, amplitude, x_0, y_0, r_0):
        """Two dimensional Sphere model function normed to integral"""
        rr = (x - x_0) ** 2 + (y - y_0) ** 2
        rr_0 = r_0 ** 2

        # Because np.select evaluates on the whole rr array
        # we have to catch the invalid value warnings
        with np.errstate(invalid='ignore'):
            values = np.select([rr <= rr_0, rr > rr_0], [2 * np.sqrt(rr_0 - rr), 0])
        return amplitude * values / (4 / 3. * np.pi * rr_0 * r_0)

    @staticmethod
    def evaluate_peak_norm(x, y, amplitude, x_0, y_0, r_0):
        """Two dimensional Sphere model function normed to peak value"""
        rr = (x - x_0) ** 2 + (y - y_0) ** 2
        rr_0 = r_0 ** 2

        # Because np.select evaluates on the whole rr array
        # we have to catch the invalid value warnings
        with np.errstate(invalid='ignore'):
            values = np.select([rr <= rr_0, rr > rr_0], [np.sqrt(rr_0 - rr), 0])
        return amplitude * values / r_0

    @property
    def bounding_box(self):
        """
        Tuple defining the default ``bounding_box`` limits.
        ``((y_low, y_high), (x_low, x_high))``
        """
        return ((self.y_0 - self.r_0, self.y_0 + self.r_0),
                (self.x_0 - self.r_0, self.x_0 + self.r_0))


class Template2D(Fittable2DModel):
    """Two dimensional table model.

    Parameters
    ----------
    amplitude : float
        Amplitude of the template model.
    """
    amplitude = Parameter('amplitude')

    def __init__(self, image, amplitude=1., frame='galactic', **constraints):
        self.image = image
        self.frame = frame
        super(Template2D, self).__init__(amplitude=amplitude, **constraints)

    @lazyproperty
    def _interpolate_data(self):
        """Interpolate data using `~scipy.interpolate.RegularGridInterpolator`."""
        from scipy.interpolate import RegularGridInterpolator
        # TODO: move e.g. to SkyImage.interpolate()

        y = np.arange(self.image.data.shape[0])
        x = np.arange(self.image.data.shape[1])

        data_normed = self.image.data / self.image.data.sum()
        data_normed /= self.image.solid_angle().to('deg2').value

        f = RegularGridInterpolator((y, x), data_normed, fill_value=0,
                                    bounds_error=False)

        def interpolate(y, x, method='linear'):
            shape = y.shape
            coords = np.column_stack([y.flat, x.flat])
            val = f(coords, method=method)
            return val.reshape(shape)

        return interpolate

    @classmethod
    def read(cls, filename, **kwargs):
        """Read spatial template model from FITS image.

        Parameters
        ----------
        filename : str
            Fits image filename.
        """
        template = SkyImage.read(filename, **kwargs)
        return cls(template)

    def evaluate(self, x, y, amplitude):
        # TODO: don't hardcode Galactic frame!!!
        coord = SkyCoord(x, y, frame='galactic', unit='deg')
        x_pix, y_pix = self.image.wcs_skycoord_to_pixel(coord)
        values = self._interpolate_data(y_pix, x_pix)
        return amplitude * values

    @property
    def bounding_box(self):
        width = self.image.width.deg
        height = self.image.height.deg
        center = self.image.center
        x_0 = center.data.lon.wrap_at('180d').deg
        y_0 = center.data.lat.deg
        return ((y_0 - height / 2, y_0 + height / 2),
                (x_0 - width, x_0 + height / 2))


# TODO: change this to a model registry
morph_types = OrderedDict()
morph_types.__doc__ = """Spatial model registry (`~collections.OrderedDict`)."""
morph_types['delta2d'] = Delta2D
morph_types['gauss2d'] = Gaussian2D
morph_types['shell2d'] = Shell2D
morph_types['sphere2d'] = Sphere2D
morph_types['template2d'] = Template2D
