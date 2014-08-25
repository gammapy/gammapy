# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Sky and image region classes and functions.

References:

* https://pypi.python.org/pypi/pyregion
* http://ds9.si.edu/doc/ref/region.html
"""
from __future__ import print_function, division


__all__ = ['make_ds9_region',
           'Circle',
           'SkyCircle',
           ]


class Circle(object):
    """Little helper to work with regions.
    """

    @staticmethod
    def write(x=0, y=0, radius=1, system='galactic',
              attrs={}):
        string = '{system};circle({x},{y},{radius})'
        if attrs:
            string += ' #'
            for key, val in attrs.items():
                if isinstance(val, str):  # and ' ' in val:
                    val = '{{%s}}' % val
                string += ' {0}={1}'.format(key, val)
        string += '\n'
        return string.format(**locals())

    def parse(self, string):
        raise NotImplementedError
        t = string.split(',')
        x, y, radius = float(t[0]), float(t[1]), float(t[2])
        return x, y, radius  # , system, text, color


class SkyCircle(object):
    """A circle on the sky.

    Parameters
    ----------
    center : `~astropy.coordinates.SkyCoord`
        Circle center coordinate
    radius : `~astropy.coordinates.Angle`
        Circle radius
    """

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def contains(self, coordinate):
        """Checks if the coordinate lies inside the circle.

        Parameters
        ----------
        coordinate : `~astropy.coordinates.SkyCoord`
            Coordinate to check for containment.

        Returns
        -------
        contains : bool
            Does this region contain the coordinate?
        """
        return self.center.separation(coordinate) <= self.radius

    def intersects(self, other):
        """Checks if two sky circles overlap.

        Parameters
        ----------
        other : `SkyCircle`
            Other region.
        """
        return self.center.separation(other.center) <= self.radius + other.radius


def make_ds9_region(source, attrs, scale=3):
    """Make ds9 region strings.

    * circle x y radius
    * ellipse x y radius radius angle
    * annulus x y inner outer

    Parameters
    ----------
    source : dict
        Dictionary with 'GLON', 'GLAT', 'Type', ... keys.
    attrs : dict
        Dictionary of attributes
    scale : float
        Gaussian scale factor

    Returns
    -------
    region_string : str
        DS9 region string

    Examples
    --------
    TODO
    """
    x, y = source['GLON'], source['GLAT']

    if source['Type'] == 'Gaussian' or 'NormGaussian':
        radius = scale * float(source['Sigma'])
        pars = [x, y, radius]
        return _region_string('circle', pars, attrs)
    elif source['Type'] == 'ElongatedGaussian':
        # We scale the ellipse so that the major axis has size theta
        major = scale * float(source['Sigma'])
        minor = scale * float(source['Minor'])
        angle = source['PositionAngle']
        pars = [x, y, major, minor, angle]
        return _region_string('ellipse', pars, attrs)
    elif source['Type'] == 'Shell':
        inner = float(source['InnerRadius'])
        outer = inner + float(source['Width'])
        pars = [x, y, inner, outer]
        return _region_string('annulus', pars, attrs)


def _region_string(shape, pars, attrs, system='galactic'):
    pars = ','.join([str(_) for _ in pars])
    string = '{system};{shape}({pars})'
    if attrs:
        string += ' #'
        for key, val in attrs.items():
            if isinstance(val, str):  # and ' ' in val:
                val = '{{%s}}' % val
            string += ' {0}={1}'.format(key, val)
    string += '\n'
    return string.format(**locals())
