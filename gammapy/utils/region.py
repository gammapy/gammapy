# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utils to work with ds9 regions"""
from __future__ import print_function, division


__all__ = ['make_ds9_region', 'Circle']


class Circle(object):
    """Little helper to work with regions"""
    @staticmethod
    def write(x=0, y=0, radius=1, system='galactic',
              attrs={}):
        string = '{system};circle({x},{y},{radius})'
        if attrs:
            string += ' #'
            for key, val in attrs.items():
                if isinstance(val, str):  # and ' ' in val:
                    val = '{{%s}}' % val
                string += ' %s=%s' % (key, val)
        string += '\n'
        return string.format(**locals())

    def parse(self, string):
        raise NotImplementedError
        t = string.split(',')
        x, y, radius = float(t[0]), float(t[1]), float(t[2])
        return x, y, radius  # , system, text, color


def _region_string(shape, pars, attrs, system='galactic'):
    pars = ','.join([str(_) for _ in pars])
    string = '{system};{shape}({pars})'
    if attrs:
        string += ' #'
        for key, val in attrs.items():
            if isinstance(val, str):  # and ' ' in val:
                val = '{{%s}}' % val
            string += ' %s=%s' % (key, val)
    string += '\n'
    return string.format(**locals())


def make_ds9_region(source, theta, attrs, SCALE=3):
    """
    circle x y radius
    ellipse x y radius radius angle
    annulus x y inner outer

    SCALE=Gaussian scale factor
    """
    x, y = source['GLON'], source['GLAT']
    if source['Type'] == 'Gaussian' or 'NormGaussian':
        radius = SCALE * float(source['Sigma'])
        pars = [x, y, radius]
        return _region_string('circle', pars, attrs)
    elif source['Type'] == 'ElongatedGaussian':
        # We scale the ellipse so that the major axis has size theta
        major = SCALE * float(source['Sigma'])
        minor = SCALE * float(source['Minor'])
        angle = source['PositionAngle']
        pars = [x, y, major, minor, angle]
        return _region_string('ellipse', pars, attrs)
    elif source['Type'] == 'Shell':
        inner = float(source['InnerRadius'])
        outer = inner + float(source['Width'])
        pars = [x, y, inner, outer]
        return _region_string('annulus', pars, attrs)
