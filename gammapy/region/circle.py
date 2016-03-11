# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from .core import SkyRegion, PixRegion

__all__ = [
    'PixCircleRegion',
    'SkyCircleRegion',
]


class PixCircleRegion(PixRegion):
    """
    Circular region, defined in pixel coordinates.

    Parameters
    ----------
    pos : tuple, list, array
        Pixel coordinates of the circle center
    radius : float
        Circle radius, in pixels
    """

    def __init__(self, pos, radius):

        self.pos = (pos[0], pos[1])
        self.radius = radius

    def to_sky(self, wcs, frame='galactic'):
        """
        Return a `~gammapy.regions.SkyCircleRegion`.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            WCS object
        """
        val = pixel_to_skycoord(self.pos[0], self.pos[1], wcs, mode='wcs', origin=1)
        if frame == 'galactic':
            sky_position = val.galactic
        elif frame == 'icrs':
            sky_position = val.icrs

        sky_radius = Angle(self.radius * np.abs(wcs.wcs.cdelt[0]), 'deg')

        return SkyCircleRegion(sky_position, sky_radius)

    def offset(self, pos):
        """
        Compute offset wrt to a certain pixel position

        Parameters
        ----------
        pos : tuple
            Position to which offset is computed

        Returns
        -------
        offset : float
            Offset in pix

        """
        x2 = (self.pos[0] - pos[0]) ** 2
        y2 = (self.pos[1] - pos[1]) ** 2
        offset = np.sqrt(x2 + y2)
        return offset

    def angle(self, pos):
        """
        Compute angle wrt to a certain pixel position

        Parameters
        ----------
        pos : tuple
            Position to which angle is computed

        Returns
        -------
        angle : `~astropy.units.Quantity`
            Angle
        """
        dx = self.pos[0] - pos[0]
        dy = self.pos[1] - pos[1]
        angle = np.arctan2(dx, dy)
        return Angle(angle, 'rad')

    def is_inside_exclusion(self, exclusion_mask):
        """
        Check if region overlaps with a given exclusion mask

        Parameters
        ----------
        exclusion_mask : `~gammapy.region.ExclusionMask`
            Exclusion mask

        Returns
        -------
        bool
        """
        x, y = self.pos
        excl_dist = exclusion_mask.distance_image
        val = excl_dist[np.round(y).astype(int), np.round(x).astype(int)]
        return val < self.radius

    def to_mpl_artist(self, **kwargs):
        """Convert to mpl patch.
        """
        import matplotlib.patches as mpatches

        patch = mpatches.Circle(self.pos, self.radius.value, **kwargs)
        return patch


class SkyCircleRegion(SkyRegion):
    """
    Circular region, defined in sky coordinates.

    Parameters
    ----------
    pos : `~astropy.coordinates.SkyCoord`
        Sky coordinates of the circe center
    radius : `~astropy.units.Quantity`
        Circle radius
    """

    def __init__(self, pos, radius):
        self.pos = SkyCoord(pos)
        self.radius = Angle(radius)

    @property
    def area(self):
        """Circle area (solid angle [sr])

        For a circular region with radius :math:`2\\theta`
        the solid angle :math:`\\Theta` reads

        .. math:: \\Theta = 2\\pi (1-\\cos(\\theta))

        see: https://en.wikipedia.org/wiki/Solid_angle#Cone.2C_spherical_cap.2C_hemisphere
        """
        val = 2 * np.pi * (1 - np.cos(self.radius))
        return val * u.steradian

    def __contains__(self, coordinate):
        return self.pos.separation(coordinate) <= self.radius

    def intersects(self, other):
        """Checks if two sky circles overlap.

        Parameters
        ----------
        other : `~gammapy.region.SkyCircleRegion`
            Other region.
        """
        return self.pos.separation(other.pos) <= self.radius + other.radius

    def info(self):
        """Print some basic information"""
        ss = '\nSkyCircleRegion'
        ss += '\nCenter: {}'.format(self.pos)
        ss += '\nRadius: {}'.format(self.radius)

        return ss

    def to_pixel(self, wcs):
        """
        Return a `~gammapy.regions.PixCircleRegion`.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            WCS object
        """

        x, y = skycoord_to_pixel(self.pos, wcs, mode='wcs', origin=1)
        pix_radius = self.radius.deg / np.abs(wcs.wcs.cdelt[0])

        # TODO understand what is going on here
        # from photutils.utils.wcs_helpers import skycoord_to_pixel_scale_angle
        # central_pos = SkyCoord([wcs.wcs.crval], frame=self.pos.name, unit=wcs.wcs.cunit)
        # xc, yc, scale, angle = skycoord_to_pixel_scale_angle(central_pos, wcs)
        # val = (scale * self.radius).to(u.pixel).value
        # pix_radius = np.round(val[0],4)
        # pix_position = np.array([x, y]).transpose()

        return PixCircleRegion((x, y), pix_radius)

    def to_mpl_artist(self, ax, **kwargs):
        """Convert to mpl patch using a given wcs transformation

        Parameters
        ----------
        ax : `~astropy.wcsaxes.WCSAxes`
            WCS axis object
        kwargs : dict
            kwargs are forwarded to mpatches.Circle

        Returns
        -------
        patch : `~matplotlib.mpatches.Circle`
            Matplotlib patch object
        """

        import matplotlib.patches as mpatches

        val = self.pos.galactic
        center = (val.l.value, val.b.value)

        temp = dict(transform=ax.get_transform('galactic'),
                    radius=self.radius.value)
        kwargs.update(temp)
        patch = mpatches.Circle(center, **kwargs)

        return patch

    def plot(self, ax, **kwargs):
        """Plot region

        Parameters
        ----------
        ax : `~astropy.wcsaxes.WCSAxes`
            WCS axis object
        kwargs : dict
            kwargs are forwarded to mpatches.Circle
        """
        patch = self.to_mpl_artist(ax, **kwargs)
        ax.add_patch(patch)

    def to_ds9(self):
        """Convert to ds9 region string
        """
        l = self.pos.galactic.l.value
        b = self.pos.galactic.b.value
        r = self.radius.value
        sys = self.pos.name

        ss = '{sys}; circle({l},{b},{r})\n'.format(**locals())
        return ss

    def to_dict(self):
        rtdict = dict()
        rtdict['radius'] = '{}'.format(self.radius)
        rtdict['center_x'] = '{}'.format(self.pos.icrs.ra)
        rtdict['center_y'] = '{}'.format(self.pos.icrs.dec)
        rtdict['system'] = 'icrs'
        return rtdict

    @classmethod
    def from_dict(cls, dict):
        radius = Angle(dict['radius'])
        x = dict['center_x']
        y = dict['center_y']
        frame = dict['system']
        center = SkyCoord(x, y, frame=frame)
        return cls(center, radius)


