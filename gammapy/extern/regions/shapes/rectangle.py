from astropy import units as u

from ..core import PixelRegion, SkyRegion


class RectanglePixelRegion(PixelRegion):
    """
    An rectangle in pixel coordinates.

    Parameters
    ----------
    center : :class:`~regions.core.pixcoord.PixCoord`
        The position of the center of the rectangle.
    height : float
        The height of the rectangle
    width : float
        The width of the rectangle
    angle : :class:`~astropy.units.Quantity`
        The rotation of the rectangle. If set to zero (the default), the width
        is lined up with the x axis.
    """

    def __init__(self, center, height, width, angle=0 * u.deg, meta=None, visual=None):
        # TODO: use quantity_input to check that angle is an angle
        self.center = center
        self.height = height
        self.width = width
        self.angle = angle
        self.meta = meta or {}
        self.visual = visual or {}

    @property
    def area(self):
        return self.width * self.height

    def contains(self, pixcoord):
        # TODO: needs to be implemented
        raise NotImplementedError("")

    def to_shapely(self):
        # TODO: needs to be implemented
        raise NotImplementedError("")

    def to_sky(self, wcs, mode='local', tolerance=None):
        # TODO: needs to be implemented
        raise NotImplementedError("")

    def to_mask(self, mode='center'):
        # TODO: needs to be implemented
        raise NotImplementedError("")



class RectangleSkyRegion(SkyRegion):
    """
    An rectangle in sky coordinates.

    Parameters
    ----------
    center : :class:`~regions.core.pixcoord.PixCoord`
        The position of the center of the rectangle.
    height : :class:`~astropy.units.Quantity`
        The height radius of the rectangle
    width : :class:`~astropy.units.Quantity`
        The width radius of the rectangle
    angle : :class:`~astropy.units.Quantity`
        The rotation of the rectangle. If set to zero (the default), the width
        is lined up with the longitude axis of the celestial coordinates.
    """

    def __init__(self, center, height, width, angle=0 * u.deg, meta=None, visual=None):
        # TODO: use quantity_input to check that height, width, and angle are angles
        self.center = center
        self.height = height
        self.width = width
        self.angle = angle
        self.meta = meta or {}
        self.visual = visual or {}

    @property
    def area(self):
        return self.width * self.height

    def contains(self, skycoord):
        # TODO: needs to be implemented
        raise NotImplementedError("")

    def to_pixel(self, wcs, mode='local', tolerance=None):
        # TODO: needs to be implemented
        raise NotImplementedError("")
