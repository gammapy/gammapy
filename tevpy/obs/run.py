"""Run and RunList class"""

__all__ = ['Run', 'RunList']

class Run(object):
    """
    Run parameters container
    """
    def __init__(self, GLON, GLAT, livetime=1800,
                 eff_area=1e12, background=0):
        self.GLON = GLON
        self.GLAT = GLAT
        self.livetime = livetime

    def wcs_header(self, system='FOV'):
        """
        Create a WCS FITS header for an image centered on the
        run position in one of these systems:
        FOV, Galactic, Equatorial
        """
        raise NotImplementedError


class RunList(list):
    """
    Run list container
    """
    pass


