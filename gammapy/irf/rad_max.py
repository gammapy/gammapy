from .core import IRF


class RadMax2D(IRF):
    """small class to exploit the IRF class to directly read in the FITS file the
    RAD_MAX hud, this has identical axes to the EFFECTIVE AREA HDU"""
    tag = "rad_max_2d"
    required_axes = ["energy", "offset"]
