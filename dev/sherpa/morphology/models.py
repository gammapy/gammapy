"""Define Sherpa 2D morphology models"""
 
import numpy as np
import sherpa.astro.ui as sau
from sherpa.models import ArithmeticModel, Parameter


class normdisk2d(ArithmeticModel):

    def __init__(self, name='normdisk2d'):
        self.xpos = Parameter(name, 'xpos', 0)
        self.ypos = Parameter(name, 'ypos', 0)
        self.ampl = Parameter(name, 'ampl', 1) # misnomer ... this is really the integral
        self.r0 = Parameter(name, 'r0', 1, 0)
        ArithmeticModel.__init__(self, name, (self.xpos, self.ypos, self.ampl, self.r0))
 
    def calc(self, p, x, y, *args, **kwargs):
        xpos, ypos, ampl, r0 = p
        r2 = (x - xpos) ** 2 + (y - ypos) ** 2 
        area = np.pi * r0 ** 2
        # Note that the ampl parameter is supposed to be the integral
        value = np.select([r2 <= r0 ** 2], [ampl / area])
        return value
 

class normshell2d(ArithmeticModel):

    def __init__(self, name='normshell2d'):
        self.xpos = Parameter(name, 'xpos', 0)
        self.ypos = Parameter(name, 'ypos', 0)
        self.ampl = Parameter(name, 'ampl', 1) # misnomer ... this is really the integral
        self.r0 = Parameter(name, 'r0', 1, 0)
        self.width = Parameter(name, 'width', 0.1, 0)
        ArithmeticModel.__init__(self, name, (self.xpos, self.ypos, self.ampl, self.r0, self.width))
 
    def calc(self, p, x, y, *args, **kwargs):
        """Homogeneously emitting spherical shell,
        projected along the z-direction
        (this is not 100% correct for very large shells on the sky)."""
        xpos, ypos, ampl, r_0, width = p
 
        r2 = (x - xpos) * (x - xpos) + (y - ypos) * (y - ypos)
        r_out = r_0 + width
        r_in2, r_out2 = r_0 * r_0, r_out * r_out
        # r_in3, r_out3 = r_in * r_in2, r_out * r_out2
        # We only call abs() in sqrt() to avoid warning messages.
        sphere_out = np.sqrt(np.abs(r_out2 - r2))
        sphere_in = np.sqrt(np.abs(r_in2 - r2))
        # Note: for r > r_out 'np.select' fills automatically zeros!
        non_normalized = np.select([r2 <= r_in2, r2 <= r_out2],
                                   [sphere_out - sphere_in, sphere_out])
        # Computed with Mathematica:
        integral = 2 * np.pi / 3 * (r_out ** 3 - r_0 ** 3)
        # integral = 1
        return ampl * non_normalized / integral

 
sau.add_model(normdisk2d)
sau.add_model(normshell2d)
