from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from ..utils.modeling import Parameter, ParameterList

__all__ = [
    'PhaseCurve',
]

log = logging.getLogger(__name__)


class PhaseCurve(object):

 """
 This is to calculate phase of a periodic system and to provide the normalization
 factor for the phase using circular interpolation. The required data for interpolation
 is given as an astropy table

# TODO: 1. Checking table values if the phases are in increasing order
        2. Check normalization constant to set maximum values

 Parameters
 ----------
 table = A table of 'PHASE' vs 'NORM' should be given
 reference: The MJD value where phase is considered as 0.
 phase0: phase at the reference MJD
 f0:1st order derivative of the function phi with time
 f1:2nd order derivative of the function phi with time
 f2:3rd order derivative of the function phi with time
 phase as function of time is computed using
 Φ(t)=Φ0+f0(t−t0)+(1/2)f1(t−t0)^2+(1/6)f2(t−t0)^3


 Returns
 --------
 Phase corresponding to the time provided as input
 It also calculates the normalization constant for that phase

 Examples
 --------
 This shows how to get the phase and the normalization constant

    phase_curve = PhaseCurve(table,reference,phase0,f0,f1,f2)
    phase = phase_curve.phase(time)
    norm_const = phase_curve.evaluate(time)

"""


      
 def __init__(self,table,reference,phase0,f0,f1,f2):

    self.table = table
    self.parameters = ParameterList([
    Parameter('reference', reference),
    Parameter('phase0', phase0),
    Parameter('f0', f0),
    Parameter('f1', f1),
    Parameter('f2', f2)])
     

 def phase(self,time):
    """ To assign the phase to the time """
    c1 = 0.5
    c2 = 1.0/6.0
    pars = self.parameters
    reference = pars['reference'].value
    phase0    = pars['phase0'].value
    f0        = pars['f0'].value
    f1        = pars['f1'].value
    f2        = pars['f2'].value

    t = time - reference

    phase = phase0 + t * (f0 + t * (c1 * f1 +  c2 * f2 * t))
          
    phase = np.remainder(phase,1)
    return phase
     
 def evaluate(self,time):
    """
    To calculate the normalization constant for
    the phase calculated from the given time
    """
    phase = self.phase(time)
    tbdata = self.table
    phase_col = tbdata['PHASE']
    norm_col  = tbdata['NORM']

    x_values = phase_col
    f_values = norm_col

    f_norm = np.interp(x = phase,xp= x_values,fp = f_values, period = 1)
    return f_norm

