# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from ..utils.modeling import Parameter, ParameterList

__all__ = [
    'PhaseCurve',
]


class PhaseCurve(object):
    """Temporal phase curve model.
    
    Phase for a given time is computed as

    .. math::

        \phi(t) = \phi0 + f0(t−t0) + (1/2)f1(t−t0)^2 + (1/6)f2(t−t0)^3

    Strictly periodic sources such as gamma-ray binaries have ``f1=0`` and ``f2=0``.
    Sources like some pulsars where the period spins up or down have ``f1!=0``
    and / or ``f2 !=0``.

    The "phase curve", i.e. multiplicative flux factor for a given phase is given
    by a `~astropy.table.Table` of nodes ``(PHASE, NORM)``, using linear interpolation
    and circular behaviour, where ``norm(phase=0) == norm(phase=1)``.

    Parameters
    ----------
    table : `~astropy.table.Table`
        A table of 'PHASE' vs 'NORM' should be given
    reference : float
        The MJD value where phase is considered as 0.
    phase0 : float
        Phase at the reference MJD
    f0, f2, f2 : float
        Derivatives of the function phi with time of order 1, 2, 3

    Examples
    --------
    Create an example phase curve object::

        from astropy.table import Table
        from gammapy.utils.scripts import make_path
        from gammapy.time.models import PhaseCurve
        filename = make_path('$GAMMAPY_EXTRA/test_datasets/phasecurve_LSI_DC.fits')
        table = Table.read(str(filename))
        phase_curve = PhaseCurve(table, reference=43366.275, phase0=0.0, f0=0.5, f1=0.0, f2=0.0)

    Use it to compute a phase and evaluate the phase curve model for a given time:

    >>> phase_curve.phase(time=46300.0)
    0.8624999999992724
    >>> phase_curve.evaluate(time=46300.0)
    0.05
    """

    def __init__(self, table, reference, phase0, f0, f1, f2):
        self.table = table
        self.parameters = ParameterList([
            Parameter('reference', reference),
            Parameter('phase0', phase0),
            Parameter('f0', f0),
            Parameter('f1', f1),
            Parameter('f2', f2)]
        )

    def phase(self, time):
        """Evaluate phase for a given time.
        
        Parameters
        ----------
        time : array_like
        
        Returns
        -------
        phase : array_like
        """
        pars = self.parameters
        reference = pars['reference'].value
        phase0 = pars['phase0'].value
        f0 = pars['f0'].value
        f1 = pars['f1'].value
        f2 = pars['f2'].value

        t = time - reference

        c1 = 0.5
        c2 = 1.0 / 6.0
        phase = phase0 + t * (f0 + t * (c1 * f1 + c2 * f2 * t))

        return np.remainder(phase, 1)

    def evaluate(self, time):
        """Evaluate model for a given time.

        Parameters
        ----------
        time : array_like

        Returns
        -------
        norm : array_like
        """
        x = self.phase(time)
        xp = self.table['PHASE']
        fp = self.table['NORM']
        return np.interp(x=x, xp=xp, fp=fp, period=1)
