# Licensed under a 3-clause BSD style license - see LICENSE.rst

__all__ = ["Likelihood"]


# TODO: get rid of this wrapper class? Or use it in a better way?
class Likelihood:
    """Wrapper of the likelihood function used by the optimiser.

    This might become superfluous if we introduce a
    generic ``Likelihood`` interface and use that directly,
    or change the ``Fit`` class to work with ``Model``
    and ``Likelihood`` objects.

    For now, this class does the translation of parameter
    values and the parameter factors the optimiser sees.

    Parameters
    ----------
    parameters : `~gammapy.modeling.Parameters`
        Parameters with starting values
    function : callable
        Likelihood function
    """

    def __init__(self, function, parameters, store_trace):
        self.function = function
        self.parameters = parameters
        self.trace = []
        self.store_trace = store_trace

    def store_trace_iteration(self, total_stat):
        row = {"total_stat": total_stat}
        pars = self.parameters.free_parameters
        names = [f"par-{idx}" for idx in range(len(pars))]
        vals = dict(zip(names, pars.value))
        row.update(vals)
        self.trace.append(row)

    def fcn(self, factors):
        self.parameters.set_parameter_factors(factors)
        total_stat = self.function()

        if self.store_trace:
            self.store_trace_iteration(total_stat)

        return total_stat
