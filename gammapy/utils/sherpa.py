from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np
from astropy import units as u
from sherpa.models import ArithmeticModel
from sherpa.data import BaseData, Data
from sherpa.stats import Likelihood
from sherpa.optmethods import LevMar, NelderMead, MonCar, GridSearch

__all__ = [
    'SherpaDataWrapper',
    'SherpaModelWrapper',
    'SherpaStatWrapper',
]

SHERPA_OPTMETHODS = OrderedDict()
SHERPA_OPTMETHODS['levmar'] = LevMar()
SHERPA_OPTMETHODS['simplex'] = NelderMead()
SHERPA_OPTMETHODS['moncar'] = MonCar()
SHERPA_OPTMETHODS['gridsearch'] = GridSearch()


class SherpaDataWrapper(Data):
    def __init__(self, gp_data, name='GPData'):
        # sherpa does some magic here: it sets class attributes from constructor
        # arguments so `gp_data` will be available later on the instance.
        self._data_dummy = np.empty_like(gp_data.e_ref)
        BaseData.__init__(self)

    def to_fit(self, staterr):
        return self._data_dummy, None, None

    def eval_model_to_fit(self, model):
        return self._data_dummy


class SherpaStatWrapper(Likelihood):
    def __init__(self, gp_stat):
        statname = 'GPStat'
        self.gp_stat = gp_stat
        Likelihood.__init__(self, statname)

    def calc_stat(self, data, model, *args, **kwargs):
        gp_models = []
        for part in model.parts:
            part._update_pars()
            gp_models.append(part.gp_model)
        gp_datasets = [_.gp_data for _ in data.datasets]
        # right now we only pass the first dataset and model
        # but in general we'd like pass all
        return self.gp_stat(gp_datasets[0], gp_models[0])

    def calc_stat_error(self):
        staterr = np.array([np.nan])
        return staterr


class SherpaModelWrapper(ArithmeticModel):
    """Wrapper to call Gammapy models from sherpa.
    """

    def __init__(self, gp_model):
        self.gp_model = gp_model
        self.parts = (self,)
        modelname = 'GP' + gp_model.__class__.__name__

        sherpa_pars = []
        for par in gp_model.parameters.parameters:
            sherpa_pars.append(par.to_sherpa(modelname=modelname))

        ArithmeticModel.__init__(self, modelname, sherpa_pars)

    def to_gp_model(self):
        kwargs = {}

        for par in self.pars:
            kwargs[par.name] = u.Quantity(par.val, par.units)

        return self.gp_model.__class__(**kwargs)

    def _update_pars(self):
        for par, gp_par in zip(self.pars, self.gp_model.parameters.parameters):
            gp_par.value = par.val

    def calc(self, pars, *args):
        return np.empty_like(args[0])
