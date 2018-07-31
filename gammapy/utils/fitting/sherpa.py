from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
from sherpa.optmethods import LevMar, NelderMead, MonCar, GridSearch

SHERPA_OPTMETHODS = OrderedDict()
SHERPA_OPTMETHODS['levmar'] = LevMar()
SHERPA_OPTMETHODS['simplex'] = NelderMead()
SHERPA_OPTMETHODS['moncar'] = MonCar()
SHERPA_OPTMETHODS['gridsearch'] = GridSearch()
