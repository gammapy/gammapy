import numpy as np
from sherpa.ui import *

# TODO: It would be better to use a table model
# and not integrate over bins to simplify this!

# Set a few parameters
numbins = 2
data_value = np.arange(numbins)  # 10
model_value = 5  # np.arange(numbins) #11
staterror_value = 10

# Note: We have a DataSpace1DInt data space here,
# i.e. the model will be integrated over the bins
# and the bin width matters.
binwidth = 1
stop = binwidth * numbins

# Set up data and model
dataspace1d(start=0, stop=stop, numbins=numbins)
get_data().y = data_value * np.ones(numbins)
get_data().staterror = staterror_value * np.ones(numbins)
set_model(polynom1d('model'))
model.c0 = model_value
model.c1 = 1

# Compute statistic
stat_names = ('cash cstat chi2constvar chi2datavar chi2gehrels '
              'chi2modvar chi2xspecvar'.split())
for stat_name in stat_names:
    set_stat(stat_name)
    stat_value = calc_stat()
    print('%20s %20.10f' % (stat_name, stat_value))
