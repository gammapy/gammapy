"""Compute some statistics directly using these methods:
sherpa.stats.Stat.calc_stat(self, data, model, staterror=None, syserror=None, weight=None)
sherpa.stats.Stat.calc_staterror(self, data)


data = np.array([1, 2])
model = np.array([3,4])
staterror = np.array([4,5])
stat.calc_stat(data, model, staterror)

# returns (25.0, 5.0)
#o this is what is going on:
def calc_stat(data, model, staterror):
    fvec = (model - data) / staterror
    stat = (fvec ** 2).sum()
    return stat, fvec
# This is not what is described at
# http://cxc.cfa.harvard.edu/sherpa/statistics/#userstat
There they write
   fvec = ((data - model) / staterror)**2
   stat = fvec.sum()
 which results in a different fvec, but the same stat.

"""
import sherpa.stats as ss

stats = [('cash', ss.Cash),
         ('cstat', ss.CStat),
         ('chi2constvar', ss.Chi2ConstVar),
         ('chi2datavar', ss.Chi2DataVar),
         ('chi2gehrels', ss.Chi2Gehrels),
         ('chi2modvar', ss.Chi2ModVar),
         ('chi2xspecvar', ss.Chi2XspecVar)]

for stat_name, stat_class in stats:
    stat_object = stat_class()
    try:
        stat_error = stat_object.calc_staterror(0)
    except ValueError:
        stat_error = -999
    print('%20s %20.10f' % (stat_name, stat_error))
    # stat_object.calc_staterror(data=2, model=12, staterror=2)
