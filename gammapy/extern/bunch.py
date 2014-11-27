# Licensed under a 3-clause BSD style license - see LICENSE.rst
# Implementation is taken from scipy.optimize.OptimizeResult and renamed to 'Bunch'

__all__ = ['Bunch']


class Bunch(dict):
    """
    Dictionary with attribute access for result objects.

    Example is `~gammapy.detect.TSMapResult`
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"
