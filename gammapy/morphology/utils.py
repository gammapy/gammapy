# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Morphology utility functions (mostly I/O)."""
from __future__ import print_function, division
import json

__all__ = ['read_ascii', 'read_json', 'write_all', 'write_ascii', 'write_json']


def _name(ii):
    """Use this to make the model name for source number `ii`."""
    return 'gauss2d.source_{0:02d}'.format(ii)


def _set(name, par, val):
    """Set a source parameter."""
    import sherpa.astro.ui as sau
    sau.set_par('{name}.{par}'.format(**locals()), val)
    # try:
    #    exec(name + '.' + par + '=' + str(val))
    # except Exception as e:
    #    print e


def _model(source_names):
    """Build additive model string for Gaussian sources."""
    return ' + '.join(['gauss2d.' + name for name in source_names])


def read_json(source, setter):
    """Read from JSON file."""
    if isinstance(source, dict):
        # Assume source is a dict with correct format
        d = source
    else:
        # Assume source is a filename with correct format
        d = json.load(open(source))
    source_names = d.keys()
    model = _model(source_names)
    setter(model)
    for name, pars in d.items():
        for par, val in pars.items():
            _set(name, par, val)


def read_ascii(filename, setter):
    """Read from ASCII file."""
    lines = open(filename).readlines()
    tokens = [line.split() for line in lines]
    names = set([token[0] for token in tokens])
    pars = set([token[1] for token in tokens])
    vals = set([token[2] for token in tokens])

    model = _model(names)
    setter(model)
    for name, par, val in zip(names, pars, vals):
        _set(name, par, val)


def write_json(pars, filename):
    """Write to JSON file."""
    d = {}

    for par in pars:
        if not par.modelname in d.keys():
            d[par.modelname] = {}

        d[par.modelname][par.name] = par.val

    json.dump(d, open(filename, 'w'), sort_keys=True, indent=4)


def write_ascii(pars, filename):
    """Write to ASCII"""
    fh = open(filename, 'w')
    for par in pars:
        fh.write('{0} {1} {2}\n'.format(par.modelname, par.name, par.val))


def write_all(filename='results.json'):
    """Dump source, fit results and conf results to a JSON file.

    http://www.astropython.org/snippet/2010/7/Save-sherpa-fit-and-conf-results-to-a-JSON-file
    """
    import sherpa.astro.ui as sau
    out = dict()

    if 0:
        src = sau.get_source()
        src_par_attrs = ('name', 'frozen', 'modelname', 'units', 'val', 'fullname')
        out['src'] = dict(name=src.name,
                          pars=[dict((attr, getattr(par, attr)) for attr in src_par_attrs)
                                for par in src.pars])

    try:
        fit_attrs = ('methodname', 'statname', 'succeeded', 'statval', 'numpoints', 'dof',
               'rstat', 'qval', 'nfev', 'message', 'parnames', 'parvals')
        fit = sau.get_fit_results()
        out['fit'] = dict((attr, getattr(fit, attr)) for attr in fit_attrs)
    except Exception as err:
        print(err)

    try:
        conf_attrs = ('datasets', 'methodname', 'fitname', 'statname', 'sigma', 'percent',
                      'parnames', 'parvals', 'parmins', 'parmaxes', 'nfits')
        conf = sau.get_conf_results()
        out['conf'] = dict((attr, getattr(conf, attr)) for attr in conf_attrs)
    except Exception as err:
        print(err)

    try:
        covar_attrs = ('datasets', 'methodname', 'fitname', 'statname', 'sigma', 'percent',
                      'parnames', 'parvals', 'parmins', 'parmaxes', 'nfits')
        covar = sau.get_covar_results()
        out['covar'] = dict((attr, getattr(covar, attr)) for attr in covar_attrs)
    except Exception as err:
        print(err)

    if 0:
        out['pars'] = []
        for par in src.pars:
            fullname = par.fullname
            if any(fullname == x['name'] for x in out['pars']):
                continue  # Parameter was already processed
            outpar = dict(name=fullname, kind=par.name)

            # None implies no calculated confidence interval for Measurement
            parmin = None
            parmax = None
            try:
                if fullname in conf.parnames:  # Confidence limits available from conf
                    i = conf.parnames.index(fullname)
                    parval = conf.parvals[i]
                    parmin = conf.parmins[i]
                    parmax = conf.parmaxes[i]
                if parmin == None:
                    parmin = -float('inf')  # None from conf means infinity, so set accordingly
                if parmax == None:
                    parmax = float('inf')
                elif fullname in fit.parnames:  # Conf failed or par is uninteresting and wasn't sent to conf
                    i = fit.parnames.index(fullname)
                    parval = fit.parvals[i]
                else:  # No fit or conf value (maybe frozen)
                    parval = par.val
            except Exception as err:
                print(err)

            out['pars'].append(outpar)
    if filename is None:
        return out
    else:
        json.dump(out, open(filename, 'w'), sort_keys=True, indent=4)
