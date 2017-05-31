# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Gammacat open TeV source catalog.

https://github.com/gammapy/gamma-cat
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import logging
import numpy as np
from astropy.extern import six
from astropy.tests.helper import ignore_warnings
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import Angle
from astropy.modeling.models import Gaussian2D
from ..utils.modeling import SourceModel, SourceLibrary
from ..utils.scripts import make_path
from ..spectrum import FluxPoints
from ..spectrum.models import PowerLaw, PowerLaw2, ExponentialCutoffPowerLaw
from ..image.models import Shell2D, Delta2D
from .core import SourceCatalog, SourceCatalogObject

__all__ = [
    'SourceCatalogGammaCat',
    'SourceCatalogObjectGammaCat',
    'GammaCatDataCollection',
    'GammaCatResource',  # TODO: public or not?
    'GammaCatResourceIndex',  # TODO: public or not?
]

log = logging.getLogger(__name__)


class NoDataAvailableError(LookupError):
    """Generic error used in Gammapy, when some data isn't available.
    """
    pass


class GammaCatNotFoundError(OSError):
    """The gammapy-cat repo is not available.

    You have to set the GAMMA_CAT environment variable so that it's found.
    """
    pass


class SourceCatalogObjectGammaCat(SourceCatalogObject):
    """One object from the gamma-cat source catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalogGammaCat`.
    """
    _source_name_key = 'common_name'
    _source_index_key = 'catalog_row_index'

    def __str__(self):
        """Print default summary info string"""
        d = self.data

        ss = 'Source: {}\n'.format(d['common_name'])
        ss += 'source_id: {}\n'.format(d['source_id'])
        ss += 'reference_id: {}\n'.format(d['reference_id'])
        ss += '\n'

        ss += 'RA          : {:.2f}\n'.format(d['ra'])
        ss += 'DEC         : {:.2f}\n'.format(d['dec'])
        ss += 'GLON        : {:.2f}\n'.format(d['glon'])
        ss += 'GLAT        : {:.2f}\n'.format(d['glat'])
        ss += '\n'
        return ss

    def info(self):
        """Print summary info."""
        print(self)

    @property
    def spectral_model(self):
        """Source spectral model (`~gammapy.spectrum.models.SpectralModel`).

        TODO: how to handle systematic errors? (ignored at the moment)
        """
        data = self.data
        spec_type = data['spec_type']
        pars, errs = {}, {}

        if spec_type == 'pl':
            model_class = PowerLaw
            pars['amplitude'] = data['spec_pl_norm']
            errs['amplitude'] = data['spec_pl_norm_err']
            pars['index'] = data['spec_pl_index'] * u.Unit('')
            errs['index'] = data['spec_pl_index_err'] * u.Unit('')
            pars['reference'] = data['spec_pl_e_ref']
        elif spec_type == 'pl2':
            model_class = PowerLaw2
            pars['amplitude'] = data['spec_pl2_flux']
            errs['amplitude'] = data['spec_pl2_flux_err']
            pars['index'] = data['spec_pl2_index'] * u.Unit('')
            errs['index'] = data['spec_pl2_index_err'] * u.Unit('')
            pars['emin'] = data['spec_pl2_e_min']
            e_max = data['spec_pl2_e_max']
            DEFAULT_E_MAX = u.Quantity(1e5, 'TeV')
            if np.isnan(e_max.value):
                e_max = DEFAULT_E_MAX
            pars['emax'] = e_max
        elif spec_type == 'ecpl':
            model_class = ExponentialCutoffPowerLaw
            pars['amplitude'] = data['spec_ecpl_norm']
            errs['amplitude'] = data['spec_ecpl_norm_err']
            pars['index'] = data['spec_ecpl_index'] * u.Unit('')
            errs['index'] = data['spec_ecpl_index_err'] * u.Unit('')
            pars['lambda_'] = 1. / data['spec_ecpl_e_cut']
            errs['lambda_'] = data['spec_ecpl_e_cut_err'] / data['spec_ecpl_e_cut'] ** 2
            pars['reference'] = data['spec_ecpl_e_ref']
        else:
            raise ValueError('Invalid spec_type: {}'.format(spec_type))

        model = model_class(**pars)
        model.parameters.set_parameter_errors(errs)

        return model

    def spatial_model(self, emin=1 * u.TeV, emax=10 * u.TeV):
        """Source spatial model."""
        d = self.data
        flux = self.spectral_model.integral(emin, emax)
        morph_type = d['morph_type']
        pars = {}

        glon = Angle(d['glon']).wrap_at('180d')
        glat = Angle(d['glat']).wrap_at('180d')

        if morph_type == 'gauss':
            pars['x_mean'] = glon.value
            pars['y_mean'] = glat.value
            pars['x_stddev'] = d['morph_sigma'].value
            pars['y_stddev'] = d['morph_sigma'].value
            if not np.isnan(d['morph_sigma2']):
                pars['y_stddev'] = d['morph_sigma2'].value
            if not np.isnan(d['morph_pa']):
                # TODO: handle reference frame for rotation angle
                pars['theta'] = Angle(d['morph_pa'], 'deg').rad
            ampl = flux.to('cm-2 s-1').value
            pars['amplitude'] = ampl * 1 / (2 * np.pi * pars['x_stddev'] * pars['y_stddev'])
            return Gaussian2D(**pars)
        elif morph_type == 'shell':
            pars['amplitude'] = flux.to('cm-2 s-1').value
            pars['x_0'] = glon.value
            pars['y_0'] = glat.value
            pars['r_in'] = d['morph_sigma'].value * 0.8
            pars['width'] = 0.2 * d['morph_sigma'].value
            return Shell2D(**pars)
        elif morph_type == 'point':
            pars['amplitude'] = flux.to('cm-2 s-1').value
            pars['x_0'] = glon.value
            pars['y_0'] = glat.value
            return Delta2D(**pars)
        elif morph_type == 'none':
            raise NoDataAvailableError('No spatial model available: {}'.format(self.name))
        else:
            raise NotImplementedError('Unknown spatial model: {!r}'.format(morph_type))

    def _add_source_meta(self, table):
        """Copy over some info to table.meta"""
        d = self.data
        m = table.meta
        m['origin'] = 'Data from gamma-cat'
        m['source_id'] = d['source_id']
        m['common_name'] = d['common_name']
        m['reference_id'] = d['reference_id']

    @property
    def flux_points(self):
        """Differential flux points (`~gammapy.spectrum.FluxPoints`)."""
        d = self.data
        table = Table()
        table.meta['SED_TYPE'] = 'dnde'
        self._add_source_meta(table)

        valid = np.isfinite(d['sed_e_ref'].value)

        if valid.sum() == 0:
            raise NoDataAvailableError('No flux points available: {}'.format(self.name))

        table['e_ref'] = d['sed_e_ref']
        table['e_min'] = d['sed_e_min']
        table['e_max'] = d['sed_e_max']

        table['dnde'] = d['sed_dnde']
        table['dnde_err'] = d['sed_dnde_err']
        table['dnde_errn'] = d['sed_dnde_errn']
        table['dnde_errp'] = d['sed_dnde_errp']
        table['dnde_ul'] = d['sed_dnde_ul']

        # Only keep rows that actually contain information
        table = table[valid]

        # Only keep columns that actually contain information
        def _del_nan_col(table, colname):
            if np.isfinite(table[colname]).sum() == 0:
                del table[colname]

        for colname in table.colnames:
            _del_nan_col(table, colname)

        return FluxPoints(table)

    @property
    def is_pointlike(self):
        """
        Source is pointlike.
        """
        return self.data['morph_type'] == 'point'


class SourceCatalogGammaCat(SourceCatalog):
    """Gammacat open TeV source catalog.

    See: https://github.com/gammapy/gamma-cat

    One source is represented by `~gammapy.catalog.SourceCatalogObjectGammaCat`.

    Parameters
    ----------
    filename : str
        Path to the gamma-cat fits file.

    Examples
    --------
    Load the catalog data:

    >>> from gammapy.catalog import SourceCatalogGammaCat
    >>> cat = SourceCatalogGammaCat()

    Access a source by name:

    >>> source = cat['Vela Junior']

    Access source spectral data and plot it:

    >>> source.spectral_model.plot()
    >>> source.spectral_model.plot_error()
    >>> source.flux_points.plot()
    """
    name = 'gamma-cat'
    description = 'An open catalog of gamma-ray sources'
    source_object_class = SourceCatalogObjectGammaCat

    def __init__(self, filename='$GAMMA_CAT/docs/data/gammacat.fits.gz'):
        filename = str(make_path(filename))

        with ignore_warnings():  # ignore FITS units warnings
            table = Table.read(filename, hdu=1)
        self.filename = filename

        source_name_key = 'common_name'
        source_name_alias = ('other_names', 'gamma_names')
        super(SourceCatalogGammaCat, self).__init__(
            table=table,
            source_name_key=source_name_key,
            source_name_alias=source_name_alias,
        )

    def to_source_library(self):
        """Convert to a `~gammapy.utils.modeling.SourceLibrary`.

        TODO: add an option whether to skip or raise on missing models or data.
        """
        source_list = []

        for source_idx in range(len(self.table)):
            source = self[source_idx]
            try:
                source_model = SourceModel.from_gammacat(source)
            except NoDataAvailableError:
                log.warning('Skipping source {} (missing data in gamma-cat)'.format(source.name))
                continue
            source_list.append(source_model)

        return SourceLibrary(source_list=source_list)


class GammaCatDataCollection(object):
    """Data store for gamma-cat.

    Gives access to all data from https://github.com/gammapy/gamma-cat .

    Holds a `GammaCatResourceIndex` to locate resources,
    but also more info about gamma-cat, as well as methods to create
    Gammapy objects (spectral models, flux points, lightcurves) from the datasets.
    """

    def __init__(self, data_index):
        self.data_index = data_index

    @classmethod
    def from_index_file(cls, filename='$GAMMA_CAT/docs/data/gammacat-datasets.json'):
        """Create from index file."""
        filename = str(make_path(filename))
        # TODO: make a list of `GammaCatResource`, as well as a dict by ``resource_id`` for lookup!
        data_index = load_json(filename)
        return cls(data_index=data_index)

    def info(self):
        """Print some info."""
        ss = 'version = {}'.format(self.data_index['info']['version'])
        return ss


class GammaCatResource(object):
    """Reference for a single resource in gamma-cat.

    This can be considered an implementation detail,
    used to assign ``global_id`` and to load resources.

    TODO: explain how ``global_id``, ``type`` and ``location`` work.
    Uses the Python ``hash`` function on the tuple ``(source_id, reference_id, file_id)``

    Parameters
    ----------
    source_id : int
        Gamma-cat source ID
    reference_id : str
        Gamma-cat reference ID (usually the ADS paper bibcode)
    file_id : int
        File ID (a counter for cases with multiple measurements per reference / source)
        (use integer -1 if missing)
    type : str
        Resource type (use string 'none' if missing)
    location : str
        Resource location (use string 'none' if missing)

    Examples
    --------
    >>> from gammapy.catalog.gammacat import GammaCatResource
    >>> resource = GammaCatResource(source_id=42, reference_id='2010A&A...516A..62A', file_id=2)
    >>> resource
    GammaCatResource(source_id=42, reference_id='2010A&A...516A..62A', file_id=2, type='none', location='none')
    """
    _NA_FILL = dict(file_id=-1, type='none', location='none')

    def __init__(self, source_id, reference_id, file_id=-1, type='none', location='none'):
        self.source_id = int(source_id)
        self.reference_id = six.text_type(reference_id)
        self.file_id = int(file_id)
        self.type = six.text_type(type)
        self.location = six.text_type(location)

    @property
    def global_id(self):
        """Globally unique (within gamma-cat) resource ID (str).

        (see class docstring for explanation and example).
        """
        return '|'.join((str(self.source_id), self.reference_id, str(self.file_id), self.type))

    def __repr__(self):
        fmt = '{}(source_id={!r}, reference_id={!r}, file_id={!r}, type={!r}, location={!r})'
        return fmt.format(self.__class__.__name__, self.source_id, str(self.reference_id),
                          self.file_id, str(self.type), str(self.location))

    def __eq__(self, other):
        return (
            self.source_id == other.source_id and
            self.reference_id == other.reference_id and
            self.file_id == other.file_id and
            self.type == other.type and
            self.location == other.location
        )

    def __lt__(self, other):
        return (
            self.source_id < other.source_id or
            self.reference_id < other.reference_id or
            self.file_id < other.file_id or
            self.type < other.type or
            self.location < other.location
        )

    def to_dict(self):
        """Convert to `collections.OrderedDict`."""
        data = OrderedDict()
        data['source_id'] = self.source_id
        data['reference_id'] = self.reference_id
        data['file_id'] = self.file_id
        data['type'] = self.type
        data['location'] = self.location
        return data

    @classmethod
    def from_dict(cls, data):
        """Create from dict."""
        return cls(
            source_id=data['source_id'],
            reference_id=data['reference_id'],
            file_id=data.get('file_id', cls._NA_FILL['file_id']),
            type=data.get('type', cls._NA_FILL['type']),
            location=data.get('location', cls._NA_FILL['location'])
        )


class GammaCatResourceIndex(object):
    """Resource index for gamma-cat.

    Parameters
    ----------
    resources : list
        List of `GammaCatResource` objects
    """

    def __init__(self, resources):
        self.resources = resources

    def __repr__(self):
        return '{}(n_resources={})'.format(self.__class__.__name__, len(self.resources))

    def __eq__(self, other):
        if len(self.resources) != len(other.resources):
            return False
        return all(a == b for (a, b) in zip(self.resources, other.resources))

    @property
    def unique_source_ids(self):
        """Sorted list of unique source IDs (`list(int)`)."""
        return sorted(set([resource.source_id for resource in self.resources]))

    @property
    def unique_reference_ids(self):
        """Sorted list of unique source IDs (`list(str)`)."""
        return sorted(set([resource.reference_id for resource in self.resources]))

    @property
    def global_ids(self):
        """List of global resource IDs (`list(str)`).

        In original order, not sorted.
        """
        return [resource.global_id for resource in self.resources]

    def sort(self):
        """Return a sorted copy (leave self unchanged)."""
        return self.__class__(sorted(self.resources))

    def to_list(self):
        """Convert to list of dict."""
        return [resource.to_dict() for resource in self.resources]

    @classmethod
    def from_list(cls, data):
        """Create from list of dicts."""
        return cls([GammaCatResource.from_dict(_) for _ in data])

    def to_table(self):
        """Convert to `~astropy.table.Table`."""
        rows = self.to_list()
        return Table(rows=rows, names=list(rows[0].keys()))

    @classmethod
    def from_table(cls, table):
        """Create from `~astropy.table.Table`."""
        resources = []
        for row in table:
            data = OrderedDict((k, row[k]) for k in table.colnames)
            resources.append(GammaCatResource.from_dict(data))
        return cls(resources=resources)

    def to_pandas(self):
        """Convert to `pandas.DataFrame`."""
        # This is inefficient. Could implement direct conversion if needed.
        table = self.to_table()
        return table.to_pandas()

    @classmethod
    def from_pandas(cls, dataframe):
        """Create from `pandas.DataFrame`."""
        table = Table.from_pandas(dataframe)
        return cls.from_table(table)

    def query(self, *args, **kwargs):
        """Query to select subset of resources.

        Calls `pandas.DataFrame.query` and passes arguments to that method.

        Examples
        --------
        >>> resource_index = GammaCatResourceIndex(...)
        >>> resource_index2 = resource_index.query('type == "sed" and source_id == 42')
        """
        df = self.to_pandas()
        df2 = df.query(*args, **kwargs)
        return self.from_pandas(df2)
