# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Gammacat open TeV source catalog.

https://github.com/gammapy/gamma-cat
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict, namedtuple
import functools
import logging
import json
import numpy as np
from ..extern import six
from astropy import units as u
from astropy.table import Table
from ..utils.scripts import make_path
from ..spectrum import FluxPoints
from ..spectrum.models import PowerLaw, PowerLaw2, ExponentialCutoffPowerLaw
from ..image.models import SkyPointSource, SkyGaussian, SkyShell
from ..cube.models import SkyModel, SkyModels
from .core import SourceCatalog, SourceCatalogObject

__all__ = [
    "SourceCatalogGammaCat",
    "SourceCatalogObjectGammaCat",
    "GammaCatDataCollection",
    "GammaCatResource",
    "GammaCatResourceIndex",
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

    _source_name_key = "common_name"
    _source_index_key = "catalog_row_index"

    def __str__(self):
        return self.info()

    def info(self, info="all"):
        """Info string.

        Parameters
        ----------
        info : {'all', 'basic', 'position, 'model'}
            Comma separated list of options
        """

        if info == "all":
            info = "basic,position,model"

        ss = ""
        ops = info.split(",")
        if "basic" in ops:
            ss += self._info_basic()
        if "position" in ops:
            ss += self._info_position()
        if "model" in ops:
            ss += self._info_morph()
            ss += self._info_spectral_fit()
            ss += self._info_spectral_points()

        return ss

    def _info_basic(self):
        """Print basic info."""
        d = self.data
        ss = "\n*** Basic info ***\n\n"
        ss += "Catalog row index (zero-based) : {}\n".format(d["catalog_row_index"])
        ss += "{:<15s} : {}\n".format("Common name", d["common_name"])

        # ss += '{:<15s} : {}\n'.format('Gamma names', d['gamma_names'])
        # ss += '{:<15s} : {}\n'.format('Fermi names', d['fermi_names'])
        # ss += '{:<15s} : {}\n'.format('Other names', d['other_names'])

        def get_nonentry_keys(keys):
            vals = [d[_].strip() for _ in keys]
            return ",".join([_ for _ in vals if _ != ""])

        keys = ["gamma_names", "fermi_names", "other_names"]
        other_names = get_nonentry_keys(keys)
        ss += "{:<15s} : {}\n".format("Other names", other_names)
        ss += "{:<15s} : {}\n".format("Location", d["where"])
        ss += "{:<15s} : {}\n".format("Class", d["classes"])

        ss += "\n{:<15s} : {}\n".format("TeVCat ID", d["tevcat_id"])
        ss += "{:<15s} : {}\n".format("TeVCat 2 ID", d["tevcat2_id"])
        ss += "{:<15s} : {}\n".format("TeVCat name", d["tevcat_name"])

        ss += "\n{:<15s} : {}\n".format("TGeVCat ID", d["tgevcat_id"])
        ss += "{:<15s} : {}\n".format("TGeVCat name", d["tgevcat_name"])

        ss += "\n{:<15s} : {}\n".format("Discoverer", d["discoverer"])
        ss += "{:<15s} : {}\n".format("Discovery date", d["discovery_date"])
        ss += "{:<15s} : {}\n".format("Seen by", d["seen_by"])
        ss += "{:<15s} : {}\n".format("Reference", d["reference_id"])

        return ss

    def _info_position(self):
        """Print position info."""
        d = self.data
        ss = "\n*** Position info ***\n\n"

        ss += "SIMBAD:\n"
        ss += "{:<20s} : {:.3f}\n".format("RA", d["ra"])
        ss += "{:<20s} : {:.3f}\n".format("DEC", d["dec"])
        ss += "{:<20s} : {:.3f}\n".format("GLON", d["glon"])
        ss += "{:<20s} : {:.3f}\n".format("GLAT", d["glat"])

        ss += "\nMeasurement:\n"
        ss += "{:<20s} : {:.3f}\n".format("RA", d["pos_ra"])
        ss += "{:<20s} : {:.3f}\n".format("DEC", d["pos_dec"])
        ss += "{:<20s} : {:.3f}\n".format("GLON", d["pos_glon"])
        ss += "{:<20s} : {:.3f}\n".format("GLAT", d["pos_glat"])
        ss += "{:<20s} : {:.3f}\n".format("Position error", d["pos_err"])

        return ss

    def _info_morph(self):
        """Print morphology info."""
        ss = "\n*** Morphology info ***\n\n"
        d = self.data
        ss += "{:<25s} : {}\n".format("Morphology model type", d["morph_type"])

        # TODO: change to morphology model dependent printout
        # (see spectra printout and `spatial_model` property)
        ss += "{:<25s} : {:.3f}\n".format("Sigma", d["morph_sigma"])
        ss += "{:<25s} : {:.3f}\n".format("Sigma error", d["morph_sigma_err"])
        ss += "{:<25s} : {:.3f}\n".format("Sigma2", d["morph_sigma2"])
        ss += "{:<25s} : {:.3f}\n".format("Sigma2 error", d["morph_sigma2_err"])

        ss += "{:<25s} : {:.3f}\n".format("Position angle", d["morph_pa"])
        ss += "{:<25s} : {:.3f}\n".format("Position angle error", d["morph_pa_err"])
        ss += "{:<25s} : {}\n".format("Position angle frame", d["morph_pa_frame"])

        return ss

    def _info_spectral_fit(self):
        """Print spectral info."""
        d = self.data
        ss = "\n*** Spectral info ***\n\n"
        ss += "{:<15s} : {:.3f}\n".format("Significance", d["significance"])
        ss += "{:<15s} : {:.3f}\n".format("Livetime", d["livetime"])

        spec_type = d["spec_type"]
        ss += "\n{:<15s} : {}\n".format("Spectrum type", spec_type)

        # Spectral model parameters
        if spec_type == "pl":
            ss += "{:<15s} : {:.3} +- {:.3} (stat) +- {:.3} (sys) {}\n".format(
                "norm",
                d["spec_pl_norm"].value,
                d["spec_pl_norm_err"].value,
                d["spec_pl_norm_err_sys"].value,
                "cm-2 s-1 TeV-1",
            )
            ss += "{:<15s} : {:.3} +- {:.3} (stat) +- {:.3} (sys)\n".format(
                "index",
                d["spec_pl_index"],
                d["spec_pl_index_err"],
                d["spec_pl_index_err_sys"],
            )
            ss += "{:<15s} : {:.3}\n".format("reference", d["spec_pl_e_ref"])

        elif spec_type == "pl2":
            ss += "{:<15s} : {:.3} +- {:.3} (stat) +- {:.3} (sys) {}\n".format(
                "flux",
                d["spec_pl2_flux"].value,
                d["spec_pl2_flux_err"].value,
                d["spec_pl2_flux_err_sys"].value,
                "cm-2 s-1",
            )
            ss += "{:<15s} : {:.3} +- {:.3} (stat) +- {:.3} (sys)\n".format(
                "index",
                d["spec_pl2_index"],
                d["spec_pl2_index_err"],
                d["spec_pl2_index_err_sys"],
            )
            ss += "{:<15s} : {:.3}\n".format("e_min", d["spec_pl2_e_min"])
            ss += "{:<15s} : {:.3}\n".format("e_max", d["spec_pl2_e_max"])

        elif spec_type == "ecpl":
            ss += "{:<15s} : {:.3g} +- {:.3g} (stat) +- {:.03g} (sys) {}\n".format(
                "norm",
                d["spec_ecpl_norm"].value,
                d["spec_ecpl_norm_err"].value,
                d["spec_ecpl_norm_err_sys"].value,
                "cm-2 s-1 TeV-1",
            )
            ss += "{:<15s} : {:.3} +- {:.3} (stat) +- {:.3} (sys)\n".format(
                "index",
                d["spec_ecpl_index"],
                d["spec_ecpl_index_err"],
                d["spec_ecpl_index_err_sys"],
            )
            ss += "{:<15s} : {:.3} +- {:.3} (stat) +- {:.3} (stat) {}\n".format(
                "e_cut",
                d["spec_ecpl_e_cut"].value,
                d["spec_ecpl_e_cut_err"].value,
                d["spec_ecpl_e_cut_err_sys"].value,
                "TeV",
            )
            ss += "{:<15s} : {:.3}\n".format("reference", d["spec_ecpl_e_ref"])

        else:
            # raise ValueError('Spectral model printout not implemented: {}'.format(spec))
            ss += "\nSpectral model printout not yet implemented.\n"

        ss += "\n{:<20s} : ({:.3}, {:.3}) TeV\n".format(
            "Energy range", d["spec_erange_min"].value, d["spec_erange_max"].value
        )
        ss += "{:<20s} : {:.3}\n".format("theta", d["spec_theta"])

        ss += "\n\nDerived fluxes:\n"

        ss += "{:<30s} : {:.3} +- {:.3} (stat) {}\n".format(
            "Spectral model norm (1 TeV)",
            d["spec_dnde_1TeV"].value,
            d["spec_dnde_1TeV_err"].value,
            "cm-2 s-1 TeV-1",
        )
        ss += "{:<30s} : {:.3} +- {:.3} (stat) {}\n".format(
            "Integrated flux (>1 TeV)",
            d["spec_flux_1TeV"].value,
            d["spec_flux_1TeV_err"].value,
            "cm-2 s-1",
        )
        ss += "{:<30s} : {:.3f} +- {:.3f} {}\n".format(
            "Integrated flux (>1 TeV)",
            d["spec_flux_1TeV_crab"],
            d["spec_flux_1TeV_crab_err"],
            "(% Crab)",
        )
        ss += "{:<30s} : {:.3} +- {:.3} (stat) {}\n".format(
            "Integrated flux (1-10 TeV)",
            d["spec_eflux_1TeV_10TeV"].value,
            d["spec_eflux_1TeV_10TeV_err"].value,
            "erg cm-2 s-1",
        )

        return ss

    def _info_spectral_points(self):
        """Print spectral points info."""
        d = self.data
        ss = "\n*** Spectral points ***\n\n"
        ss += "{:<25s} : {}\n".format("SED reference id", d["sed_reference_id"])
        ss += "{:<25s} : {}\n".format("Number of spectral points", d["sed_n_points"])
        ss += "{:<25s} : {}\n\n".format("Number of upper limits", d["sed_n_ul"])

        try:
            lines = self._flux_points_table_formatted.pformat(
                max_width=-1, max_lines=-1
            )
            ss += "\n".join(lines)
        except NoDataAvailableError:
            ss += "\nNo spectral points available for this source."

        return ss + "\n"

    @property
    def spectral_model(self):
        """Source spectral model (`~gammapy.spectrum.models.SpectralModel`).

        Parameter errors are statistical errors only.
        """
        data = self.data
        spec_type = data["spec_type"]
        pars, errs = {}, {}

        if spec_type == "pl":
            model_class = PowerLaw
            pars["amplitude"] = data["spec_pl_norm"]
            errs["amplitude"] = data["spec_pl_norm_err"]
            pars["index"] = data["spec_pl_index"] * u.Unit("")
            errs["index"] = data["spec_pl_index_err"] * u.Unit("")
            pars["reference"] = data["spec_pl_e_ref"]
        elif spec_type == "pl2":
            model_class = PowerLaw2
            pars["amplitude"] = data["spec_pl2_flux"]
            errs["amplitude"] = data["spec_pl2_flux_err"]
            pars["index"] = data["spec_pl2_index"] * u.Unit("")
            errs["index"] = data["spec_pl2_index_err"] * u.Unit("")
            pars["emin"] = data["spec_pl2_e_min"]
            e_max = data["spec_pl2_e_max"]
            DEFAULT_E_MAX = u.Quantity(1e5, "TeV")
            if np.isnan(e_max.value):
                e_max = DEFAULT_E_MAX
            pars["emax"] = e_max
        elif spec_type == "ecpl":
            model_class = ExponentialCutoffPowerLaw
            pars["amplitude"] = data["spec_ecpl_norm"]
            errs["amplitude"] = data["spec_ecpl_norm_err"]
            pars["index"] = data["spec_ecpl_index"] * u.Unit("")
            errs["index"] = data["spec_ecpl_index_err"] * u.Unit("")
            pars["lambda_"] = 1. / data["spec_ecpl_e_cut"]
            errs["lambda_"] = data["spec_ecpl_e_cut_err"] / data["spec_ecpl_e_cut"] ** 2
            pars["reference"] = data["spec_ecpl_e_ref"]
        else:
            raise ValueError("Invalid spec_type: {}".format(spec_type))

        model = model_class(**pars)
        model.parameters.set_parameter_errors(errs)

        return model

    @property
    def spatial_model(self):
        """Source spatial model (`~gammapy.image.models.SkySpatialModel`).

        TODO: add parameter errors!
        """
        d = self.data
        morph_type = d["morph_type"]

        glon = d["glon"]
        glat = d["glat"]

        if morph_type == "point":
            return SkyPointSource(lon_0=glon, lat_0=glat)
        elif morph_type == "gauss":
            # TODO: add infos back once model support elongation
            # pars['x_stddev'] = d['morph_sigma']
            # pars['y_stddev'] = d['morph_sigma']
            # if not np.isnan(d['morph_sigma2']):
            #     pars['y_stddev'] = d['morph_sigma2']
            # if not np.isnan(d['morph_pa']):
            #     # TODO: handle reference frame for rotation angle
            #     pars['theta'] = Angle(d['morph_pa'], 'deg').rad
            return SkyGaussian(lon_0=glon, lat_0=glat, sigma=d["morph_sigma"])
        elif morph_type == "shell":
            return SkyShell(
                lon_0=glon,
                lat_0=glat,
                # TODO: probably we shouldn't guess a shell width here!
                radius=0.8 * d["morph_sigma"],
                width=0.2 * d["morph_sigma"],
            )
        elif morph_type == "none":
            raise NoDataAvailableError(
                "No spatial model available: {}".format(self.name)
            )
        else:
            raise NotImplementedError("Unknown spatial model: {!r}".format(morph_type))

    @property
    def sky_model(self):
        """Source sky model (`~gammapy.cube.models.SkyModel`)."""
        spatial_model = self.spatial_model
        spectral_model = self.spectral_model
        return SkyModel(spatial_model, spectral_model, name=self.name)

    def _add_source_meta(self, table):
        """Copy over some info to table.meta"""
        d = self.data
        m = table.meta
        m["origin"] = "Data from gamma-cat"
        m["source_id"] = d["source_id"]
        m["common_name"] = d["common_name"]
        m["reference_id"] = d["reference_id"]

    @property
    def _flux_points_table_formatted(self):
        """Returns formatted version of self.flux_points.table"""
        table = self.flux_points.table.copy()
        table["e_ref"].format = ".3f"
        flux_cols = ["dnde", "dnde_errn", "dnde_errp", "dnde_err"]
        for _ in set(table.colnames) & set(flux_cols):
            table[_].format = ".3e"
        return table

    @property
    def flux_points(self):
        """Differential flux points (`~gammapy.spectrum.FluxPoints`)."""
        d = self.data
        table = Table()
        table.meta["SED_TYPE"] = "dnde"
        self._add_source_meta(table)

        valid = np.isfinite(d["sed_e_ref"].value)

        if valid.sum() == 0:
            raise NoDataAvailableError("No flux points available: {}".format(self.name))

        table["e_ref"] = d["sed_e_ref"]
        table["e_min"] = d["sed_e_min"]
        table["e_max"] = d["sed_e_max"]

        table["dnde"] = d["sed_dnde"]
        table["dnde_err"] = d["sed_dnde_err"]
        table["dnde_errn"] = d["sed_dnde_errn"]
        table["dnde_errp"] = d["sed_dnde_errp"]
        table["dnde_ul"] = d["sed_dnde_ul"]

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
        return self.data["morph_type"] == "point"


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

    name = "gamma-cat"
    description = "An open catalog of gamma-ray sources"
    source_object_class = SourceCatalogObjectGammaCat

    def __init__(self, filename="$GAMMA_CAT/output/gammacat.fits.gz"):
        filename = str(make_path(filename))
        table = Table.read(filename, hdu=1)
        self.filename = filename

        source_name_key = "common_name"
        source_name_alias = ("other_names", "gamma_names")
        super(SourceCatalogGammaCat, self).__init__(
            table=table,
            source_name_key=source_name_key,
            source_name_alias=source_name_alias,
        )

    def to_sky_models(self):
        """Convert to a `~gammapy.cube.models.SkyModels`.

        TODO: add an option whether to skip or raise on missing models or data.
        """
        source_list = []

        for source_idx in range(len(self.table)):
            source = self[source_idx]
            try:
                source_list.append(source.sky_model)
            except NoDataAvailableError:
                log.warning(
                    "Skipping source {} (missing data in gamma-cat)".format(source.name)
                )
                continue

        return SkyModels(source_list)


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
    def from_index_file(cls, filename="$GAMMA_CAT/output/gammacat-datasets.json"):
        """Create from index file."""
        path = make_path(filename)
        # TODO: make a list of `GammaCatResource`, as well as a dict by ``resource_id`` for lookup!
        data_index = json.load(path.read_text())
        return cls(data_index=data_index)

    def __str__(self):
        ss = "version = {}".format(self.data_index["info"]["version"])
        return ss


@functools.total_ordering
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

    _NA_FILL = dict(file_id=-1, type="none", location="none")

    def __init__(
        self, source_id, reference_id, file_id=-1, type="none", location="none"
    ):
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
        return "|".join(
            (str(self.source_id), self.reference_id, str(self.file_id), self.type)
        )

    def __repr__(self):
        fmt = "{}(source_id={!r}, reference_id={!r}, file_id={!r}, type={!r}, location={!r})"
        return fmt.format(
            self.__class__.__name__,
            self.source_id,
            str(self.reference_id),
            self.file_id,
            str(self.type),
            str(self.location),
        )

    def __eq__(self, other):
        return self.to_namedtuple() == other.to_namedtuple()

    def __lt__(self, other):
        return self.to_namedtuple() < other.to_namedtuple()

    def to_namedtuple(self):
        """Convert to `collections.namedtuple`."""
        d = self.to_dict()
        return namedtuple("GammaCatResourceNamedTuple", d.keys())(**d)

    def to_dict(self):
        """Convert to `collections.OrderedDict`."""
        data = OrderedDict()
        data["source_id"] = self.source_id
        data["reference_id"] = self.reference_id
        data["file_id"] = self.file_id
        data["type"] = self.type
        data["location"] = self.location
        return data

    @classmethod
    def from_dict(cls, data):
        """Create from dict."""
        return cls(
            source_id=data["source_id"],
            reference_id=data["reference_id"],
            file_id=data.get("file_id", cls._NA_FILL["file_id"]),
            type=data.get("type", cls._NA_FILL["type"]),
            location=data.get("location", cls._NA_FILL["location"]),
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
        return "{}(n_resources={})".format(self.__class__.__name__, len(self.resources))

    def __eq__(self, other):
        if len(self.resources) != len(other.resources):
            return False
        return all(a == b for (a, b) in zip(self.resources, other.resources))

    @property
    def unique_source_ids(self):
        """Sorted list of unique source IDs (list of int)."""
        return sorted(set([resource.source_id for resource in self.resources]))

    @property
    def unique_reference_ids(self):
        """Sorted list of unique source IDs (list of str)."""
        return sorted(set([resource.reference_id for resource in self.resources]))

    @property
    def global_ids(self):
        """List of global resource IDs (list of str).

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
