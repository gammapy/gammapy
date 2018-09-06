# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Example and test datasets.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import warnings
from astropy.units import Quantity, UnitsWarning
from astropy.io import fits
from astropy.table import Table
from .core import gammapy_extra

__all__ = [
    "load_poisson_stats_image",
    "load_tev_spectrum",
    "load_crab_flux_points",
    "load_diffuse_gamma_spectrum",
    "load_electron_spectrum",
]


def load_poisson_stats_image(extra_info=False, return_filenames=False):
    """Load Poisson statistics counts image of a Gaussian source on flat background.

    See poissson_stats_image/README.md for further info.
    TODO: add better description (extract from README?)

    Parameters
    ----------
    extra_info : bool
        If true, a dict of images is returned.
    return_filenames : bool
        If true, return filenames instead of images

    Returns
    -------
    data : numpy array or dict of arrays or filenames
        Depending on the ``extra_info`` and ``return_filenames`` options.
    """
    path = gammapy_extra.dir / "test_datasets/unbundled/poisson_stats_image"

    if extra_info:
        out = dict()
        for name in ["counts", "model", "source", "background", "exposure"]:
            filename = str(path / "{}.fits.gz".format(name))
            if return_filenames:
                out[name] = filename
            else:
                data = fits.getdata(filename)
                out[name] = data.astype("float64")
        if return_filenames:
            out["psf"] = str(path / "psf.json")
    else:
        filename = str(path / "counts.fits.gz")
        if return_filenames:
            out = filename
        else:
            out = fits.getdata(filename).astype("float64")

    if extra_info and not return_filenames:
        filename = str(path / "counts.fits.gz")
        out["header"] = fits.getheader(filename)

    return out


def load_tev_spectrum(source_name):
    """Load published TeV flux point measurements.

    TODO: give references to publications and describe the returned table.

    Parameters
    ----------
    source_name : str
        Source name

    Returns
    -------
    spectrum : `~astropy.table.Table`
        Energy spectrum as a table (one flux point per row).
    """
    if source_name == "crab":
        filename = gammapy_extra.filename(
            "test_datasets/unbundled/tev_spectra/crab_hess_spec.txt"
        )
    else:
        raise ValueError("Data not available for source: {!r}".format(source_name))

    names = ["energy", "flux", "flux_lo", "flux_hi"]
    table = Table.read(filename, format="ascii", names=names)
    table["flux_err"] = 0.5 * (table["flux_lo"] + table["flux_hi"])
    return table


def load_crab_flux_points(component="both"):
    """Load published Crab pulsar and nebula flux points.

    Besides the usual flux point columns, this table contains
    the following two columns:

    * component : {'pulsar', 'nebula'}
    * paper : Short string describing which point originates from which paper.

    TODO:

    * Add link to Crab flux point tutorial in Gammapy where these points are plotted.
    * Add html links to ADS directly in the docstring and as a table column.

    Parameters
    ----------
    component : {'pulsar', 'nebula', 'both'}
        Which emission component to include

    Returns
    -------
    flux_points : `~astropy.table.Table`
        Flux point table

    Notes
    -----
    This data compilation is from Buehler and Blandford, Rep. Prog. Phys. 77, 2014.
    It was contributed to Gammapy directly by Rolf Buehler via a pull request.

    The data for the nebula were taken from Meyer et al. Astron. Astrophys. 523 2010
    with the addition of the Fermi-LAT measurement reported in Buehler et al. ApJ 749 2012.

    The pulsar spectrum is reproduced from Kuiper et al Astron. Astrophys. 378 2001 .
    Additionally shown are infrared measurements reported in Sollerman et al. ApJ 537 2000
    and Tziamtzis et al. Astron. Astrophys. 508 2009, radio measurements referenced in
    Thompson et al. ApJ 516 1999 and gamma-ray measurements referenced in
    Aleksic et al. ApJ 742 2011, Aliu et al. Science 334 2011,
    Aleksic et al. Astron. Astrophys. 540 2012
    and Abdo et al. Astrophys. J. Suppl. Ser. 208 2013.
    """
    filename = gammapy_extra.filename(
        "test_datasets/unbundled/tev_spectra/crab_mwl.fits.gz"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnitsWarning)
        table = Table.read(filename)

    if component == "pulsar":
        mask = table["component"] == "pulsar"
        table = table[mask]
    elif component == "nebula":
        mask = table["component"] == "nebula"
        table = table[mask]
    elif component == "both":
        pass
    else:
        raise ValueError("Invalid component: {}".format(component))

    return table


def load_diffuse_gamma_spectrum(reference):
    """Load published diffuse gamma-ray spectrum.

    TODO: give references to publications and describe the returned table.

    Parameters
    ----------
    reference : {'Fermi', 'Fermi2'}
        Which publication.

    Returns
    -------
    spectrum : `~astropy.table.Table`
        Energy spectrum as a table (one flux point per row).
    """
    dir = gammapy_extra.dir / "test_datasets/unbundled/tev_spectra"

    if reference == "Fermi":
        filename = str(dir / "diffuse_isotropic_gamma_spectrum_fermi.txt")
    elif reference == "Fermi2":
        filename = str(dir / "diffuse_isotropic_gamma_spectrum_fermi2.txt")
    else:
        raise ValueError("Data not available for reference: {}".format(reference))

    return _read_diffuse_gamma_spectrum_fermi(filename)


def _read_diffuse_gamma_spectrum_fermi(filename):
    table = Table.read(
        filename, format="ascii", names=["energy", "flux", "flux_hi", "flux_lo"]
    )
    table["flux_err"] = 0.5 * (table["flux_lo"] + table["flux_hi"])

    table["energy"] = Quantity(table["energy"], "MeV").to("TeV")

    for colname in table.colnames:
        if "flux" in colname:
            energy = Quantity(table["energy"], "TeV")
            energy2_flux = Quantity(table[colname], "MeV cm^-2 s^-1 sr^-1")
            table[colname] = (energy2_flux / energy ** 2).to("m^-2 s^-1 TeV^-1 sr^-1")

    return table


def load_electron_spectrum(reference):
    """Load published electron spectrum.

    TODO: give references to publications and describe the returned table.

    Parameters
    ----------
    reference : {'HESS', 'HESS low energy', 'Fermi'}
        Which publication.

    Returns
    -------
    spectrum : `~astropy.table.Table`
        Energy spectrum as a table (one flux point per row).
    """
    dir = gammapy_extra.dir / "test_datasets/unbundled/tev_spectra"

    if reference == "HESS":
        filename = str(dir / "electron_spectrum_hess.txt")
        return _read_electron_spectrum_hess(filename)
    elif reference == "HESS low energy":
        filename = str(dir / "electron_spectrum_hess_low_energy.txt")
        return _read_electron_spectrum_hess(filename)
    elif reference == "Fermi":
        filename = str(dir / "electron_spectrum_fermi.txt")
        return _read_electron_spectrum_fermi(filename)
    else:
        raise ValueError("Data not available for reference: {}".format(reference))


def _read_electron_spectrum_hess(filename):
    table = Table.read(
        filename, format="ascii", names=["energy", "flux", "flux_lo", "flux_hi"]
    )
    table["flux_err"] = 0.5 * (table["flux_lo"] + table["flux_hi"])

    table["energy"] = Quantity(table["energy"], "GeV").to("TeV")

    # The ascii files store fluxes as (E ** 3) * dN / dE.
    # Here we change this to dN / dE.
    for colname in table.colnames:
        if "flux" in colname:
            energy = Quantity(table["energy"], "TeV")
            energy3_flux = Quantity(table[colname], "GeV^2 m^-2 s^-1 sr^-1")
            table[colname] = (energy3_flux / energy ** 3).to("m^-2 s^-1 TeV^-1 sr^-1")

    return table


def _read_electron_spectrum_fermi(filename):
    t = Table.read(filename, format="ascii")

    table = Table()
    table["energy"] = Quantity(t["E"], "GeV").to("TeV")
    table["flux"] = Quantity(t["y"], "m-2 s-1 GeV-1 sr-1").to("m-2 s-1 TeV-1 sr-1")
    val = 0.5 * (t["yerrtot_lo"] + t["yerrtot_up"])
    table["flux_err"] = Quantity(val, "m-2 s-1 GeV-1 sr-1").to("m-2 s-1 TeV-1 sr-1")

    return table
