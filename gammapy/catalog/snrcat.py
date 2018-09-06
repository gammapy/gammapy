# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from ..extern import six
from astropy.utils.data import download_file
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table, Column
from .core import SourceCatalog, SourceCatalogObject

__all__ = ["SourceCatalogSNRcat", "SourceCatalogObjectSNRcat"]


class SourceCatalogObjectSNRcat(SourceCatalogObject):
    """One source from the SNRcat catalog."""

    pass


class SourceCatalogSNRcat(SourceCatalog):
    """SNRcat supernova remnant catalog.

    `SNRcat <http://www.physics.umanitoba.ca/snr/SNRcat/>`__
    is a census of high-energy observations of Galactic supernova remnants.

    This function downloads the following CSV-format tables
    and adds some useful columns and unit information:

    * http://www.physics.umanitoba.ca/snr/SNRcat/SNRdownload.php?table=SNR
    * http://www.physics.umanitoba.ca/snr/SNRcat/SNRdownload.php?table=OBS

    This only represents a subset of the information available in SNRcat,
    to get at the rest we would have to scrape their web pages.

    * ``table`` (`~astropy.table.Table`) -- SNR info table
    * ``obs_table`` (`~astropy.table.Table`) -- High-energy observation info table

    Each table has a ``version`` string containing the download date in the ``table.meta`` dictionary.
    """

    name = "snrcat"
    description = "SNRcat supernova remnant catalog."
    source_object_class = SourceCatalogObjectSNRcat

    def __init__(self, cache=False):
        # TODO: load from gammapy-extra?
        # At least optionally?
        self.snr_table = _fetch_catalog_snrcat_snr_table(cache=cache)
        self.obs_table = _fetch_catalog_snrcat_obs_table(cache=cache)

        super(SourceCatalogSNRcat, self).__init__(table=self.snr_table)


def _fetch_catalog_snrcat_snr_table(cache):
    url = "http://www.physics.umanitoba.ca/snr/SNRcat/SNRdownload.php?table=SNR"
    filename = download_file(url, cache=cache)

    # Note: currently the first line contains this comment, which we skip via `header_start=1`
    table = Table.read(filename, format="ascii.csv", header_start=1, delimiter=";")
    table.meta["url"] = url
    table.meta["version"] = _snrcat_parse_download_date(filename)

    # TODO: doesn't work properly ... don't call for now.
    # _snrcat_fix_na(table)

    table.rename_column("G", "Source_Name")

    table.rename_column("J2000_ra (hh:mm:ss)", "RAJ2000_str")
    table.rename_column("J2000_dec (dd:mm:ss)", "DEJ2000_str")

    data = Angle(table["RAJ2000_str"], unit="hour").deg
    index = table.index_column("RAJ2000_str") + 1
    table.add_column(Column(data=data, name="RAJ2000", unit="deg"), index=index)

    data = Angle(table["DEJ2000_str"], unit="deg").deg
    index = table.index_column("DEJ2000_str") + 1
    table.add_column(Column(data=data, name="DEJ2000", unit="deg"), index=index)

    radec = SkyCoord(table["RAJ2000"], table["DEJ2000"], unit="deg")
    galactic = radec.galactic
    table.add_column(Column(data=galactic.l.deg, name="GLON", unit="deg"))
    table.add_column(Column(data=galactic.b.deg, name="GLAT", unit="deg"))

    table.rename_column("age_min (yr)", "age_min")
    table["age_min"].unit = "year"
    table.rename_column("age_max (yr)", "age_max")
    table["age_max"].unit = "year"
    distance = np.mean([table["age_min"], table["age_max"]], axis=0)
    index = table.index_column("age_max") + 1
    table.add_column(Column(distance, name="age", unit="year"), index=index)

    table.rename_column("distance_min (kpc)", "distance_min")
    table["distance_min"].unit = "kpc"
    table.rename_column("distance_max (kpc)", "distance_max")
    table["distance_max"].unit = "kpc"
    distance = np.mean([table["distance_min"], table["distance_max"]], axis=0)
    index = table.index_column("distance_max") + 1
    table.add_column(Column(distance, name="distance", unit="kpc"), index=index)

    table.rename_column("size_radio", "diameter_radio_str")
    diameter_radio_mean = _snrcat_parse_diameter(table["diameter_radio_str"])
    index = table.index_column("diameter_radio_str") + 1
    table.add_column(
        Column(diameter_radio_mean, name="diameter_radio_mean", unit="arcmin"),
        index=index,
    )

    table.rename_column("size_X", "diameter_xray_str")
    diameter_xray_mean = _snrcat_parse_diameter(table["diameter_xray_str"])
    index = table.index_column("diameter_xray_str") + 1
    table.add_column(
        Column(diameter_xray_mean, name="diameter_xray_mean", unit="arcmin"),
        index=index,
    )

    table.rename_column("size_coarse (arcmin)", "diameter_mean")
    table["diameter_mean"].unit = "arcmin"

    table.rename_column("size_imprecise", "diameter_mean_is_imprecise")

    return table


def _fetch_catalog_snrcat_obs_table(cache):
    url = "http://www.physics.umanitoba.ca/snr/SNRcat/SNRdownload.php?table=OBS"
    filename = download_file(url, cache=cache)

    # Note: currently the first line contains this comment, which we skip via `header_start=1`
    table = Table.read(filename, format="ascii.csv", header_start=1, delimiter=";")
    table.meta["url"] = url
    table.meta["version"] = _snrcat_parse_download_date(filename)

    # TODO: doesn't work properly ... don't call for now.
    # _snrcat_fix_na(table)

    return table


def _snrcat_fix_na(table):
    """Fix N/A entries in string columns in SNRcat."""
    for colname in table.colnames:
        if isinstance(table[colname][0], six.text_type):
            mask1 = table[colname] == "N / A"
            mask2 = table[colname] == "N/A"
            table[colname].mask = mask1 | mask2
            table[colname].fill_value = ""


def _snrcat_parse_download_date(filename):
    text = open(filename).readline()
    # Format: "This file was downloaded on 2015-06-07T03:39:53 CDT ..."
    tokens = text.split()
    date = tokens[5]
    return date[:10]


def _snrcat_parse_diameter(text_col):
    """Parse SNRcat diameter string column"""
    d_means = []
    for text in text_col:
        try:
            # Parse this text field:
            if "x" in text:
                a, b = text.split("x")
                d_major = Angle(a).arcmin
                d_minor = Angle(b).arcmin
            else:
                d_major = Angle(text).arcmin
                d_minor = d_major
            d_mean = _snr_mean_diameter(d_major, d_minor)
        except:
            # print('Parsing error:', text)
            d_mean = np.nan

        d_means.append(d_mean)

    return d_means


def _snr_mean_diameter(d_major, d_minor):
    """Compute geometric mean diameter (preserves area)"""
    diameter = np.sqrt(d_major * d_minor)
    # If no `d_minor` is given, use `d_major` as mean radius
    with np.errstate(invalid="ignore"):
        diameter = np.where(d_minor > 0, diameter, d_major)

    return diameter
