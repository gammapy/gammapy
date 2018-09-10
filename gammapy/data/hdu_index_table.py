# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import numpy as np
import logging
from astropy.io import fits
from astropy.table import Table
from astropy.utils import lazyproperty
from ..utils.scripts import make_path

__all__ = ["HDULocation", "HDUIndexTable"]

log = logging.getLogger(__name__)


class HDULocation(object):
    """HDU localisation, loading and Gammapy object mapper.

    This represents one row in `HDUIndexTable`.

    It's more a helper class, that is wrapped by `~gammapy.data.DataStoreObservation`,
    usually those objects will be used to access data.

    See also :ref:`gadf:hdu-index`.
    """

    def __init__(
        self, obs_id, hdu_type, hdu_class, base_dir, file_dir, file_name, hdu_name
    ):
        self.obs_id = obs_id
        self.hdu_type = hdu_type
        self.hdu_class = hdu_class
        self.base_dir = base_dir
        self.file_dir = file_dir
        self.file_name = file_name
        self.hdu_name = hdu_name

    def info(self, file=None):
        """Print some summary info to stdout."""
        if not file:
            file = sys.stdout

        print("OBS_ID = {}".format(self.obs_id), file=file)
        print("HDU_TYPE = {}".format(self.hdu_type), file=file)
        print("HDU_CLASS = {}".format(self.hdu_class), file=file)
        print("BASE_DIR = {}".format(self.base_dir), file=file)
        print("FILE_DIR = {}".format(self.file_dir), file=file)
        print("FILE_NAME = {}".format(self.file_name), file=file)
        print("HDU_NAME = {}".format(self.hdu_name), file=file)

    def path(self, abs_path=True):
        """Full filename path.

        Include ``base_dir`` if ``abs_path`` is True.
        """
        if abs_path:
            return make_path(self.base_dir) / self.file_dir / self.file_name
        else:
            return make_path(self.file_dir) / self.file_name

    def get_hdu(self):
        """Get HDU."""
        filename = str(self.path(abs_path=True))
        # Here we're intentionally not calling `with fits.open`
        # because we don't want the file to remain open.
        hdu_list = fits.open(filename, memmap=False)
        return hdu_list[self.hdu_name]

    def load(self):
        """Load HDU as appropriate class.

        TODO: this should probably go via an extensible registry.
        """
        hdu_class = self.hdu_class
        filename = self.path()
        hdu = self.hdu_name

        if hdu_class == "events":
            from ..data import EventList

            return EventList.read(filename, hdu=hdu)
        elif hdu_class == "gti":
            from ..data import GTI

            return GTI.read(filename, hdu=hdu)
        elif hdu_class == "aeff_2d":
            from ..irf import EffectiveAreaTable2D

            return EffectiveAreaTable2D.read(filename, hdu=hdu)
        elif hdu_class == "edisp_2d":
            from ..irf import EnergyDispersion2D

            return EnergyDispersion2D.read(filename, hdu=hdu)
        elif hdu_class == "psf_table":
            from ..irf import PSF3D

            return PSF3D.read(filename, hdu=hdu)
        elif hdu_class == "psf_3gauss":
            from ..irf import EnergyDependentMultiGaussPSF

            return EnergyDependentMultiGaussPSF.read(filename, hdu=hdu)
        elif hdu_class == "psf_king":
            from ..irf import PSFKing

            return PSFKing.read(filename, hdu=hdu)
        elif hdu_class == "bkg_2d":
            from ..irf import Background2D

            return Background2D.read(filename, hdu=hdu)
        elif hdu_class == "bkg_3d":
            from ..irf import Background3D

            return Background3D.read(filename, hdu=hdu)
        else:
            raise ValueError("Invalid hdu_class: {}".format(hdu_class))


class HDUIndexTable(Table):
    """HDU index table.

    See :ref:`gadf:hdu-index`.
    """

    VALID_HDU_TYPE = ["events", "gti", "aeff", "edisp", "psf", "bkg"]
    """Valid values for `HDU_TYPE`."""

    VALID_HDU_CLASS = [
        "events",
        "gti",
        "aeff_2d",
        "edisp_2d",
        "psf_table",
        "psf_3gauss",
        "psf_king",
        "bkg_2d",
        "bkg_3d",
    ]
    """Valid values for `HDU_CLASS`."""

    @classmethod
    def read(cls, filename, **kwargs):
        """Read :ref:`gadf:hdu-index`.

        Parameters
        ----------
        filename : `~gammapy.extern.pathlib.Path`, str
            Filename
        """
        filename = make_path(filename)
        table = super(HDUIndexTable, cls).read(str(filename), **kwargs)
        table.meta["BASE_DIR"] = filename.parent.as_posix()

        return table

    @property
    def base_dir(self):
        """Base directory."""
        return make_path(self.meta.get("BASE_DIR", ""))

    def hdu_location(self, obs_id, hdu_type=None, hdu_class=None):
        """Create `HDULocation` for a given selection.

        Parameters
        ----------
        obs_id : int
            Observation ID
        hdu_type : str
            HDU type (see `~gammapy.data.HDUIndexTable.VALID_HDU_TYPE`)
        hdu_class : str
            HDU class (see `~gammapy.data.HDUIndexTable.VALID_HDU_CLASS`)

        Returns
        -------
        location : `~gammapy.data.HDULocation`
            HDU location
        """
        self._validate_selection(obs_id=obs_id, hdu_type=hdu_type, hdu_class=hdu_class)

        idx = self.row_idx(obs_id=obs_id, hdu_type=hdu_type, hdu_class=hdu_class)

        if len(idx) == 1:
            idx = idx[0]
        elif len(idx) == 0:
            raise IndexError(
                "No HDU found matching: OBS_ID = {}, HDU_TYPE = {}, HDU_CLASS = {}"
                "".format(obs_id, hdu_type, hdu_class)
            )
        else:
            idx = idx[0]
            log.warning(
                "Found multiple HDU matching: OBS_ID = {}, HDU_TYPE = {}, HDU_CLASS = {}."
                "".format(obs_id, hdu_type, hdu_class)
                + " Returning the first entry, which has HDU_TYPE = {} and HDU_CLASS = {}"
                "".format(self[idx]["HDU_TYPE"], self[idx]["HDU_CLASS"])
            )

        return self.location_info(idx)

    def _validate_selection(self, obs_id, hdu_type, hdu_class):
        """Validate HDU selection.

        The goal is to give helpful error messages to the user.
        """
        if hdu_type is None and hdu_class is None:
            raise ValueError("You have to specify `hdu_type` or `hdu_class`.")

        if hdu_type and hdu_type not in self.VALID_HDU_TYPE:
            msg = "Invalid hdu_type: {}. ".format(hdu_type)
            valid = [str(_) for _ in self.VALID_HDU_TYPE]
            msg += "Valid values are: {}".format(valid)
            raise ValueError(msg)

        if hdu_class and hdu_class not in self.VALID_HDU_CLASS:
            msg = "Invalid hdu_class: {}. ".format(hdu_class)
            valid = [str(_) for _ in self.VALID_HDU_CLASS]
            msg += "Valid values are: {}".format(valid)
            raise ValueError(msg)

        if obs_id not in self["OBS_ID"]:
            raise IndexError("No entry available with OBS_ID = {}".format(obs_id))

    def row_idx(self, obs_id, hdu_type=None, hdu_class=None):
        """Table row indices for a given selection.

        Parameters
        ----------
        obs_id : int
            Observation ID
        hdu_type : str
            HDU type (see `~gammapy.data.HDUIndexTable.VALID_HDU_TYPE`)
        hdu_class : str
            HDU class (see `~gammapy.data.HDUIndexTable.VALID_HDU_CLASS`)

        Returns
        -------
        idx : list of int
            List of row indices matching the selection.
        """
        selection = self["OBS_ID"] == obs_id

        if hdu_class:
            is_hdu_class = self._hdu_class_stripped == hdu_class
            selection &= is_hdu_class

        if hdu_type:
            is_hdu_type = self._hdu_type_stripped == hdu_type
            selection &= is_hdu_type

        idx = np.where(selection)[0]
        return list(idx)

    def location_info(self, idx):
        """Create `HDULocation` for a given row index."""
        row = self[idx]
        return HDULocation(
            obs_id=row["OBS_ID"],
            hdu_type=row["HDU_TYPE"].strip(),
            hdu_class=row["HDU_CLASS"].strip(),
            base_dir=self.base_dir.as_posix(),
            file_dir=row["FILE_DIR"].strip(),
            file_name=row["FILE_NAME"].strip(),
            hdu_name=row["HDU_NAME"].strip(),
        )

    @lazyproperty
    def _hdu_class_stripped(self):
        return np.array([_.strip() for _ in self["HDU_CLASS"]])

    @lazyproperty
    def _hdu_type_stripped(self):
        return np.array([_.strip() for _ in self["HDU_TYPE"]])

    @lazyproperty
    def obs_id_unique(self):
        """Observation IDs (unique)."""
        return np.unique(np.sort(self["OBS_ID"]))

    @lazyproperty
    def hdu_type_unique(self):
        """HDU types (unique)."""
        return list(np.unique(np.sort([_.strip() for _ in self["HDU_TYPE"]])))

    @lazyproperty
    def hdu_class_unique(self):
        """HDU classes (unique)."""
        return list(np.unique(np.sort([_.strip() for _ in self["HDU_CLASS"]])))

    def summary(self):
        """Summary report (str)"""
        obs_id = self.obs_id_unique
        return "\n".join(
            [
                "HDU index table:",
                "BASE_DIR: {}".format(self.base_dir),
                "Rows: {}".format(len(self)),
                "OBS_ID: {} -- {}".format(obs_id[0], obs_id[-1]),
                "HDU_TYPE: {}".format(self.hdu_type_unique),
                "HDU_CLASS: {}".format(self.hdu_class_unique),
            ]
        )
