# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy.table import Table
from astropy.utils import lazyproperty
from gammapy.utils.fits import HDULocation
from gammapy.utils.scripts import make_path

__all__ = ["HDUIndexTable"]

log = logging.getLogger(__name__)


class HDUIndexTable(Table):
    """HDU index table.

    See :ref:`gadf:hdu-index`.
    """

    VALID_HDU_TYPE = ["events", "gti", "aeff", "edisp", "psf", "bkg", "rad_max"]
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
        "rad_max_2d",
    ]
    """Valid values for `HDU_CLASS`."""

    @classmethod
    def read(cls, filename, **kwargs):
        """Read :ref:`gadf:hdu-index`.

        Parameters
        ----------
        filename : `pathlib.Path`, str
            Filename
        """
        filename = make_path(filename)
        table = super().read(filename, **kwargs)
        table.meta["BASE_DIR"] = filename.parent.as_posix()

        # TODO: this is a workaround for the joint-crab validation with astropy>4.0.
        # TODO: Remove when handling of empty columns is clarified
        table["FILE_DIR"].fill_value = ""

        return table.filled()

    @property
    def base_dir(self):
        """Base directory."""
        return make_path(self.meta.get("BASE_DIR", ""))

    def hdu_location(self, obs_id, hdu_type=None, hdu_class=None, warn_missing=True):
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
            if warn_missing:
                log.warning(
                    f"No HDU found matching: OBS_ID = {obs_id}, HDU_TYPE = {hdu_type},"
                    " HDU_CLASS = {hdu_class}"
                )
            return None
        else:
            idx = idx[0]
            log.warning(
                f"Found multiple HDU matching: OBS_ID = {obs_id}, HDU_TYPE = {hdu_type},"
                " HDU_CLASS = {hdu_class}."
                f" Returning the first entry, which has "
                f"HDU_TYPE = {self[idx]['HDU_TYPE']} and HDU_CLASS = {self[idx]['HDU_CLASS']}"
            )

        return self.location_info(idx)

    def _validate_selection(self, obs_id, hdu_type, hdu_class):
        """Validate HDU selection.

        The goal is to give helpful error messages to the user.
        """
        if hdu_type is None and hdu_class is None:
            raise ValueError("You have to specify `hdu_type` or `hdu_class`.")

        if hdu_type and hdu_type not in self.VALID_HDU_TYPE:
            valid = [str(_) for _ in self.VALID_HDU_TYPE]
            raise ValueError(f"Invalid hdu_type: {hdu_type}. Valid values are: {valid}")

        if hdu_class and hdu_class not in self.VALID_HDU_CLASS:
            valid = [str(_) for _ in self.VALID_HDU_CLASS]
            raise ValueError(
                f"Invalid hdu_class: {hdu_class}. Valid values are: {valid}"
            )

        if obs_id not in self["OBS_ID"]:
            raise IndexError(f"No entry available with OBS_ID = {obs_id}")

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
        """Summary report (str)."""
        obs_id = self.obs_id_unique
        return (
            "HDU index table:\n"
            f"BASE_DIR: {self.base_dir}\n"
            f"Rows: {len(self)}\n"
            f"OBS_ID: {obs_id[0]} -- {obs_id[-1]}\n"
            f"HDU_TYPE: {self.hdu_type_unique}\n"
            f"HDU_CLASS: {self.hdu_class_unique}\n"
        )
