# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
import astropy.units as u
from gammapy.data import GTI
from gammapy.datasets import Datasets
from gammapy.maps import LabelMapAxis, Map, TimeMapAxis
from gammapy.utils.pbar import progress_bar
from .core import FluxPoints
from .sed import FluxPointsEstimator

__all__ = ["LightCurveEstimator"]

log = logging.getLogger(__name__)


class LightCurveEstimator(FluxPointsEstimator):
    """Estimate light curve.

    The estimator will apply flux point estimation on the source model component to datasets
    in each of the provided time intervals.  The normalization is the only
    parameter of the source model left free to vary. Other model components
    can be left free to vary with the reoptimize option.

    If no time intervals are provided, the estimator will use the time intervals
    defined by the datasets GTIs.

    To be included in the estimation, the dataset must have their GTI fully
    overlapping a time interval.

    Time intervals without any dataset GTI fully overlapping will be dropped. They will not
    be stored in the final lightcurve `FluxPoints` object.

    Parameters
    ----------
    time_intervals : list of `astropy.time.Time`
        Start and stop time for each interval to compute the LC
    source : str or int
        For which source in the model to compute the flux points. Default is 0
    energy_edges : `~astropy.units.Quantity`
        Energy edges of the light curve.
    atol : `~astropy.units.Quantity`
        Tolerance value for time comparison with different scale. Default 1e-6 sec.
    norm_min : float
        Minimum value for the norm used for the fit statistic profile evaluation.
    norm_max : float
        Maximum value for the norm used for the fit statistic profile evaluation.
    norm_n_values : int
        Number of norm values used for the fit statistic profile.
    norm_values : `numpy.ndarray`
        Array of norm values to be used for the fit statistic profile.
    n_sigma : int
        Number of sigma to use for asymmetric error computation. Default is 1.
    n_sigma_ul : int
        Number of sigma to use for upper limit computation. Default is 2.
    selection_optional : list of str
        Which steps to execute. Available options are:

            * "all": all the optional steps are executed
            * "errn-errp": estimate asymmetric errors.
            * "ul": estimate upper limits.
            * "scan": estimate fit statistic profiles.

        Default is None so the optional steps are not executed.
    fit : `Fit`
        Fit instance specifying the backend and fit options.
    reoptimize : bool
        Re-optimize other free model parameters. Default is True.

    Examples
    --------
    For a usage example see :doc:`/tutorials/analysis-time/light_curve` tutorial.

    """

    tag = "LightCurveEstimator"

    def __init__(self, time_intervals=None, atol="1e-6 s", **kwargs):
        self.time_intervals = time_intervals
        self.atol = u.Quantity(atol)
        super().__init__(**kwargs)

    def run(self, datasets):
        """Run light curve extraction.

        Normalize integral and energy flux between emin and emax.

        Parameters
        ----------
        datasets : list of `~gammapy.datasets.SpectrumDataset` or `~gammapy.datasets.MapDataset`
            Spectrum or Map datasets.

        Returns
        -------
        lightcurve : `~gammapy.estimators.FluxPoints`
            Light curve flux points
        """
        datasets = Datasets(datasets)

        if self.time_intervals is None:
            gti = datasets.gti
        else:
            gti = GTI.from_time_intervals(self.time_intervals)

        gti = gti.union(overlap_ok=False, merge_equal=False)

        rows = []
        valid_intervals = []
        for t_min, t_max in progress_bar(gti.time_intervals, desc="Time intervals"):
            datasets_to_fit = datasets.select_time(
                time_min=t_min, time_max=t_max, atol=self.atol
            )

            if len(datasets_to_fit) == 0:
                log.info(
                    f"No Dataset for the time interval {t_min} to {t_max}. Skipping interval."
                )
                continue

            valid_intervals.append([t_min, t_max])
            fp = self.estimate_time_bin_flux(datasets=datasets_to_fit)

            for name in ["counts", "npred", "npred_excess"]:
                fp._data[name] = self.expand_map(
                    fp._data[name], dataset_names=datasets.names
                )
            rows.append(fp)

        if len(rows) == 0:
            raise ValueError("LightCurveEstimator: No datasets in time intervals")

        gti = GTI.from_time_intervals(valid_intervals)
        axis = TimeMapAxis.from_gti(gti=gti)
        return FluxPoints.from_stack(
            maps=rows,
            axis=axis,
        )

    @staticmethod
    def expand_map(m, dataset_names):
        """Expand map in dataset axis

        Parameters
        ----------
        map : `Map`
            Map to expand.
        dataset_names : list of str
            Dataset names

        Returns
        -------
        map : `Map`
            Expanded map.
        """
        label_axis = LabelMapAxis(labels=dataset_names, name="dataset")
        geom = m.geom.replace_axis(axis=label_axis)
        result = Map.from_geom(geom, data=np.nan)

        coords = m.geom.get_coord(sparse=True)
        result.set_by_coord(coords, vals=m.data)
        return result

    def estimate_time_bin_flux(self, datasets):
        """Estimate flux point for a single energy group.

        Parameters
        ----------
        datasets : `~gammapy.modeling.Datasets`
            List of dataset objects

        Returns
        -------
        result : `FluxPoints`
            Resulting flux points.
        """
        return super().run(datasets)
