"""Tools to create profiles (i.e. 1D "slices" from 2D images)."""
import numpy as np
from astropy import units as u
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import Angle, SkyCoord
from regions import RectangleSkyRegion, PixCoord
from gammapy.maps import MapAxis
from gammapy.utils.table import table_from_row_data
from gammapy.estimators import ImageProfile
from gammapy.stats import WStatCountsStatistic, CashCountsStatistic
from gammapy.datasets import SpectrumDatasetOnOff
from gammapy.modeling.models import SkyModel, PowerLawSpectralModel

__all__ = ["ExcessProfileEstimator", "make_orthogonal_rectangle_sky_regions"]


class ExcessProfileEstimator:
    """Estimate profile from a DataSet.

    Parameters
    ----------
    regions : list of `regions`
        regions to use
    axis : `~gammapy.maps.MapAxis`
        Radial axis of the profiles
    n_sigma : float (optional)
        Number of sigma to compute errors
    n_sigma_ul : float (optional)
        Number of sigma to compute upper limit

    Example
    --------
    This example shows how to compute a counts profile for the Fermi galactic
    center region::

        import matplotlib.pyplot as plt
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from gammapy.data import GTI
        from gammapy.estimators import ExcessProfileEstimator, make_orthogonal_rectangle_sky_regions, image_profile
        from gammapy.datasets import Datasets

        # load example data
        datasets = Datasets.read("$GAMMAPY_DATA/fermi-3fhl-crab/",
            "Fermi-LAT-3FHL_datasets.yaml", "Fermi-LAT-3FHL_models.yaml")
        # configuration
        datasets[0].gti = GTI.create("0s", "1e7s", "2010-01-01")

        # creation of the boxes and axis
        start_line = SkyCoord(182.5, -5.8, unit='deg', frame='galactic')
        end_line = SkyCoord(186.5, -5.8, unit='deg', frame='galactic')
        boxes, axis = make_orthogonal_rectangle_sky_regions(start_line,
                                        end_line,
                                        datasets[0].counts.geom.wcs,
                                        1.*u.deg,
                                        11)

        # set up profile estimator and run
        prof_maker = ExcessProfileEstimator(boxes, axis)
        fermi_prof = prof_maker.run(datasets[0])

        # smooth and plot the data using the ImageProfile class
        fermi_prof.peek()
        plt.show()

        ax = plt.gca()
        ax.set_yscale('log')
        ax = fermi_prof.plot("flux", ax=ax)

    """

    def __init__(self, regions, axis, n_sigma = 1, n_sigma_ul=3):
        self.regions = regions
        self.axis = axis
        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul
        self.spectrum_datasets = None

    def get_spectrum_datasets(self, dataset):
        """ Utility to make the final `~gammapy.datasts.Datasets`

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.MapDatasetOnOff`
            the dataset to use for profile extraction
        Returns
        --------
        datasets : `~gammapy.datasts.Datasets`
            the list of `~gammapy.datasets.SpectrumDataset` computed in each box
        """
        sp_datasets = []
        for reg in self.regions:
            spds = dataset.to_spectrum_dataset(reg)
            sp_datasets.append(spds)
        return sp_datasets

    def make_prof(self, sp_datasets, steps, spectrum=None):
        """ Utility to make the profile in each region

        Parameters
        ----------
        sp_datasets : `~gammapy.datasets.MapDatasets` of `~gammapy.datasets.SpectrumDataset` or \
        `~gammapy.datasets.SpectrumDatasetOnOff`
            the dataset to use for profile extraction
        steps : list of str
            the steps to be used.
        spectrum : `~gammapy.modeling.models.SpectralModel`
            Spectral model to compute the fluxes or brightness.
            Default is power-law with spectral index of 2.
        Returns
        --------
        results : list of dictionary
            the list of results (list of keys: x_min, x_ref, x_max, alpha, counts, background, excess, ts, sqrt_ts, \
            err, errn, errp, ul, exposure, solid_angle)
        """
        if steps == "all":
            steps = ["err", "ts", "errn-errp", "ul"]
        if spectrum is None:
            spectrum = PowerLawSpectralModel(index=2.0, amplitude="1.e-12 cm-2 s-1 TeV-1", reference=1*u.TeV)

        results = []
        for index, spds in enumerate(sp_datasets):
            old_model = None
            if spds.models is not None:
                old_model = spds.models
            spds.models = SkyModel(spectral_model=spectrum)
            e_reco = spds.counts.geom.get_axis_by_name("energy").edges

            # ToDo: When the function to_spectrum_dataset will manage the masks, use the following line
            # mask = spds.mask if spds.mask is not None else slice(None)
            mask = slice(None)
            if isinstance(spds, SpectrumDatasetOnOff):
                # stats_sum = WStatCountsStatistic(
                #     spds.counts.data[mask].sum(),
                #     spds.counts_off.data[mask].sum(),
                #     spds.alpha.data[0,0,0] # At some point, should replace with averaging over energy
                # )
                stats = WStatCountsStatistic(
                    spds.counts.data[mask][:, 0, 0],
                    spds.counts_off.data[mask][:, 0, 0],
                    spds.alpha.data[mask][:, 0, 0]
                )

            else:
                # stats_sum = CashCountsStatistic(
                #     spds.counts.data[mask].sum(),
                #     spds.background.data[mask].sum(),
                # )
                stats = CashCountsStatistic(
                    spds.counts.data[mask][:, 0, 0],
                    spds.background.data[mask][:, 0, 0],
                )

            result = {
                "x_min": self.axis.edges[index],
                "x_max": self.axis.edges[index+1],
                "x_ref": self.axis.center[index],
                "energy_edge": e_reco
            }
            if isinstance(spds, SpectrumDatasetOnOff):
                result["alpha"] = stats.alpha
            result.update({
                "counts": stats.n_on,
                "background": stats.background,
                "excess": stats.excess,
            })

            if "ts" in steps:
                result["ts"] = stats.delta_ts
                result["sqrt_ts"] = stats.significance

            if "err" in steps:
                result["err"] = stats.error

            if "errn-errp" in steps:
                result["errn"] = stats.compute_errn(self.n_sigma)
                result["errp"] = stats.compute_errp(self.n_sigma)

            if "ul" in steps:
                result["ul"] = stats.compute_upper_limit(self.n_sigma_ul)

            npred = spds.npred_sig().data[mask][:, 0, 0]
            e_reco_lo = e_reco[:-1]
            e_reco_hi = e_reco[1:]
            flux = stats.excess / npred * \
                            spds.models[0].spectral_model.integral(e_reco_lo, e_reco_hi).value
            result["flux"] = flux

            if "err" in steps:
                result["flux_err"] = stats.error / stats.excess * flux

            if "errn-errp" in steps:
                result["flux_errn"] = np.abs(result["errn"]) / stats.excess * flux
                result["flux_errp"] = result["errp"] / stats.excess * flux

            if "ul" in steps:
                result["flux_ul"] = result["ul"] / stats.excess * flux

            result["solid_angle"] = np.full(result['counts'].shape, spds.counts.geom.solid_angle()) * u.steradian

            results.append(result)
            if old_model is not None:
                spds.models = old_model

        return results

    def run(self, dataset, steps="all", spectrum=None):
        """Make the profiles

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.MapDatasetOnOff`
            the dataset to use for profile extraction
        steps : list of str
            the steps to be used.
        spectrum : `~gammapy.modeling.models.SpectralModel`
            Spectral model to compute the fluxes or brightness.
            Default is power-law with spectral index of 2.

        Returns
        --------
        imageprofile : `~gammapy.estimators.ImageProfile`
            Return an image profile class containing the result
        """
        self.spectrum_datasets = self.get_spectrum_datasets(dataset)
        results = self.make_prof(self.spectrum_datasets, steps, spectrum)
        table = table_from_row_data(results)
        if isinstance(self.regions[0], RectangleSkyRegion):
            table.meta["PROFILE_TYPE"] = "orthogonal_rectangle"
        if spectrum is None:
            spectrum = PowerLawSpectralModel(index=2.0)
        table.meta["SPECTRAL_MODEL"] = spectrum.to_dict()

        return ImageProfile(table)


def make_orthogonal_boxes_new(start_pos, end_pos, wcs, fullwidth, nbins=1):
    """Utility returning an array of regions to make orthogonal projections

    Parameters
    ----------
    start_pos : `~astropy.regions.SkyCoord'
        First sky coordinate defining the line to which the orthogonal boxes made
    end_pos : `~astropy.regions.SkyCoord'
        Second sky coordinate defining the line to which the orthogonal boxes made
    fullwidth : Angle
        Full width of the orthogonal dimension of the boxes
    wcs : `~astropy.wcs.WCS`
        WCS projection object
    nbins : int
        Number of boxes along the line

    Returns
    --------
    regions : Array of `~astropy.regions`
        Regions in which the profiles are made
    rad_axis : `~gammapy.maps.MapAxis`
        Radial axis of the profiles
    """
    pix_start = start_pos.to_pixel(wcs)
    pix_stop = end_pos.to_pixel(wcs)

    points = np.linspace(start=pix_start, stop=pix_stop, num=nbins+1).T
    centers = 0.5*(points[:, :-1]+points[:, 1:])
    coords = SkyCoord.from_pixel(centers[0], centers[1], wcs)
    box_width = start_pos.separation(end_pos).to("rad")/nbins
    rot_angle = end_pos.position_angle(start_pos)-90*u.deg
    regions = []
    for i in range(nbins):
        reg = RectangleSkyRegion(center=coords[i],
                                 width=box_width,
                                 height=u.Quantity(fullwidth),
                                 angle=rot_angle)
        regions.append(reg)
    
    axis = MapAxis.from_nodes(coords[0].separation(coords))
    axis.name = 'projected distance'

    return regions, axis


def make_orthogonal_rectangle_sky_regions(start_pos, end_pos, wcs, fullwidth, nbins=1):
    """Utility returning an array of rectangular regions that are orthogonal to a like

    This function is used to produce excess profiles (plus flux and brightness) with the estimator \
    `~gammapy.estimators.ExcessProfileEstimator`. Its output, a `~gammapy.estimators.ImageProfile`, contains then the \
    following metadata: `PROFILE_TYPE` = "orthogonal_rectangle".

    Parameters
    ----------
    start_pos : `~astropy.regions.SkyCoord'
        First sky coordinate defining the line to which the orthogonal boxes made
    end_pos : `~astropy.regions.SkyCoord'
        Second sky coordinate defining the line to which the orthogonal boxes made
    fullwidth : `~astropy.units.Quantity`
        Full width in degrees of the orthogonal dimension of the boxes
    wcs : `~astropy.wcs.WCS`
        WCS projection object
    nbins : int
        Number of boxes along the line

    Returns
    --------
    regions : Array of `~astropy.regions`
        Regions in which the profiles are made
    rad_axis : `~gammapy.maps.MapAxis`
        Radial axis of the profiles
    """

    pix_start_pos = PixCoord.from_sky(start_pos, wcs)
    pix_end_pos = PixCoord.from_sky(end_pos, wcs)
    xx_s = pix_start_pos.x
    yy_s = pix_start_pos.y
    xx_e = pix_end_pos.x
    yy_e = pix_end_pos.y

    rot_ang = Angle(np.arctan2(yy_e-yy_s, xx_e-xx_s) * u.rad)
    length = _pix_dist(pix_start_pos, pix_end_pos)
    binwidth = length / nbins
    stepx = binwidth * np.cos(rot_ang.rad)
    stepy = binwidth * np.sin(rot_ang.rad)
    binz = proj_plane_pixel_scales(wcs)[0]

    # Line center
    xx_c = (xx_s + xx_e) / 2.
    yy_c = (yy_s + yy_e) / 2.
    pix_center_pos = PixCoord(xx_c, yy_c)

    regions = []
    edges = []
    xx = xx_s
    yy = yy_s
    pix_edge_pos = PixCoord(xx, yy)
    dist = _pix_dist(pix_edge_pos, pix_center_pos)*binz
    sign = 1.
    if xx < xx_c:
        sign = -1.
    edges.append(sign * dist)

    for _ in range(nbins):
        xx1 = xx + stepx
        yy1 = yy + stepy
        pix_center_reg = PixCoord((xx1 + xx) / 2., (yy1 + yy) / 2.)
        rectangle_sky = RectangleSkyRegion(center=pix_center_reg.to_sky(wcs),
                                           width=binwidth*binz*u.deg, height=fullwidth,
                                           angle=rot_ang.degree * u.deg)
        regions.append(rectangle_sky)
        dist = _pix_dist(PixCoord(xx1, yy1), pix_center_pos)*binz
        sign = 1.
        if xx1 < xx_c:
            sign = -1.
        edges.append(sign * dist)

        xx = xx1
        yy = yy1

    axis = MapAxis.from_edges(edges, unit='degree')
    axis.name = 'projected distance'
    return regions, axis


def _pix_dist(pix_coord1, pix_coord2):
    xx_s = pix_coord1.x
    yy_s = pix_coord1.y
    xx_e = pix_coord2.x
    yy_e = pix_coord2.y
    return np.sqrt((xx_e-xx_s)**2+(yy_e-yy_s)**2)
