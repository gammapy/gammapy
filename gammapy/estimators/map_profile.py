"""Tools to create profiles (i.e. 1D "slices" from 2D images)."""
import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import Angle, SkyCoord
from regions import PolygonPixelRegion, CircleAnnulusPixelRegion, RectangleSkyRegion, PixCoord
from gammapy.maps import MapAxis
from gammapy.utils.table import table_from_row_data
from gammapy.visualization import ImageProfile
from gammapy.stats import WStatCountsStatistic, CashCountsStatistic
from gammapy.datasets import SpectrumDatasetOnOff

__all__ = ["MapProfileEstimator", "make_orthogonal_boxes"]


class MapProfileEstimator:
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
        from gammapy.estimator import MapProfileEstimator, make_orthogonal_boxes
        from gammapy.datasets import Datasets
        from gammapy.visualization import image_profile

        # load example data
        datasets = Datasets.read("$GAMMAPY_DATA/fermi-3fhl-crab/",
            "Fermi-LAT-3FHL_datasets.yaml", "Fermi-LAT-3FHL_models.yaml")
        # configuration
        datasets[0].gti = GTI.create("0s", "1e7s", "2010-01-01")

        # creation of the boxes and axis
        start_line = SkyCoord(182.5, -5.8, unit='deg', frame='galactic')
        end_line = SkyCoord(186.5, -5.8, unit='deg', frame='galactic')
        boxes, axis = make_orthogonal_boxes(start_line,
                                        end_line,
                                        datasets[0].counts.geom.wcs,
                                        1.*u.deg,
                                        11)

        # set up profile estimator and run
        prof_maker = MapProfileEstimator(boxes, axis)
        fermi_prof = prof_maker.run(datasets[0])

        # plot directly the data from the output dictionnary
        x=fermi_prof.table["x_ref"]
        y=fermi_prof.table["excess"]
        yerr = [-fermi_prof.table["errn"], fermi_prof.table["errp"]]
        plt.errorbar(x,y,yerr=yerr, fmt='o')
        plt.yscale('log')
        plt.show()

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

    def make_prof(self, dataset, steps):
        """ Utility to make the profile in each region

        Parameters
        ----------
        dataset : MapDataset or MapDatasetOnOff
            the dataset to use for profile extraction
        steps : list of str
            the steps to be used.
        Returns
        --------
        results : list of dictionnary
            the list of results (list of keys: x_min, x_ref, x_max, alpha, counts, background, excess, ts, sqrt_ts, err,
             errn, errp, ul, exposure, solid_angle)
        """
        if steps == "all":
            steps = ["err", "ts", "errn-errp", "ul"]
        
        results = []
        for index, reg in enumerate(self.regions):
            spds = dataset.to_spectrum_dataset(reg)
            mask = spds.mask if spds.mask is not None else slice(None)
            if isinstance(spds, SpectrumDatasetOnOff):
                stats = WStatCountsStatistic(
                    spds.counts.data[mask].sum(),
                    spds.counts_off.data[mask].sum(),
                    spds.alpha.data[0,0,0] # At some point, should replace with averaging over energy
                )
            else:
                stats = CashCountsStatistic(
                    spds.counts.data[mask].sum(),
                    spds.background.data[mask].sum(),
                )
                
            result = {
                "x_min": self.axis.edges[index],
                "x_max": self.axis.edges[index+1],
                "x_ref": self.axis.center[index]
            }
            if isinstance(spds, SpectrumDatasetOnOff):
                result["alpha"] = stats.alpha
            result.update({
                "counts": stats.n_on,
                "background": stats.background,
                "excess": stats.excess,
            })

            if "ts" in steps:
                if isinstance(stats.delta_ts, list) or isinstance(stats.delta_ts, np.ndarray):
                    result["ts"] = np.asanyarray(stats.delta_ts[0],dtype=np.float32)
                else:
                    result["ts"] = np.asanyarray(stats.delta_ts, dtype=np.float32)
                if isinstance(stats.significance, list) or isinstance(stats.significance, np.ndarray):
                    result["sqrt_ts"] = np.asanyarray(stats.significance[0], dtype=np.float32)
                else:
                    result["sqrt_ts"] = np.asanyarray(stats.significance, dtype=np.float32)

            if "err" in steps:
                result["err"] = np.asanyarray(stats.error, dtype=np.float32)
            
            if "errn-errp" in steps:
                result["errn"] = stats.compute_errn(self.n_sigma)
                #ToDo: check the type of errp compared to errn (maybe not the same)
                _errp = stats.compute_errp(self.n_sigma)
                if isinstance(_errp, list) or isinstance(_errp, np.ndarray):
                    result["errp"] = np.asanyarray(_errp[0], dtype=np.float32)
                else:
                    result["errp"] = np.asanyarray(_errp, dtype=np.float32)
 
            if "ul" in steps:
                result["ul"] = stats.compute_upper_limit(self.n_sigma_ul)
 
            # What exposure to take? And waht about the mask?
            result.update(
                {"exposure": spds.aeff.data.data.max() * spds.livetime,
                 "solid_angle": spds.counts.geom.solid_angle()}
            )
            results.append(result)
        return results

    def run(self, dataset, steps="all"):
        """Make the profiles

        Parameters
        ----------
        dataset : MapDataset or MapDatasetOnOff
            the dataset to use for profile extraction
        steps : list of str
            the steps to be used.

        Returns
        --------
        imageprofile : `~gammapy.estomators.ImageProfile`
            Return an image profile class containing the result
        """
        results = self.make_prof(dataset, steps)
        table = table_from_row_data(results)
        if isinstance(self.regions[0], RectangleSkyRegion):
            table.meta["PROFILE_TYPE"] = "orthogonal"

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
    centers = 0.5*(points[:,:-1]+points[:,1:])
    coords = SkyCoord.from_pixel(centers[0], centers[1] ,wcs)
    
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


def make_orthogonal_boxes(start_pos, end_pos, wcs, fullwidth, nbins=1):
    """Utility returning an array of regions to make orthogonal projections

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
    center_pos = pix_center_pos.to_sky(wcs)

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

    for i in range(nbins):
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
