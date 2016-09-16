"""Starting with only gamma-ray event lists, make images and a source catalog.

This is an example showcasing some of the Gammapy features.

With just ~ 100 lines of high-level code we can do this:

- Make a global event list from a datastore (uses `gammapy.data`)
- Apply an energy and spatial event selection (`EventList` class)
- Bin events into a counts image (uses `gammapy.data` and `gammapy.image`)
- Estimate a background image (uses `gammapy.background`)
- Compute significance = sqrt(TS) image (uses `gammapy.detect`)
- Create a source catalog via a peak finder (uses `gammapy.detect`)
- Make a pretty picture of the images and circle detected sources (uses `gammapy.image`)

You can use this script to run certain steps by commenting in or out the functions in main().

To look at the output significance image and source catalog you can use ds9:

    $ ds9 -cmap bb -scale sqrt significance.fits -region sources.reg

TODO:
- Use adaptive ring (largest sources are cut out)
- The peak finder gives weird results in some regions I don't understand
  ... need to have a look.

"""
import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.units import Quantity
from astropy.coordinates import Angle, SkyCoord
from astropy.convolution import Gaussian2DKernel
from photutils.detection import find_peaks
from gammapy.data import DataStore, EventListDataset, EventList
from gammapy.detect import KernelBackgroundEstimator, KernelBackgroundEstimatorData
from gammapy.image import binary_disk, binary_ring
from gammapy.detect import compute_ts_image
from gammapy.catalog import to_ds9_region, coordinate_iau_format

HESSFITS_MPP = 'pa/Model_Deconvoluted_Prod26/Mpp_Std/'
REF_IMAGE = 'significance_reference.fits.gz'
TOTAL_EVENTS_FILE = os.path.join(HESSFITS_MPP, 'all_events.fits')
COUNTS_IMAGE = 'counts.fits'
MASK_IMAGE = 'mask.fits'
BACKGROUND_IMAGE = 'background.fits'
TS_IMAGES = 'ts.fits'
SIGNIFICANCE_IMAGE = 'significance.fits'
SOURCE_CATALOG = 'sources.fits'
SOURCE_REGIONS = 'sources.reg'


def make_total_event_list():
    """Make total event list.

    TODO: move this function to the datastore class
    and the sky box selection to the `make_counts_image` function.
    """
    data_store = DataStore(dir=HESSFITS_MPP)

    observation_selection = dict(type='sky_box', frame='galactic',
                                 lon=Quantity([-120, 70], 'deg'),
                                 lat=Quantity([-5, 5], 'deg'), border=Quantity(2, 'deg')    )
    observation_table = data_store.make_observation_table(observation_selection)

    # For testing, only process a small subset of observations
    # observation_table = observation_table.select_linspace_subset(num=1000)

    event_list_files = data_store.make_table_of_files(observation_table, filetypes=['events'])

    ds = EventListDataset.vstack_from_files(event_list_files['filename'])
    print('Total number of events: {}'.format(len(ds.event_list)))
    print('Total number of GTIs: {}'.format(len(ds.good_time_intervals)))

    print('Converting EventListDataset to HDUList ...')
    hdu_list = ds.to_fits()

    print('Writing {}'.format(TOTAL_EVENTS_FILE))
    hdu_list.writeto(TOTAL_EVENTS_FILE, clobber=True)


def make_counts_image(energy_band):
    """Apply event selections and bin event positions into a counts image."""
    event_list = EventList.read(TOTAL_EVENTS_FILE)
    n_events = len(event_list)
    print('Number of events: {}'.format(n_events))
    print('Applying energy band selection: {}'.format(energy_band))
    event_list = event_list.select_energy(energy_band)
    n_events_selected = len(event_list)
    fraction = 100 * n_events_selected / n_events
    print('Number of events: {}. Fraction: {:.1f}%'.format(n_events_selected, fraction))

    print('Filling counts image ...')
    header = fits.getheader(REF_IMAGE)
    image = event_list.fill_counts_header(header)

    print('Writing {}'.format(COUNTS_IMAGE))
    image.writeto(COUNTS_IMAGE, clobber=True)


def make_background_image():
    """Estimate background image.

    See the `KernelBackgroundEstimator` tutorial and
    documentation how it works, or the SciNeGHe 2014 proceeding
    by Ellis Owen et al.
    """
    radius = Angle(0.2, 'deg')
    r_in = Angle(0.3, 'deg')
    r_out = Angle(0.7, 'deg')
    significance_threshold = 5
    mask_dilation_radius = Angle(0.1, 'deg')
    max_iterations = 3

    hdu = fits.open(COUNTS_IMAGE)['COUNTS']
    binsz = hdu.header['CDELT2']
    images = KernelBackgroundEstimatorData(counts=hdu.data, header=hdu.header)

    # TODO: we should have utility functions to initialise
    # kernels with angles so that we don't have to convert to pix here.
    source_kernel = binary_disk(radius=radius.deg/binsz)
    background_kernel = binary_ring(r_in=r_in.deg/binsz,
                                    r_out=r_out.deg/binsz)

    estimator = KernelBackgroundEstimator(
        images=images, source_kernel=source_kernel, background_kernel=background_kernel,
        significance_threshold=significance_threshold,
        mask_dilation_radius=mask_dilation_radius.deg/binsz,
    )
    print('Running background estimation ...')
    estimator.run(max_iterations=max_iterations)

    print('Writing {}'.format(MASK_IMAGE))
    estimator.mask_image_hdu.writeto(MASK_IMAGE, clobber=True)

    print('Writing {}'.format(BACKGROUND_IMAGE))
    estimator.background_image_hdu.writeto(BACKGROUND_IMAGE, clobber=True)


def make_significance_image():
    """Make significance = sqrt(TS) image using a Gaussian kernel.
    """
    gauss_kernel_sigma = 5  # pix
    header = fits.getheader(REF_IMAGE)
    counts = fits.getdata(COUNTS_IMAGE)
    background = fits.getdata(BACKGROUND_IMAGE)
    exposure = 1e11 * np.ones_like(counts, dtype='float32')
    kernel = Gaussian2DKernel(gauss_kernel_sigma)

    print('Computing TS image ...')
    result = compute_ts_image(counts, background, exposure, kernel)
    print('TS image computation took: {}'.format(result.runtime))

    print('Writing {}'.format(TS_IMAGES))
    result.write(TS_IMAGES, header=header, overwrite=True)


def make_source_catalog():
    """Make source catalog from images.

    TODO: use other images to do measurements for the sources,
    e.g. excess, npred, flux.
    """
    significance_threshold = 7

    hdu = fits.open(TS_IMAGES)['sqrt_ts']
    header = fits.getheader(REF_IMAGE)
    wcs = WCS(header)

    print('Running find_peaks ...')
    table = find_peaks(data=hdu.data, threshold=significance_threshold, wcs=wcs)
    print('Number of sources detected: {}'.format(len(table)))

    # Add some useful columns
    icrs = SkyCoord(table['icrs_ra_peak'], table['icrs_dec_peak'], unit='deg')
    galactic = icrs.galactic
    table['Source_Name'] = coordinate_iau_format(icrs, ra_digits=5, prefix='J')
    table['GLON'] = galactic.l.deg
    table['GLAT'] = galactic.b.deg

    # table.show_in_browser(jsviewer=True)
    print('Writing {}'.format(SOURCE_CATALOG))
    table.write(SOURCE_CATALOG, overwrite=True)

    ds9_string = to_ds9_region(table, label='Source_Name')
    print('Writing {}'.format(SOURCE_REGIONS))
    with open(SOURCE_REGIONS, 'w') as fh:
        fh.write(ds9_string)

    # TODO: move this to `make_significance_image`.
    # At the moment the TS image output images don't have WCS info in the header
    hdu = fits.PrimaryHDU(data=hdu.data, header=header)
    print('Writing {}'.format(SIGNIFICANCE_IMAGE))
    hdu.writeto(SIGNIFICANCE_IMAGE, clobber=True)


def main():
    """Run the whole analysis chain.

    Here the steps communicate via FITS files.
    """
    energy_band = Quantity([1, 10], 'TeV')

    #make_total_event_list()
    #make_counts_image(energy_band)
    #make_background_image()
    make_significance_image()
    #make_source_catalog()


if __name__ == '__main__':
    main()

