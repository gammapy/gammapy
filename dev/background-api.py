def simulate_toy_bg(alt, az, time):
    """
    Function to create dummy bg simulations for test purposes at a particular alt-az pair.
    Optional: give number of events, instead of observation time.
    TODO: is there already an established data format?

    Parameters
    ----------
    alt : `~astropy.coordinates.Angle`
        Value of the altitude angle for the simulated observations.
    az: `~astropy.coordinates.Angle`
        Value of the azimuth angle for the simulated observations.
    time : `~astropy.coordinates`
        Value of the simulated observation time.

    Returns
    -------
    image : `~astropy.io.fits.ImageHDU`
        Image filled with simulated bg events.

    """


def simulate_toy_bg(time):
    """
    Function to create dummy bg simulations for test purposes.
    Perform several simulations at different alt-az values, caling recursively simulate_toy_bg(alt, az, time).
    Optional: give number of events, instead of observation time.
    TODO: is there already an established data format?

    Parameters
    ----------
    time : `~astropy.coordinates`
        Value of the simulated observation time.

    Returns
    -------
    images : array of `~astropy.io.fits.ImageHDU`
        Array of images filled with simulated bg events. One image per alt-az pair.

    """


def group_runs(runlist):
    """
    Function that takes a list of runs and groups them in bins of alt-az. The output of the function is one file per alt-az bin with the list of the runs of the corresponding bin. It can be used for both, ON and OFF runlists.
    TODO: define an alt-az binning!!!
    TODO: how to access run properties? (i.e. run header?)!!!

    Parameters
    ----------
    runlist : `string`
        Name of file with list of runs.

    """


def stack_runs(runlist):
    """
    Function to stack the events of a runlist in a 3D cube. It takes a file with a runlist as input and stacks all the events in a `~astropy.io.fits.ImageHDU` 3D cube (x,y, energy).
    Optional: the output could be a 2D histogram (no energy axis) for the fov bg from Berge 2007.
    Optional: the output could be saved into fits files.
    This can be used to stack both the ON or the OFF events of a certain alt-az bin as in group_off_runs.
    TODO: think of coordinate system to use!! (nominal system?)!!!
    TODO: mask exclusion regions!!! (and correct obs time/exposure accordingly!!!)
    TODO: how to access events? (i.e. events/DST dataset?)!!!

    Parameters
    ----------
    runlist : `string`
        Name of file with list of runs.

    Returns
    -------
    image : `~astropy.io.fits.ImageHDU`
        Image filled with stacked events.

    """


def subtract_fov_bg(image_on, image_off):
    """
    Function to subtract the background events from the ON image using the OFF image events using the FoV method.
    Optional: the output could be saved into fits files.
    This can be used to subtract the background of a certain alt-az bin as in group_off_runs.
    TODO: need to calculate exposure ratio between ON and OFF (alpha)!!!

    Parameters
    ----------
    image_on : `~astropy.io.fits.ImageHDU`
        Histogram with ON events.
    image_off : `~astropy.io.fits.ImageHDU`
        Histogram with OFF events.

    Returns
    -------
    image_excess : `~astropy.io.fits.ImageHDU`
        Image with background subtracted.
    """


def subtract_fov_bg(runlist_on, runlist_off):
    """
    Function to subtract the background from the ON runlist using the OFF runlist using the FoV method.
    Optional: check if appropriate bg models exist on file.
    Optional: the output could be saved into fits files.
    1) Call group_runs for runlist_on and runlist_off, to split the runlists into alt-az bins.
    2) Call stack_runs for each alt-az bin for ON and OFF.
    3) Call subtract_fov_bg(image_on, image_off) for each alt-az bin.

    Parameters
    ----------
    runlist_on : `string`
        Name of file with list of ON runs.
    runlist_off : `string`
        Name of file with list of OFF runs.

    Returns
    -------
    image_excess : `~astropy.io.fits.ImageHDU`
        Image with background subtracted.
    """
