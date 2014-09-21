EVENTS_LIST=events_list.txt
EVENTS=events.fits
COUNTS=counts_cube.fits
SPACECRAFT=spacecraft.fits
LIVETIME=livetime_cube.fits
EXPOSURE=exposure_cube.fits

gtselect infile=$EVENTS_LIST outfile=$EVENTS \
         ra=266.404947172699 dec=-28.9362422432238 \
         rad=30 tmin=239587200 tmax=397353600 emin=50 \
         emax=1000000 zmax=105 \

gtbin algorithm=CCUBE evfile=$EVENTS outfile=$COUNTS \
      scfile=$SPACECRAFT nxpix=31 nypix=11 binsz=2 \
      xref=0 yref=0 axisrot=0 proj=CAR coordsys=GAL \
      ebinalg=LOG emin=50 emax=1000000 enumbins=20 \

gtltcube evfile=$EVENTS scfile=$SPACECRAFT \
         outfile=$LIVETIME dcostheta=0.1 binsz=2 \

gtexpcube2 infile=$LIVETIME cmap=$COUNTS \
           outfile=$EXPOSURE irf=P7SOURCEV6 \
