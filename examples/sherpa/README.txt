Notes
-----

If the fit statistic is summed for many (1e6 or 1e9 or even more) bins or events, 

If you download the sherpa source code, you can find the computation in the
$ASCDS_INSTALL/src/pkg/sherpa/sherpa/sherpa/include/sherpa/stats.hh
file.
Sherpa uses the Kahan sum for these computation (see eg http://en.wikipedia.org/wiki/Kahan_summation_algorithm )

https://github.com/numpy/numpy/issues/2448

Sherpa: http://cxc.cfa.harvard.edu/sherpa/

TODO:
- add unit tests
- add documentation
- integrate in astropy or gammapy
