# Simple test case: Gauss on flat background

The purpose of this test case is to cross-check
* optimal parameter estimation
* HESSE error computation
* Correlation coefficient computation
* TS computation
* expected counts image computation

## Data

The test data counts.fits.gz was generated with Sherpa with simulate_data.py .

It consists of a Gauss on flat background with these "true" parameters:

* Image size 200 x 200
* Gauss position: (100, 100)
* Gauss sigma: 5
* Gauss norm: 1e3 counts
* Background level: 1 count / bin

Note that by `norm` we mean the integral over the 2D Gauss, which is related
to the `amplitude` (i.e. the value at the center) via

    norm = amplitude * 2 * pi * sigma ** 2

## Fit procedure

* Likelihood fit
* Integrate model over bins
* Errors in normal approximation ("HESSE errors")

## Fit results

* `norm_sigma_corr` is the correlation between the `norm` and `sigma` parameters,
which can be obtained from the covariance matrix.
* To quantify if the expected counts (a.k.a. "model excess") images are correct,
please compute the image

    exp_counts_diff = abs(exp_counts - exp_counts_ref)

and then the numbers

    exp_counts_diff_max = max(exp_counts_diff)
    exp_counts_diff_sum = sum(exp_counts_diff)

### Sherpa

* xpos = 99.3065 +- 0.281104
* ypos = 99.9734 +- 0.268053
* sigma = 4.85304 +- 0.151214
* norm = 986.474 +- 43.8434
* background = 0.997473 +- 0.00497959
* norm_sigma_corr = 0.43339655
* TS = 2653.55288
* exp_counts_diff_max and exp_counts_diff_sum are not given since this is the reference result.

### gammalib

* xpos = ? +- ?
* ypos = ? +- ?
* sigma = ? +- ?
* norm = ? +- ?
* background = ? +- ?
* norm_sigma_corr = ?
* TS = ?
* exp_counts_diff_max = ?
* exp_counts_diff_sum = ?
