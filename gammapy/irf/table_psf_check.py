from collections import OrderedDict


class TablePSFChecker(object):
    """Automated quality checks for Table PSF.
    
    At the moment used for HESS HAP HD.
    """
    def __init__(self, psf, config):
        """Constructor

        Parameters
        ----------
        psf : `~gammapy.irf.PSF3D`
            Table PSF to check
        config : `~OrderedDict`
            Dictionary with configuration parameters:
                d_norm:
                    maximum norm deviation from 1
                containment_fraction:
                    containment fraction to check
                d_rel_containment:
                    maximum relative difference of containment
                    radius between neighboring bins
        """
        self.psf = psf
        self.config = config
        
    
    def check_all(self):

        # init dict for all results
        dict = OrderedDict()

        # run checks
        self.check_nan(dict)
        self.check_normalise(dict)
        self.check_containment(dict)
        
        return dict
    
    def check_nan(self, dict):
        import numpy as np
        import math

        # genarate array for easier handling
        values = np.swapaxes(self.psf.psf_value, 0, 2)
        fail_count = 0

        # loop over energies
        for i, arr in enumerate(values):
            energy_hi = self.psf.energy_hi[i]
            energy_lo = self.psf.energy_lo[i]

            # check if bin is outside of safe energy threshold
            if self.psf.energy_thresh_lo > energy_hi:
                continue
            if self.psf.energy_thresh_hi < energy_lo:
                continue
            
            # loop over offsets
            for arr2 in arr:

                # loop over deltas
                for v in arr2:

                    # check for nan
                    if math.isnan(v.value):

                        # add to fail counter
                        fail_count += 1
                        break
                    
        # write results to dict
        check_dict = OrderedDict()
        if fail_count == 0:
            check_dict['status'] = 'ok'
        else:
            check_dict['status'] = 'failed'
            check_dict['n_failed_bins'] = fail_count
        dict['nan'] = check_dict
        return
    
    def check_normalise(self, dict):
        import numpy as np

        # generate array for easier handling
        values = np.swapaxes(self.psf.psf_value, 0, 2)

        # init fail count
        fail_count = 0
        
        # loop over energies
        for i, arr in enumerate(values):
            energy_hi = self.psf.energy_hi[i]
            energy_lo = self.psf.energy_lo[i]
            
            # check if energy is outside of safe energy threshold
            if self.psf.energy_thresh_lo > energy_hi:
                continue
            if self.psf.energy_thresh_hi < energy_lo:
                continue
            
            # loop over offsets
            for arr2 in arr:
                
                # init integral
                sum = 0

                # loop over deltas
                for j, v in enumerate(arr2):

                    # calculate contribution to integral
                    width = self.psf.rad_hi[j].rad - self.psf.rad_lo[j].rad
                    rad = 0.5 * (self.psf.rad_hi[j].rad + self.psf.rad_lo[j].rad)
                    sum += v.value * width * rad * 2 * np.pi

                # check if integral is close enough to 1
                if (np.abs(sum - 1.0) > self.config['d_norm']):

                    # add to fail counter
                    fail_count += 1

        # write results to dict
        check_dict = OrderedDict()
        if fail_count == 0:
            check_dict['status'] = 'ok'
        else:
            check_dict['status'] = 'failed'
            check_dict['n_failed_bins'] = fail_count
        dict['normalise'] = check_dict
        return

    def check_containment(self, dict):
        import numpy as np
        import math

        # set fraction to check for
        fraction = self.config['containment_fraction']

        # set maximum relative difference between neighboring bins
        rel_diff = self.config['d_rel_containment']

        # generate array for easier handling
        values = np.swapaxes(self.psf.psf_value, 0, 2)

        # init containment radius array
        radii = np.zeros(values[:, :, 0].shape)

        # init fail count
        fail_count = 0

        # loop over energies
        for i, arr in enumerate(values):
            energy_hi = self.psf.energy_hi[i]
            energy_lo = self.psf.energy_lo[i]
            
            # loop over offsets
            for k, arr2 in enumerate(arr):

                # if energy is outside safe energy threshold,
                # set containment radius to None
                if self.psf.energy_thresh_lo > energy_hi:
                    radii[i, k] = None
                    continue
                if self.psf.energy_thresh_hi < energy_lo:
                    radii[i, k] = None
                    continue

                # init integral and containment radius
                sum = 0
                r = None
                
                # loop over deltas
                for j, v in enumerate(arr2):
                    
                    # calculate contribution to integral
                    width = self.psf.rad_hi[j].rad - self.psf.rad_lo[j].rad
                    rad = 0.5 * (self.psf.rad_hi[j].rad + self.psf.rad_lo[j].rad)
                    sum += v.value * width * rad * 2 * np.pi
                    
                    # check if conainmant radius is reached
                    if (sum >= fraction):

                        # convert radius to degrees
                        r = rad*180./np.pi
                        break

                # store containment radius in array
                radii[i, k] = r

        # generate an array of radii with stripped edges so that each
        # element has 9 neighbors
        inner = radii[1:-1, 1:-1]

        # loop over energies
        for i, arr in enumerate(inner):
            
            # loop over offsets
            for j, v in enumerate(inner[i]):

                # check if radius is nan 
                if math.isnan(v):
                    continue

                # set distance to neighbors
                d = 1

                # calculate corresponding indices in whole radii array
                ii = i + 1
                jj = j + 1

                # retrieve array of neighbors
                nb = radii[ii-d:ii+d+1, jj-d:jj+d+1].flatten()

                # loop over neighbors
                for n in nb:

                    # check if neighbor is nan
                    if math.isnan(n):
                        continue

                    # calculate relative difference to neighbor
                    diff = np.abs(v - n) / v
                    
                    # check if difference is to big
                    if diff > rel_diff:
                        
                        # add to fail counter
                        fail_count += 1
                    
        # write results to dict
        check_dict = OrderedDict()
        if fail_count == 0:
            check_dict['status'] = 'ok'
        else:
            check_dict['status'] = 'failed'
            check_dict['n_failed_bins'] = fail_count
        dict['containment'] = check_dict
        return

def check_all_table_psf(data_store):

    # get obs ids
    obs_ids = data_store.obs_table['OBS_ID'].data
    
    # loop over observations
    for obs_id in obs_ids[:10]:

        # get observation
        obs = data_store.obs(obs_id=obs_id)

        # get table psf
        psf = obs.load(hdu_class='psf_table')

        # init config
        config = OrderedDict(
            d_norm = 0.01,
            containment_fraction = 0.68,
            d_rel_containment = 0.7
            )

        # do the checks
        results = TablePSFChecker(psf=psf, config=config).check_all()
        print(results)



if __name__ == '__main__':
    import sys
    from gammapy.data import DataStore
    args = sys.argv

    # get datastore
    data_store = DataStore.from_dir(args[1])

    # check all table psfs
    check_all_table_psf(data_store)
    
