# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import sherpa.astro.ui as sau
from ..stats import significance


class SpecModel(object):
    """Class to generate model arrays relevant for the plotting part."""

    def __init__(self, data_list, components):  # expecting list of HESS_spec objects and
        self.components = components  # list of model components as input
        self.totmodel = components[-1]
        if data_list is not None:
            self.get_plot_arrays(data_list)

        self.draw_plot()  # self.bcenter,self.mean_flux,self.mean_err,self.resid,components)

    def get_plot_arrays(self, data_list):
        """Construct arrays of model count rates."""

        sample_model = sau.get_model(data_list[0].name)
        self.get_binning(sample_model)  # do this only once assuming that true energy
        # binning does not change from run to run
        obs_exc = np.zeros_like(self.bcenter)
        obs_err = np.zeros_like(self.bcenter)
        tot_on = np.zeros_like(self.bcenter)
        tot_off = np.zeros_like(self.bcenter)
        mod_cnts = np.zeros_like(self.bcenter)
        exp_tot = np.zeros_like(self.etrue_center)
        mod_tot = np.zeros_like(self.etrue_center)

        for dat in data_list:
            datid = dat.name
            exposure = dat.data.exposure
            on_cnt_rate = dat.data.get_y()

            c_bkg = sau.get_bkg(datid)
            bg_cnt_rate = c_bkg.get_y()
            backscal = c_bkg.get_backscal()

            c_mod = sau.get_model(datid)
            arf = c_mod.arf
            arf_vals = arf.get_y()

            # Excess
            bw_expo = self.b_width * exposure
            on_cnts = on_cnt_rate * bw_expo
            off_cnts = bg_cnt_rate * bw_expo / backscal
            c_exc = on_cnts - off_cnts  # excess counts
            c_exc_err2 = on_cnts + off_cnts / backscal  # errors

            # model counts
            c_modcnts = c_mod.calc(self.para, 2.)  # second parameter is dummy...

            # Consider only noticed bins
            valid = dat.data.get_noticed_channels().astype(int)
            valid -= np.ones_like(valid)  # Channel id's start at 1!

            obs_exc[valid] = obs_exc[valid] + c_exc[valid]  # Total excess in noticed bins
            obs_err[valid] = obs_err[valid] + c_exc_err2[valid]  # Total error square
            tot_on[valid] = tot_on[valid] + on_cnts[valid]
            tot_off[valid] = tot_off[valid] + off_cnts[valid]
            mod_cnts[valid] = mod_cnts[valid] + c_modcnts[valid]  # Total noticed model counts
            valid_arf = self.ener_map[valid].sum(0) > 0  # valid pixels in true energy

            self.get_mod_val(self.totmodel, self.etrue_center)

            # Add run exposure*area*model for valid true energy bins only
            exp_tot[valid_arf] = exp_tot[valid_arf] + \
                                 arf_vals[valid_arf] * self.mod_val[valid_arf] * exposure

            ''' Not used, may be useful to produce upper limits
            #significance per bin:
            signis = significance(n_observed=tot_on, mu_background=tot_off, method='lima')
            some_significant = False
            #makeUL = []
            for i,signi in enumerate(signis):
            if signi<2:
            print('WARNING: Energy bin from', round(binmin[i]/1e9,1), 'to', \
            round(binmax[i]/1e9,1), 'TeV has', round(signi,2), 'sigma only.')
            print('...may want to convert to upper limit') # NOT YET IMPLEMENTED
            continue
            #makeUL.append(True)
            if np.isinf(signi) or np.isnan(signi): #isinf when Non = Noff = 0?
            if some_significant: # otherwise we are probably below threshold
            print('WARNING: Energy bin from', round(binmin[i]/1e9,1), 'to', \
            round(binmax[i]/1e9,1), 'TeV contains no events.')
            continue
            else:
            some_significant = True
            '''

        # compute average exposure (time*area) in each measured energy bin
        mean_expo = np.zeros(obs_exc.shape)
        for i in range(obs_exc.shape[0]):
            mean_expo[i] = exp_tot[self.ener_map[i, :]].sum() / \
                           self.mod_val[self.ener_map[i, :]].sum()
            bw_meanexpo = self.b_width * mean_expo

        # get flux and error per cm^2/s/TeV
        self.mean_flux = 1e9 * obs_exc / bw_meanexpo
        self.mean_flux[np.isnan(self.mean_flux)] = 0

        self.mean_err = 1e9 * np.sqrt(obs_err) / bw_meanexpo  # mean_flux/signis

        # Compute residuals where model counts >0
        self.resid = (-mod_cnts + obs_exc) / np.sqrt(obs_err)

        # Model spectral points
        self.bcenter /= 1e9  # keV? Nope, real high energy...
        # self.mean_flux *= 1e9
        # self.mean_err  *= 1e9

    def get_binning(self, c_mod):
        """Constructs energy arrays from a sample datid
        and finds fitted model parameters (which are also constant over the plotting part).
        """
        # datid = dat.name
        # rmf     = get_model(datid).rmf
        rmf = c_mod.rmf
        self.binmax = rmf.e_max
        self.binmin = rmf.e_min
        self.b_width = self.binmax - self.binmin
        self.bcenter = np.sqrt(self.binmax * self.binmin)  # 0.5*(binmax + binmin)

        arf = c_mod.arf
        etre_lo, emes_min = np.meshgrid(arf.energ_lo, self.binmin)
        etre_hi, emes_max = np.meshgrid(arf.energ_hi, self.binmax)
        self.ener_map = (etre_lo > emes_min) * (etre_hi <= emes_max)
        self.etrue_center = np.sqrt(c_mod.elo * c_mod.ehi)

        model_pars = c_mod.pars
        npars = len(model_pars)
        self.para = [model_pars[i].val for i in range(npars)]

    def get_mod_val(self, model, energies):
        self.mod_val = model.calc(self.para, energies)
        '''
        # THESE SAMPLING METHODS SAMPLE SOME PARAMETERS OUTSIDE THEIR RANGE!!
        # fit statistic provided is the default one, not wstat!!
        n_alt     = 1000
        #alt_pars  = t_sample(num=n_alt) # generate alternative values for parameters
        alt_pars  = normal_sample(num=n_alt)  # fit_statistic, par[0], par[1]...
        # will not work for logparabola unless correlate=Flase, make own logparabola?
        n_free    = len(alt_pars[0])-1  # value 0 is fit statistic
        alt_models = []
        for k in range(n_alt): #need to do this loop to include the frozen parameters
        alt_par_set = []
        past_fixed = 0
        for i in range(npars):
        if model_pars[i].frozen:
        alt_par_set.append(para[i])
        past_fixed += 1
        else:
        alt_par_set.append(alt_pars[k][i+1-past_fixed])
        alt_models.append(alt_par_set)
        '''

    def make_model_vals(self, xval, model):
        yval = model(xval * 1e9) * 1e9
        return yval

    def draw_plot(self):  # ,bcenter,mean_flux,mean_err,resid,components):
        resid_err = np.ones_like(self.resid)  # since we use residuals in units sigma

        Fmin = 0.3 * min(self.mean_flux[self.mean_flux > 0])
        Fmax = 2 * max(self.mean_flux)

        Emin = min(self.binmin[self.mean_flux > 0]) / 1e9
        Emax = max(self.binmax[self.mean_flux > 0]) / 1e9
        logEmin = np.log10(Emin)
        logEmax = np.log10(Emax)

        xval = np.logspace(logEmin, logEmax)
        yvals = []  # make a loop over model components to draw a set of yvals
        for sub_comp in self.components:
            yvals.append(self.make_model_vals(xval, sub_comp))

        import matplotlib.pyplot as plt  # will work in python, if installed
        import matplotlib.gridspec as gridspec

        """ Make the spectral plots using matplotlib
        """
        # Figure definitions
        fig = plt.figure()
        gs = gridspec.GridSpec(4, 1)

        # Spectrum #
        ax1 = fig.add_subplot(gs[:3, :])  # rows, cols, plot_num. This one is for the spectrum
        ax1.set_xscale("log", nonposx='clip')
        ax1.set_yscale("log", nonposy='clip')
        # ax1.set_xlim(Emin, Emax) #set limits for ax2
        ax1.set_ylim(Fmin, Fmax)
        ax1.set_xlabel('E, TeV')
        ax1.set_ylabel('dN/dE')  # , TeV^{-1}cm^{-2}s^{-1}') # WRONG FORMAT
        ax1.get_xaxis().set_visible(False)

        '''  # Model range: NOT MEANINGFUL AS LONG AS NOT USED WITH THE RIGHT FIT STATISTIC
        for alt_par_set in alt_models:
        alt_val = model.calc(alt_par_set,xval*1e9)*1e9
        ax1.plot(xval,alt_val,color=(1,0.9,0.9),linewidth=1)
        '''

        ax1.plot(xval, yvals[-1], color='r', linewidth=2)  # total model
        if len(yvals) > 2:  # there are multiple components
            style_str = ['--', '-.', ':']  # only three different styles available :(
            i = 0
            for comp in yvals[:-1]:
                ax1.plot(xval, comp, style_str[i])
                i += 1

        # spectral points
        ax1.errorbar(self.bcenter, self.mean_flux, yerr=self.mean_err, fmt='o', color='k')

        # residuals plot
        resmax = 3  # hardcoded?
        zeroline = np.zeros_like(xval)

        ax2 = fig.add_subplot(gs[3, :], sharex=ax1)
        ax2.plot(xval, zeroline, color='k')
        ax2.errorbar(self.bcenter, self.resid, yerr=resid_err, fmt='o', color='k')
        ax2.set_ylim(-resmax, resmax)
        ax2.set_xlim(Emin, Emax)
        ax2.set_xlabel('E, TeV')
        ax2.set_ylabel('sigma')

        plt.subplots_adjust(hspace=0.1)
        plt.show()
