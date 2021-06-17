import sys
import os

sys.path.append(os.path.abspath('../'))

import numpy as np
from scipy.stats import multivariate_normal
from Utilities.distributions import Uniform
from inference.base_inference import Inferer
from Utilities.exceptions import SquareRootError
from in_silico.sources import Wound, PointWound

# This is the Leukocyte radius: 15µm
dr = 15


def concentrations_initial(params, r, t):
    D, C0 = params
    return C0 / np.sqrt(4 * np.pi * D * t) * np.exp(- r ** 2 / (4 * D * t))


class AttractantInferer(Inferer):

    def __init__(self, ob_readings: dict,  priors: list = None, t_units='minutes'):
        """
        Perform inference on observed bias readings to infer the posterior distribution over the
        attractant dynamics parameters {q, D, τ, R0, κ, m, b0} or {M, D, R0, κ, m, b0} depending on which
        production equation is chosen.

        A dictionary specifying the observed bias readings must be provided, along with a certain
        instantiated wound (which can be a PointWound, a CellsOnWoundMargin or CellsInsideWound) .

        The observed bias readings should be a dictionary with elements of the following form:

        {(r1, t1): (mu1, sig1), (r2, t2): (mu2, sig2) ... }

        r and t specify the spatial and temporal location where the observed bias has been measured,
        (this could be the mid-point of their respective bins), and mu and sig represent the mean and
        standard deviation of the posterior of the observed bias at this location.

        DISTANCES SHOULD BE MEASURED IN MICRONS

        time can be measured in minutes or seconds: specify this with the t_units argument.

        The parameters are measured in the following units:

        q:      Mmol / min
        D:      µm^2 / min
        τ:      min
        R0:     Mmol / µm^2
        kappa:  Mmol / µm^2
        m:      µm^2 / Mmol
        b0:     unit less


        Parameters
        ----------
        ob_readings     The observed bias readings
        wound           A Wound class, which the observed bias is assumed to be generated from
        priors          A list of distributions, one element per parameter, specifying the priors
        t_units         The units which time is measured in, in the ob_readings dictionary keys
        """

        super().__init__()

        assert t_units in ['seconds', 'minutes'], 't_units must be either "seconds" or "minutes" but it is {}'.format(
            t_units)

        # the total number of readings
        self.TS = len(ob_readings)

        # extract a list of rs, ts, mus and sigs
        self.r = np.array([r for r, t in ob_readings.keys()])
        self.t = np.array([t for r, t in ob_readings.keys()])
        mus = np.array([mu for mu, sig in ob_readings.values()])
        sigs = np.array([sig for mu, sig in ob_readings.values()])

        # convert to minutes
        if t_units == 'seconds':
            self.t /= 60

        # this is our multivariate Gaussian observed bias distribution
        self.ob_dists = multivariate_normal(mus, sigs ** 2)

        # these are the default priors
        # the priors use truncated normal distributions to ensure that non-physical values aren't produced
        """
        The choice in dynamics, either continuous = 0 or the delta spike = 1, leads to the choice and number of priors. 
        For the delta solution there is one less parameter choice. The production time (𝝉) is no longer needed and 
        the flow rate q is related with the initial mass concentration m0
        """

        if priors is None:
            self.priors = [Uniform(1, 1000),
                           Uniform(1, 1000)]
        else:
            assert isinstance(priors, list)
            assert len(priors) == 2
            self.priors = priors

    def log_likelihood(self, params: np.ndarray):
        """
        For a set of parameters, calculate the log-likelihood of these
        parameters

        Parameters
        ----------
        params      a tuple containing q, D, tau, R_0, kappa, m, b_0

        Returns
        -------
        The log likelihood
        """
        try:
            return self.ob_dists.logpdf(concentrations_initial(params, self.r, self.t))
        except SquareRootError:
            return -np.inf

    def log_prior(self, params: np.ndarray):
        return sum([prior.logpdf(param) for prior, param in zip(self.priors, params)])


("\n"
 "if __name__ == '__main__':\n"
 "    from Utilities.plotting import plot_AD_param_dist\n"
 "\n"
 "    # TEST\n"
 "\n"
 "    # here are some example observed bias readings\n"
 "    ob_readings = {(25, 10): (0.1732, 0.02),\n"
 "                   (50, 10): (0.1541, 0.02),\n"
 "                   (75, 10): (0.1081, 0.02),\n"
 "                   (100, 10): (0.0647, 0.02),\n"
 "                   (125, 10): (0.0349, 0.02),\n"
 "                   (150, 10): (0.0174, 0.02),\n"
 "                   (25, 30): (0.1018, 0.02),\n"
 "                   (50, 30): (0.1007, 0.02),\n"
 "                   (75, 30): (0.0955, 0.02),\n"
 "                   (100, 30): (0.082, 0.02),\n"
 "                   (125, 30): (0.0659, 0.02),\n"
 "                   (150, 30): (0.0500, 0.02),\n"
 "                   (25, 50): (0.0077, 0.02),\n"
 "                   (50, 50): (0.0141, 0.02),\n"
 "                   (75, 50): (0.0196, 0.02),\n"
 "                   (100, 50): (0.0238, 0.02),\n"
 "                   (125, 50): (0.0263, 0.02),\n"
 "                   (150, 50): (0.0271, 0.02),\n"
 "                   (25, 80): (0.00309, 0.02),\n"
 "                   (50, 80): (0.00509, 0.02),\n"
 "                   (75, 80): (0.00693, 0.02),\n"
 "                   (100, 80): (0.0085, 0.02),\n"
 "                   (125, 80): (0.0098, 0.02),\n"
 "                   (150, 80): (0.0107, 0.02),\n"
 "                   (25, 120): (0.0018, 0.02),\n"
 "                   (50, 120): (0.0026, 0.02),\n"
 "                   (75, 120): (0.0034, 0.02),\n"
 "                   (100, 120): (0.004, 0.02),\n"
 "                   (125, 120): (0.004, 0.02),\n"
 "                   (150, 120): (0.005, 0.02)}\n"
 "\n"
 "    # make a new inferer and infer the distribution over underlying parameters\n"
 "    inferer = AttractantInferer(ob_readings, PointWound())\n"
 "    dist_out = inferer.multi_infer(n_walkers=5,\n"
 "                                   n_steps=300000,\n"
 "                                   burn_in=100000,\n"
 "                                   suppress_warnings=True)\n"
 "\n"
 "    # plot the distribution\n"
 "    plot_AD_param_dist(dist_out, priors=inferer.priors)\n")