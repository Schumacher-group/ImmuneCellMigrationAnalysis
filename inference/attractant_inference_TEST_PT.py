import sys
import os

sys.path.append(os.path.abspath('../'))

import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import expi
from Utilities.distributions import Uniform
from inference.base_inference import Inferer
from Utilities.exceptions import SquareRootError
from typing import Union
from numbers import Number

# This is the Leukocyte radius: 15µm
dr = 15



def concentrations_production(params: np.ndarray, r: Union[np.ndarray, float], t: Union[np.array, float]) -> Union[
    np.ndarray, float]:
    """
    This function returns the concentration of attractant at a radial distance r and
    time t for a continuous point source emitting attractand at a rate q from the origin
    from t=0 to t=τ with diffusion constant D. The equation governing this is:

    A(r, t) = - q / 4πD  *  Ei(r^2 / 4Dt),                             if t < τ
    A(r, t) =   q / 4πD  *  [ Ei(r^2 / 4D(t - τ)) - Ei(r^2 / 4Dt) ],   if t > τ

    Parameters
    ----------
    params  A numpy array containing q, D and tau
    r       the radial distance from the origin. Can be float or array
    t       time: can be a float or array.

    Returns
    -------
    A       The attractant concentration

    """

    q, D, tau = params

    if not isinstance(r, (Number, np.ndarray)):
        raise TypeError('r must be either a number or a numpy array, but it is {}'.format(type(r)))

    factor = q / (4 * np.pi * D)

    if isinstance(t, Number):

        if t < tau:
            out = -expi(- r ** 2 / (4 * D * t))
        else:
            out = expi(- r ** 2 / (4 * D * (t - tau))) - expi(- r ** 2 / (4 * D * t))

    elif isinstance(t, np.ndarray):

        if isinstance(r, np.ndarray):
            assert r.shape == t.shape, 'r and t must be the same shape, but they have shapes {} and {} respectively'.format(
                r.shape, t.shape)

        out = - expi(- r ** 2 / (4 * D * t))
        out[t > tau] += expi(- r[t > tau] ** 2 / (4 * D * (t[t > tau] - tau)))

    else:
        raise TypeError('t must be either a number or a numpy array, but it is {}'.format(type(t)))

    return factor * out

class AttractantInfererProd(Inferer):

    def __init__(self, ob_readings: dict, priors: list = None, t_units='minutes'):
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
                           Uniform(1, 1000),
                           Uniform(1, 120)]
        else:
            assert isinstance(priors, list)
            assert len(priors) == 3
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
            return self.ob_dists.logpdf(concentrations_production(params, self.r, self.t))
        except SquareRootError:
            return -np.inf

    def log_prior(self, params: np.ndarray):
        return sum([prior.logpdf(param) for prior, param in zip(self.priors, params)])

