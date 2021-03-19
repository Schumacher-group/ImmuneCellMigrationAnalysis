import sys
import numpy as np
from scipy.stats import multivariate_normal
from Utilities.distributions import Uniform, Normal, Loguniform
from inference.base_inference import Inferer
from typing import Union
from Utilities.exceptions import SquareRootError
from in_silico.sources import Wound, PointWound

sys.path.append('..')

# This is the Leukocyte radius: 15µm
dr = 15


def complexes(params: np.ndarray, r: Union[np.ndarray, float], t: Union[np.array, float], wound: Wound) -> Union[
     np.ndarray, float]:
    """
    Given a set of AD parameters, this function returns the concentration of bound complexes at radial
    distance r and time t.

    Parameters
    ----------
    params  A numpy array holding at least [tau, q, D, kappa, R_0] in that order
    r       the radial distance from the origin. Can be float or array
    t       time

    Returns
    -------
    C       The concentration of complexes
    This allows the model to choose which production equation is used, and passes the argument to the
    correct concentration equation
    """
    if len(params) == 7:
        q, d, tau, r_0, kappa = params[:5]
        a = wound.concentration(params, r, t)
    else:
        m, d, r_0, kappa = params[:4]
        a = wound.concentration_delta(params, r, t)

    k = (0.25 * (kappa + r_0 + a) ** 2 - r_0 * a)

    if np.array(k < 0).any():
        raise SquareRootError('Value to be square-rooted in complex eqn is less than zero')

    return 0.5 * (kappa + r_0 + a) - k ** 0.5


def observed_bias(params: np.ndarray, r: Union[np.ndarray, float], t: Union[np.array, float], wound: Wound) -> Union[
     np.ndarray, float]:
    """
    For a set of parameters, calculate the observed bias that would occur
    at a radial distance r and time t for leukocytes of radius dr.

    Parameters
    ----------
    params      a tuple containing q, d, tau, r_0, kappa, m, b_0
    r           the spatial descriptor - distance from the wound
    t           the temporal descriptor - when is this occurring

    Returns
    -------
    The observed bias

    This allows the model to choose which production equation is used
    """
    if len(params) == 7:
        q, d, tau, r_0, kappa, m, b_0 = params
    else:
        m0, d, r_0, kappa, m, b_0 = params
    return m * (complexes(params, r - dr, t, wound) - complexes(params, r + dr, t, wound)) + b_0


class AttractantInferer(Inferer):

    def __init__(self, ob_readings: dict, wound: Wound, priors: list = None, dynamics: int = 0, t_units='minutes'):
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

        self.wound = wound
        self.dynamics = dynamics
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
        if t_units is 'seconds':
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
        if dynamics == 0:
            if priors is None:
                self.priors = [Loguniform(1, 5000),
                               Normal(800, 100),
                               Uniform(0, 60),
                               Uniform(0, 1),
                               Uniform(0, 1),
                               Uniform(0, 50),
                               Uniform(0.0, 0.02)]
            else:
                assert isinstance(priors, list)
                assert len(priors) == 7
                self.priors = priors
        elif dynamics == 1:
            self.priors = [Uniform(200, 50),
                           Normal(800, 100),
                           Uniform(0, 1),
                           Uniform(0, 1),
                           Uniform(0, 50),
                           Uniform(0.0, 0.02)]
        else:
            raise ValueError('The concentration choice should be either continuous or delta, please re-choose')

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
            return self.ob_dists.logpdf(observed_bias(params, self.r, self.t, self.wound))
        except SquareRootError:
            return -np.inf

    def log_prior(self, params: np.ndarray):
        return sum([prior.logpdf(param) for prior, param in zip(self.priors, params)])


if __name__ == '__main__':
    from Utilities.plotting import plot_AD_param_dist

    # TEST

    # here are some example observed bias readings
    ob_readings = {(25, 10): (0.1732, 0.02),
                   (50, 10): (0.1541, 0.02),
                   (75, 10): (0.1081, 0.02),
                   (100, 10): (0.0647, 0.02),
                   (125, 10): (0.0349, 0.02),
                   (150, 10): (0.0174, 0.02),
                   (25, 30): (0.1018, 0.02),
                   (50, 30): (0.1007, 0.02),
                   (75, 30): (0.0955, 0.02),
                   (100, 30): (0.082, 0.02),
                   (125, 30): (0.0659, 0.02),
                   (150, 30): (0.0500, 0.02),
                   (25, 50): (0.0077, 0.02),
                   (50, 50): (0.0141, 0.02),
                   (75, 50): (0.0196, 0.02),
                   (100, 50): (0.0238, 0.02),
                   (125, 50): (0.0263, 0.02),
                   (150, 50): (0.0271, 0.02),
                   (25, 80): (0.00309, 0.02),
                   (50, 80): (0.00509, 0.02),
                   (75, 80): (0.00693, 0.02),
                   (100, 80): (0.0085, 0.02),
                   (125, 80): (0.0098, 0.02),
                   (150, 80): (0.0107, 0.02),
                   (25, 120): (0.0018, 0.02),
                   (50, 120): (0.0026, 0.02),
                   (75, 120): (0.0034, 0.02),
                   (100, 120): (0.004, 0.02),
                   (125, 120): (0.004, 0.02),
                   (150, 120): (0.005, 0.02)}

    # make a new inferer and infer the distribution over underlying parameters
    inferer = AttractantInferer(ob_readings, PointWound())
    dist_out = inferer.multi_infer(n_walkers=5,
                                   n_steps=300000,
                                   burn_in=100000,
                                   suppress_warnings=True)

    # plot the distribution
    plot_AD_param_dist(dist_out, priors=inferer.priors)
