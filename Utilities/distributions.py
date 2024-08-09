import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, expon, truncnorm, uniform, loguniform
from matplotlib.widgets import Slider, Button
import scipy
import time

import sys
import os

import scipy.stats
from Utilities.plotting import add_pi_ticks

sys.path.append(os.path.split(sys.path[0])[0])



"""
This file contains a variety of distribution classes. They are based roughly
on the format of scipy frozen distributions. They should all, at a minimum,
have methods 'pdf', 'logpdf' and 'sample'. Except for Bernoulli, which is
a discrete distribution with no pdf.

Generally, scipy distributions have accesss to the following methods

rvs(size)       Random variates.
pdf(x)          Probability density function.
logpdf(x)       Log of the probability density function.
cdf(x)          Cumulative distribution function.
logcdf(x)       Log of the cumulative distribution function.
sf(x)           Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).
logsf(x)        Log of the survival function.
ppf(q)          Percent point function (inverse of cdf — percentiles).
isf(q)          Inverse survival function (inverse of sf).
moment(n)       Non-central moment of order n
stats()         Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).
entropy()       (Differential) entropy of the RV.
fit(data)       Parameter estimates for generic data.
expect(func)    Expected value of a function (of one argument) with respect to the distribution.
median()        Median of the distribution.
mean()          Mean of the distribution.
var()           Variance of the distribution.
std()           Standard deviation of the distribution.
interval(alpha) Endpoints of the range that contains alpha percent of the distribution

 """


class ScipyDistribution(scipy.stats._distn_infrastructure.rv_continuous_frozen):
    """
    This is a class that some distributions can inherit from, which
    gives access to the scipy functionality. The reason for the
    inheritacne is to give them better names, and add some custom
    functionality where necessary.
    """

    def __init__(self, dist_type, *args, **kwargs):
        super().__init__(dist_type, *args, **kwargs)

        self.valid_prior = True

    def sample(self, size: Union[tuple, float]=None):
        return self.rvs(size)

    def plot(self, ax=None, N=1001, color=None):

        a, b = self.get_xlims()
        x = np.linspace(a, b, N)
        if ax is None:
            plt.plot(x, self.pdf(x), color=color)
        else:
            ax.plot(x, self.pdf(x), color=color)

    def get_xlims(self):
        raise NotImplementedError


class LogNormal(ScipyDistribution):

    def __init__(self, mu: Union[float, np.ndarray], sig: Union[float, np.ndarray]):
        """

        A log-normal distribution, where exp(X) is normally distributed.

        Parameters
        ----------
        mu      The mean of the underlying normal distribution.
        sig     The scale of the underlying normal distribution.
        """

        super().__init__(dist_type=lognorm, s=sig, scale=np.exp(mu))

        self.sig = sig
        self.mu = mu


class Normal(ScipyDistribution):

    def __init__(self, mu: Union[float, np.ndarray], sig: Union[float, np.ndarray]):
        """

        A normal distribution.

        Parameters
        ----------
        mu      The mean of the underlying normal distribution.
        sig     The scale of the underlying normal distribution.
        """

        super().__init__(dist_type=norm, loc=mu, scale=sig)

        self.sig = sig
        self.mu = mu

    def get_xlims(self):
        return self.mu - 4 * self.sig, self.mu + 4 * self.sig


class Exponential(ScipyDistribution):

    def __init__(self, lamda: Union[float, np.ndarray]):
        """

        An exponential distribution.

        Parameters
        ----------
        lamda    the lambda parameter, such that the p(x) = lambda * exp(- x * lambda)
        """
        super().__init__(dist_type=expon, scale=1/lamda)
        self.lamda = lamda

    def get_xlims(self):
        return 0, 5 / self.lamda


class Bernoulli:

    def __init__(self, mu: float):
        self.mu = mu
        self.valid_prior = False

    def sample(self, size: Union[tuple, float]=None):
        return np.random.uniform(0, 1, size) < self.mu


class WrappedNormal:

    def __init__(self, mu: Union[float, np.ndarray], sig: float):

        self.mu = mu
        self.sig = sig
        self.valid_prior = True

    def pdf(self, x: Union[float, np.ndarray]):

        # if sigma is greater than 4, the difference between it and a uniform distribution is ~1e-8
        if self.sig > 4:
            if isinstance(x, np.ndarray):
                return np.ones_like(x) / (2 * np.pi)
            else:
                return 1 / (2 * np.pi)
        if self.sig == 0:
            return 0.0

        # *blindly* opt for 5 loops either side

        # mu =  [ mu1  ,  mu2  ,  mu3  ,  mu4  ]

        #          .        .        .        .
        #          .        .        .        .
        # X =   [a - 2π, b - 2π, c - 2π, d - 2π]
        #       [  a   ,   b   ,   c   ,   d   ]
        #       [a + 2π, b + 2π, c + 2π, d + 2π]
        #       [a + 4π, b + 4π, c + 4π, d + 4π]
        #          .        .        .        .
        #          .        .        .        .

        # then sum normal(X) vertically

        X = np.array([x + 2 * np.pi * i for i in range(-8, 9)])
        return self.normal_pdf(X).sum(0)

    def normal_pdf(self, x: Union[float, np.ndarray]):
        return (1 / (self.sig * (2 * np.pi) ** 0.5)) * np.exp(- (x - self.mu) ** 2 / (2 * self.sig ** 2))

    def sample(self, size: Union[tuple, float]=None):
        return np.mod(np.random.normal(np.pi + self.mu, self.sig, size), 2 * np.pi) - np.pi

    def logpdf(self, x: Union[float, np.ndarray]):
        return np.log(self.pdf(x))

    def get_xlims(self):
        return -np.pi, np.pi

    def get_mean_lims(self):
        return -np.pi, np.pi

    def get_sig_lims(self):
        return 0.01, 6


class TruncatedNormal(ScipyDistribution):

    def __init__(self, mu: float, sig: float):
        """

        A truncated normal distribution, allowing only positive
        values.

        Parameters
        ----------
        sig    the scale of the normal distribution, which is then truncated.
        """
        a,b = (0-mu)/sig,(np.inf-mu)/sig
        super().__init__(dist_type=truncnorm, a=a, b=b, loc=mu, scale=sig)
        self.sig = sig
        self.mu  = mu

    def get_xlims(self):
        return 0, 10 * self.sig
    
    # this following definition should now be correctly inherited from the scipy distribution again
    # def logpdf(self, x: Union[float, np.ndarray]):
    #     a,b = (0-self.mu)/self.sig,(np.inf-self.mu)/self.sig
    #     return scipy.stats.truncnorm.logpdf(x, a=a, b=b, loc=self.mu, scale=self.sig)
    

class Uniform(ScipyDistribution):

    def __init__(self, a: float, b: float):
        """

        A truncated normal distribution, allowing only positive
        values.

        Parameters
        ----------
        sig    the scale of the normal distribution, which is then truncated.
        """
        super().__init__(dist_type=uniform, loc = a, scale = b - a) # From scipy docs: Using the parameters loc and scale, one obtains the uniform distribution on [loc, loc + scale].
        self.a = a
        self.b = b

    def get_xlims(self):
        return self.a, self.b
    
    # this following definition should now be correctly inherited from the scipy distribution again
    # def logpdf(self, x: Union[float, np.ndarray]):
    #     return scipy.stats.uniform.logpdf(x, loc = self.a, scale = self.b - self.a)


class Loguniform(ScipyDistribution):

    def __init__(self,a: Union[float, np.ndarray], b: Union[float, np.ndarray]):
        """

        A log-uniform distribution, which allows sampling from different orders of
        magnitude

        Parameters
        ----------
        a    the minimum of the loguniform distribution
        b    the  maximum of the loguniform distribution
        """
        super().__init__(dist_type=loguniform, a = a, b = b)
        self.a = a
        self.b = b

    def get_xlims(self):
        return self.a, self.b
