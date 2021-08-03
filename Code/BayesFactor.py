import numpy as np
from scipy.stats import hmean


def BayesFactor(sampler1, sampler2, n_discards):
    """
    Calculates the evidence that model 1 is more appropiate than model 2. If the Bayes Factor > 1 than there is evidence
    model 1 is better than model 2, alternatively if the Bayes factor < 1 then model 2 is better than model 1.
    Parameters
    ----------
    sampler1: posterior sample chain from model 1
    sampler2: posterior sample chain from model 2
    n_discards: Number of steps to be discarded from each chain

    Returns
    -------
    A print statement with the Bayes Factor and which model has the most evidence
    """
    HME1 = HME(sampler1, n_discards)
    HME2 = HME(sampler2, n_discards)
    BF = HME1/HME2
    if BF > 1:
        return print(f"The support for model 1 is {BF}")
    else:
        return print(f"The support for model 2 is {BF}")


def HME(sampler, n_discards):
    """
    The harmonic mean is implemented which uses the posterior parameter chain discarded to the correct length to
    calculate the harmonic mean for each sample to be use in the Bayes factor
    Parameters
    ----------
    sampler = the posterior sample chain from the emcee inference pipeline
    n_discards = Number of steps to be discarded from each chain

    Returns
    -------
    Harmonic mean of chains
    """
    log_probs = sampler.get_log_prob(flat=True, discard=n_discards) #(n_iterations,n_walkers)
    HME = hmean(np.exp(log_probs))
    return HME