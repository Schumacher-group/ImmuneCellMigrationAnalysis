# Import all the necessary modules needed to run the inference pipeline
import sys
import os

sys.path.append(os.path.abspath('..'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns 
import emcee
from in_silico.sources import Wound,PointWound
from Utilities.distributions import Uniform, TruncatedNormal

filesuffix = "_farhalf_1x_t2"
filesuffix_tosave = filesuffix
burnin_BP= 3000
thin_BP=29

distance = [54, 105, 195, 305, 430] # Liepe bins: [54, 105, 195, 305, 430]; 2x bins: [54, 87.5, 122.5, 167.5, 222.5, 277.5, 332.5, 395]
time = [6.25, 18.75] #[2.5,7.5,12.5,17.5,22.5]
time_labels = ["0-12.5", "12.5-25"] #["0-5","5-10","10-15","15-20","20-25"]
startSpaceBin = 0 # uses python indexing!! set to 0 to use all bins, 1 to exclude first bin on account of being at the wound edge
startTimeBin = 0

print("running attractant inference for " + filesuffix + "...")
if startSpaceBin==1:
    print("excluding first bin of observed bias data for attractant inference")
    distance = distance[startSpaceBin:]
    filesuffix_tosave = filesuffix_tosave + "_exclSBin" + str(startSpaceBin)

if startTimeBin>0:
    print("starting from time bin " + str(startTimeBin) + " (" + time_labels[startTimeBin] + " mins)")
    time = time[startTimeBin:]
    time_labels = time_labels[startTimeBin:]
    filesuffix_tosave = filesuffix_tosave + "_exclTBin" + str(startTimeBin)

stepSizeTimeBins = 1
if stepSizeTimeBins==2: # because np.round will round 4.5 to 4 (to the nearest even number...)
    nTimeBins = int(np.round(len(time_labels)/stepSizeTimeBins))+1
elif stepSizeTimeBins==1:
    nTimeBins = int(np.round(len(time_labels)/stepSizeTimeBins))

ob_readings_control = {}
ob_readings_mcr = {}

# This function reads in previously run data from walker inference pipeline, and outputs the observed bias mean and std\
# for the dataframe. The inputs are x,y which correspond to the file formats
def bias_param_samples_from_file(emcee_samples):
    # Removes the burn-in, thins the posterior data and flattens the walkers into a single array
    flattened_chain = emcee_samples.get_chain(discard=burnin_BP, thin=thin_BP, flat=True)
    w = flattened_chain[:, 0]  # The weighting parameter from the walker posterior array
    # p = flattened_chain[:, 1]  # The weighting parameter from the walker posterior array
    b = flattened_chain[:, 2]  # The bias parameter from the walker posterior array
    observedBias = w * b  # The observed bias parameter from the walker posterior array
    # observedPersistence = (1 - w) * p  # The observed persistence parameter from the walker posterior array
    mean_ob = np.mean(observedBias)
    std_ob = np.std(observedBias)
    # mean_op = np.mean(observedPersistence)
    # std_op = np.std(observedPersistence)
    return mean_ob, std_ob#, mean_op, std_op

def bias_values_control(i, j): 
    control_data = emcee.backends.HDFBackend('../data/BP_inference/Single_wound_CTR_revision'+filesuffix+'_bins{}{}.h5'.format(i,j))
    return bias_param_samples_from_file(control_data)

def bias_values_mcr(i, j): 
    mcr_data = emcee.backends.HDFBackend('../data/BP_inference/Single_wound_MCR_revision'+filesuffix+'_bins{}{}.h5'.format(i,j))
    return bias_param_samples_from_file(mcr_data)

"""
Here we define a nested for loop which populates the dictionary ob_readings with the mean and std of the current 
observed bias. This is then ready to be read into our attractant inference as the data for the inference pipeline. 
"""
for i in range(nTimeBins):
    for j in range(len(distance)):
        ObBias_cont = bias_values_control(j+startSpaceBin, i*stepSizeTimeBins)
        ObBias_mcr = bias_values_mcr(j+startSpaceBin, i*stepSizeTimeBins)

        ob_mean_cont = ObBias_cont[0]
        ob_std_cont = ObBias_cont[1]
        
        ob_mean_mcr = ObBias_mcr[0]
        ob_std_mcr = ObBias_mcr[1]
        ob_readings_control[(distance[j], time[i*stepSizeTimeBins])] = (ob_mean_cont, ob_std_cont)
        ob_readings_mcr[(distance[j], time[i*stepSizeTimeBins])] = (ob_mean_mcr, ob_std_mcr)

from inference.attractant_inference import AttractantInferer 
import multiprocessing as mp
mp.set_start_method('fork', force=True)

def infer_attractant_parameters_emcee_mp(priors_for_initialisation, log_probability, saveFilename):
    # running the sampling while checking for convergence, as in https://emcee.readthedocs.io/en/stable/tutorials/monitor/
    max_n = 1000000
    checkevery_nsteps = 20000
    listparamindcs = [1,2,6] # only check autocorrelation for parameters we expect to be able to infer well

    n_walkers = 40
    # This is the initial position of the walkers
    initial = np.array([prior.mean() for prior in priors_for_initialisation])
    ndim = len(initial)
    p0 = [np.array(initial)*(1 + 1e-2 * np.random.randn(ndim)) for i in range(n_walkers)]
    
    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n) # could probably be reduced to max_n/ checkevery?

    # This will be useful to testing convergence
    old_tau = np.inf

    # Set up the backend
    # Don't forget to clear it in case the file already exists (and you want to start a new sampler)
    backend = emcee.backends.HDFBackend('../data/Attractant_inference/'+saveFilename+'.h5')
    # uncomment this next line if initializing a new sampler
    backend.reset(n_walkers, ndim)
    print("Initial size: {0}".format(backend.iteration))
    if backend.iteration==0:
        starting_from = p0
    elif backend.iteration>0:
        starting_from = backend.get_last_sample()

    with mp.Pool() as pool:
        # Initialize the sampler
        sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability, pool=pool, backend=backend)
        # Now we'll sample for up to max_n steps
        for sample in sampler.sample(starting_from, iterations=max_n, progress=True):
            # Only check convergence every checkevery_nsteps steps
            if sampler.iteration % checkevery_nsteps:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau[listparamindcs]) 
            index += 1

            # Check convergence
            converged = np.all(tau[listparamindcs] * 50 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.05)
            if converged:
                break
            old_tau = tau

    print("Final size: {0}".format(backend.iteration))

    # plot autocorrelation
    n = checkevery_nsteps * np.arange(1, index + 1)
    y = autocorr[:index]
    plt.plot(n, n / 50.0, "--k")
    plt.plot(n, y)
    plt.xlim(0, n.max())
    plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
    plt.xlabel("number of steps")
    plt.ylabel(r"mean $\hat{\tau}$")
    plt.savefig('../Notebooks/Figures/DiagnosticPlots/autocorr_'+saveFilename+'.pdf')

    # calculate burn-in and thinning
    burnin = int(2 * np.max(tau))
    thin = int(np.min(tau))

    print("burn-in: {0}".format(burnin))
    print("thin: {0}".format(thin))

    # Trace Plots
    labels = ['q','D','τ','R0','κ','m','b0']
    ndim = len(labels)

    samples = sampler.get_chain(discard=0, thin=thin, flat=False)
    print("flat chain shape: {0}".format(samples.shape))

    plt.figure(figsize=(12, 6))
    for dim in range(ndim):
        plt.subplot(ndim, 1, dim + 1)
        for walker in range(n_walkers):
            plt.plot(samples[:, walker, dim], alpha=0.5)
        plt.ylabel(labels[dim])
        plt.xlim(0, len(samples))
    plt.xlabel("Iteration (thinned by {0})".format(thin))
    plt.tight_layout()
    plt.savefig('../Notebooks/Figures/DiagnosticPlots/traceplot_'+saveFilename+'.pdf')

    # Pairwise marginal/joint posterior Plots
    samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    labels = [labels[dim] for dim in range(ndim)]
    samples_df = pd.DataFrame(data=samples, columns=labels)

    g = sns.PairGrid(samples_df, diag_sharey=False, corner=True)
    # g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot)
    g.savefig('../Notebooks/Figures/DiagnosticPlots/pairgrid_'+saveFilename+'.pdf')
    # close figure - so that the next autocorrelation plot doesn't plot in the seaborn plot
    plt.close()

# Sets the wound location
wound = PointWound(position=np.array([0, 0]))

priors_WLpost = [Uniform(0,3500), # q # Reference: Liepe, Taylor, et al. 2012 have a 1D diffusion model with amplitude parameter U[0,1000] over 1.5hrs or more, per unit length, so we estimate 1000/90mins * wound circumference (here max 100 microns diameter) to be the upper bound for q
        TruncatedNormal(200, 50), #Uniform(64,1000), #TruncatedNormal(200, 50), # D # Reference: Weavers, Liepe, et al. 2016 Fig. 3C
        TruncatedNormal(18, 3), #Uniform(0,25), # τ # Reference: Weavers, Liepe, et al. 2016 Fig. 3D
        Uniform(0, 10000), # R0 # Reference: Liepe, Taylor, et al. 2012 Table 1
        Uniform(0, 10000), # κ # Reference: Liepe, Taylor, et al. 2012 Table 1
        Uniform(0, 100), # m 
        TruncatedNormal(0.02, 0.02)] # b0 # Reference: Weavers, Liepe, et al. 2016 Fig. 3E

# now repeat for the mcr data
priors_flat = [Uniform(0,3500), # q # Reference: Liepe, Taylor, et al. 2012 have a 1D diffusion model with amplitude parameter U[0,1000] over 1.5hrs or more, per unit length, so we estimate 1000/90mins * wound circumference (here max 100 microns diameter) to be the upper bound for q
        Uniform(0,1000), # D
        Uniform(0,25), # τ
        Uniform(0, 10000), # R0 # Reference: Liepe, Taylor, et al. 2012 Table 1
        Uniform(0, 10000), # κ # Reference: Liepe, Taylor, et al. 2012 Table 1
        Uniform(0, 100), # m 
        TruncatedNormal(0.02, 0.02)] # b0 # Reference: Weavers, Liepe, et al. 2016 Fig. 3E

priors_to_use = priors_flat
filesuffix_tosave = '_flatpriors' + filesuffix_tosave
print("starting sampling for " + filesuffix_tosave + "...")
### Control condition inference

# this code could be imported instead, by importing the AttractantInferer class and creating an instance from which to run the ensemble_infer, but with that the parallalisation was not as quick
attractantInfererCtr = AttractantInferer(ob_readings_control,priors=priors_to_use, wound=wound, t_units='minutes')
# This implements the emcee library for Ensemble Monte Carlo method
def log_probabilityCtr(params: np.ndarray):
    lp = attractantInfererCtr.log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + attractantInfererCtr.log_likelihood(params)

infer_attractant_parameters_emcee_mp(priors_to_use, log_probabilityCtr, 'attractant_inference_CTR_revision'+filesuffix_tosave)

### MCR condition inference

# this code could be imported instead, by importing the AttractantInferer class and creating an instance from which to run the ensemble_infer, but with that the parallalisation was not as quick
attractantInfererMcr = AttractantInferer(ob_readings_mcr,priors=priors_to_use, wound=wound, t_units='minutes')
# This implements the emcee library for Ensemble Monte Carlo method
def log_probabilityMcr(params: np.ndarray):
    lp = attractantInfererMcr.log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + attractantInfererMcr.log_likelihood(params)

infer_attractant_parameters_emcee_mp(priors_to_use, log_probabilityMcr, 'attractant_inference_MCR_revision'+filesuffix_tosave)