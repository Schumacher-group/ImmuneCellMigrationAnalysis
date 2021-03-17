"""
This script runs the attractant inference pipeline for two different Monte Carlo methods. The first
is a Metropolis-Hastings method and the second utilises the Emcee package to run the inference pipeline.
The outputs of this script are saved to the data folder and allow for the comparison between run times, and
number of walkers required to reached convergence.
"""
# Attractant Dynamics Parameters

import os
import sys
from inference.attractant_inference import AttractantInferer
from in_silico.sources import PointWound
import numpy as np

sys.path.append(os.path.abspath('..'))

# This takes in the Random Walker inferred parameters and outputs a dictionary with the observed bias parameters
# Attractant inference
# Variables needed for Ensemble Monte Carlo
n_iter = 10000
n_walkers = 100

# Variables needed for Metropolis-Hastings Monte Carlo
n_steps = 1000
burn_in = 100

distance = [25, 50, 75, 100, 125, 150, 175]
time = [5, 10, 30, 50]
ob_readings = {}

# This function reads in previously run data from walker inference pipeline, and outputs the observed bias mean and std\
# for the dataframe. The inputs are x,y which correspond to the file formats


def bias_persistence(x, y):
    input_data = np.load(f'../data/WalkerData/PosterData/WildTypeData-{x}{y}.npy', allow_pickle=True)
    sampler = input_data[0]  # Extracts the posterior chain array from the input_data array
    # Removes the burn-in, thins the posterior data and flattens the walkers into a single array
    input_data = sampler.get_chain(discard=250, thin=2, flat=True)
    w = input_data[:, 0]  # The weighting parameter from the walker posterior array
    b = input_data[:, 2]  # The bias parameter from the walker posterior array
    obwt = (w * b)  # The observed bias parameter from the walker posterior array
    mean = np.mean(obwt)
    std = np.std(obwt)
    return mean, std


"""
Here we define a nested for loop which populates the dictionary ob_readings with the mean and std of the current 
observed bias. This is then ready to be read into our attractant inference as the data for the inference pipeline. 
"""
for i in range(len(time)):
    for j in range(len(distance)):
        ObBias = bias_persistence(j, i)
        ob_mean = ObBias[0]
        ob_std = ObBias[1]
        ob_readings[(distance[j], time[i])] = (ob_mean, ob_std)


# Sets the wound location
wound = PointWound(position=np.array([0, 0]))
inferer = AttractantInferer(ob_readings, wound=wound, t_units='minutes')
# Emcee (Ensemble Monte Carlo)
Ensemble_out = inferer.Ensembleinfer(n_walkers, n_iter, ob_readings)
# This saves the Emcee output to the data folder
Post_Save_Emcee = np.save('..data/Emcee_posterior_chain', Ensemble_out)

# Metropolis-Hastings Monte Carlo
MH_out = inferer.infer(n_steps, burn_in, seed=0, suppress_warnings=True, use_tqdm=True)
# This saves the Metropolis-Hastings output to the data folder
Post_Save_MH = np.save('..data/MH_posterior_chain', MH_out)
