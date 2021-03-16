"""
This allows for the implementation of different heat equation solutions to be run against the data set in the inference
pipeline. The two solutions currently implemented are the delta-spike initial condition and the production time
condition. To ease in choosing which solution is to be run, an extra condition is added to the AttractantInferer
function named as dynamics. If dynamics = 0 is chosen then a continuous chemoattractant is produced within a given
time frame (t = 0 to t = 𝛕), alternatively dynamics = 1 can be chosen which will then chose the delta-spike initial
condition.
"""
import os
import sys
from inference.attractant_inference import AttractantInferer
from in_silico.sources import PointWound
import numpy as np
sys.path.append(os.path.abspath('..'))
"""
Here we initialise the number of iterations and the number of walkers we require to run the EMCEE inference package, 
some tuning of these parameters is required to optimise the rate of convergence. For the current data, a run length of
1500 and 80 walkers seems to offer the best convergence of the chains.  
"""
n_iter = 1500
n_walkers = 80
"""
The distances and times we want to run the attractant inference on are passed to as a list, and the observed readings 
are created within a dictionary.  
"""
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

"""
This declares the type of wound in which the attractant if a diffusion type is not chosen,
the assumed choice is the continuous production.
diffusion choices can be set by setting the dynamics variable =:
0 = Continuous production between t and 𝛕
1 = 𝛿 function initial condition
"""
# This runs the inference pipeline for continuous wound model

wound = PointWound(position=np.array([0, 0]))
inferer_con = AttractantInferer(ob_readings, wound=wound, t_units='minutes', dynamics=0)
Posterior_con = inferer_con.Ensembleinfer(n_walkers, n_iter)
# This saves the continuous posterior output to the data folder
Post_Save_con = np.save('..data/Delta_cont_data/cont_inference', Posterior_con)

# This runs the inference pipeline for delta spike model
inferer_del = AttractantInferer(ob_readings, wound=wound, t_units='minutes', dynamics=1)
Posterior_del = inferer_del.Ensembleinfer(n_walkers, n_iter)
# This saves the delta posterior output to the data folder
Post_Save_Del = np.save('..data/Delta_cont_data/Del_inference', Posterior_del)
