
# # Attractant Dynamics Parameters

import os
import sys
sys.path.append(os.path.abspath('..'))
from inference.attractant_inference import AttractantInferer, observed_bias
from utils.distributions import WrappedNormal, Uniform, Normal, TruncatedNormal
from in_silico.sources import CellsOnWoundMargin, PointWound, CellsInsideWound
from utils.distributions import Normal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
## This takes in the Random Walker inferred parameters and outputs a dictionary with the observed bias parameters

def Bias_persistance(x,y):
    Dataset =  np.load('../data/np_array/WB WT-{}{}.npy'.format(x,y))

    W = Dataset[:,0]
    B = Dataset[:,2]
    #WMut= Dataset[1][:,0]
    #BMut = Dataset[1][:,2]

    OBWT = (W * B)
    #OBMUT = (WMut * BMut)

    return OBWT#, OBMUT



distance = [25,50,75,100,125]
time = [5,10,30,50]#,80,120]
means=[]
ob_readings = {}
for i in range(len(time)):
    for j in range(len(distance)):
        data = Bias_persistance(j,i)
        mean = np.mean(data)
        std = np.std(data)
        means.append(mean)
        ob_readings[(distance[j], time[i])] = (mean,std)





# Attractant inference
# Variables needed for Ensemble Monte Carlo
run_time = 1000
nwalkers = 100
burn_in = 250
niter = burn_in + run_time
# Variables needed for Metroplis-Hastings Monte Carlo
nsteps = 1000
burn_in = 250

wound = PointWound(position=np.array([0, 0]))
inferer = AttractantInferer(ob_readings, wound=wound, t_units='minutes')
# Emcee (Ensemble Monte Carlo)
EnsembleOut = inferer.Ensembleinfer(nwalkers,niter, ob_readings)
#Metropolis-Hastings Monte Carlo
#MHout = inferer.MHinfer(n_steps,burn_in,seed=0,suppress_warnings=True,use_tqdm=True)


# Needed to output emcee parameters
sampler = EnsembleOut[0]
samples = sampler.flatchain
import seaborn as sns
sns.distplot(samples[:,1], label = "Wild type")
plt.xlabel("Diffusion coefficient ($D$)")
plt.ylabel("Density")

# Outputs the chains for each parameters
fig, axes = plt.subplots(7, figsize=(10, 7), sharex=True)
labels = ["q", "D", "$\\tau$", "$\kappa_d$","R_0","m","b_0"]
for i in range(len(labels)):
    ax = axes[i]
    ax.plot(samples[:,i], alpha=1)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")