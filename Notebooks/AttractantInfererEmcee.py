
# # Attractant Dynamics Parameters

import os
import sys
sys.path.append(os.path.abspath('..'))
from inference.attractant_inference import AttractantInferer
from in_silico.sources import CellsOnWoundMargin, PointWound, CellsInsideWound
import numpy as np
import matplotlib.pyplot as plt
## This takes in the Random Walker inferred parameters and outputs a dictionary with the observed bias parameters
## Variables 
# Variables needed for Ensemble Monte Carlo
niter = 7500
nwalkers = 1000

# Variables needed for Metroplis-Hastings Monte Carlo
nsteps = 500000
burn_in = 250000
n_walkers = 6

#Chose wound locations 
wound_centre = PointWound(position=np.array([0, 0]))
wound_inside = CellsInsideWound(position=np.array([0, 0]),n_cells=10,radius=15)
wound_margin = CellsOnWoundMargin(position=np.array([0, 0]),n_cells=10,radius=15)

#Load data from numpy arrays 
def Bias_persistance(x,y):
    Dataset =  np.load('../data/np_array/WB WT-{}{}.npy'.format(x,y))
    W = Dataset[:,0]
    B = Dataset[:,2]
    OBWT = (W * B)
    return OBWT



distance = [25,50,75,100,125,150,175]
time = [5,10,30,50]
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
inferer = AttractantInferer(ob_readings, wound=wound_centre, t_units='minutes')

# Emcee (Ensemble Monte Carlo)
EnsembleOut = inferer.Ensembleinfer(nwalkers,niter, ob_readings)
np.save('../data/AttractantInferenceEnsembleWT',EnsembleOut)
#Metropolis-Hastings Monte Carlo
MHOut = inferer.inferer.multi_infer(n_walkers,nsteps,burn_in,seed=0,suppress_warnings=True,use_tqdm=True)
np.save('../data/AttractantInferenceMHWT',MHOut)


# Needed to output emcee parameters
sampler = EnsembleOut[0]
samples = sampler.flatchain

# Compare parameter outputs between emcee and MH 
import seaborn as sns
sns.distplot(samples[:,2], label = "emcee")
sns.distplot(MHOut[:,2], label = "MH")
plt.xlabel("Production time ($\\tau$)")
plt.ylabel("Density")
plt.legend()


"""
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
"""