
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
    Dataset =  np.load('../data/np_array/WB Mutant-{}{}.npy'.format(x,y))

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
niter = 3000
nwalkers = 400
# Variables needed for Metroplis-Hastings Monte Carlo
nsteps = 1000
burn_in = 100

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