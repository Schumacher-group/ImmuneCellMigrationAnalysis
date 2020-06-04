
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
    Dataset = np.load('../data/np_array/WB Mutant-{}{}.npy'.format(x,y))
    W = Dataset[:,0]
    B = Dataset[:,2]
    OB = (W * B)
    mean = np.mean(OB)
    std = np.std(OB)

    return mean,std


distance = [25,50,75,100,125,150,175]
time = [5,10,30,45]

ob_readings = {}
for i in range(len(time)):
    for j in range(len(distance)):
        data = Bias_persistance(j,i)
        mean = data[0]
        std = data[1]
        ob_readings[(distance[j], time[i])] = (mean,std)

# Attractant inference
wound = PointWound(position=np.array([0, 0]))
inferer = AttractantInferer(ob_readings, wound=wound, t_units='minutes')
out1 = inferer.multi_infer(n_walkers=16,
                            n_steps=70000,
                            burn_in=30000,
                            seed=0,
                            suppress_warnings=True,
                            use_tqdm=True)
#Saves the current attractant inference numpy array for processing
np.save('../data/AttractantInferenceMutantUniformPriors',out1)
