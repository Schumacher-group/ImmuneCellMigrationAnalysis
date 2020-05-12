
# # Attractant Dynamics Parameters

import os
import sys
sys.path.append(os.path.abspath('..'))
from inference.attractant_inference import AttractantInferer, observed_bias
from in_silico.sources import CellsOnWoundMargin, PointWound, CellsInsideWound
from utils.distributions import Normal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
## This takes in the Random Walker inferred parameters and outputs a dictionary with the observed bias parameters

def Bias_persistance(x,y):
    Dataset = np.load('../data/Control_data/WB mutant-{}{}.npy'.format(x,y))
    #Dataset =  np.load('../data/Control_data/WB control mutant-{}{}.npy'.format(x,y)) # IF you want to use the mutant control dataset uncomment
    #Dataset = np.load('../data/Control_data/WB mutant-{}{}.npy'.format(x,y))

    W = Dataset[:,0]
    B = Dataset[:,2]
    OB = (W * B)
    mean = np.mean(OB,axis=0)
    std = np.std(OB,axis=0)

    return mean,std


distance = [25,50,75,100,125,150,175,200]
time = [5,10,30,50]

ob_readings = {}
for i in [0,1,2,3]:
    for j in range(len(distance)):
        data = Bias_persistance(j,i)
        mean = data[0]
        std = data[1]
        ob_readings[(distance[j], time[i])] = (mean,std)

# Attractant inference

inferer = AttractantInferer(ob_readings, wound=wound, t_units='minutes')
out1 = inferer.multi_infer(n_walkers=5,
                            n_steps=500000,
                            burn_in=300000,
                            seed=0,
                            suppress_warnings=True,
                            use_tqdm=True)
#Saves the current attractant inference numpy array for processing
AttractantInference = np.save('.../data/AttractantInference',out1)
