import os
import sys
sys.path.append(os.path.abspath('..'))
from inference.attractant_inference_delta import AttractantInferer
from in_silico.sources_delta import CellsOnWoundMargin, PointWound, CellsInsideWound
import numpy as np
import matplotlib.pyplot as plt
# Attractant inference
# Variables needed for Ensemble Monte Carlo
niter = 10000
nwalkers = 100


wound = PointWound(position=np.array([0, 0]))

def Bias_persistance(x,y):
    Dataset = np.load('/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/data/WalkerData/PosterData/WildTypeData-{}{}.npy'.format(x,y),allow_pickle=True)
    sampler = Dataset[0]
    Dataset = sampler.get_chain(discard=250,thin = 2,flat=True)
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
inferer = AttractantInferer(ob_readings, wound=wound, t_units='minutes')

        # Emcee (Ensemble Monte Carlo)
EnsembleOut = inferer.Ensembleinfer(nwalkers,niter,ob_readings)

DiffProdTime = np.save('/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/data/DiffusionDelta',EnsembleOut)
