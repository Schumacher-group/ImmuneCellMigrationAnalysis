import os, sys
import numpy as np
sys.path.append(os.path.abspath('..'))
from inference.attractant_inference import AttractantInferer, observed_bias
from in_silico.sources import PointWound
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

niter = 1500
nwalkers = 80



def Bias_persistance(x,y):
    Dataset = np.load('/Users/danieltudor/Documents/Wood group/ImmuneCellMigrationAnalysis/data/WalkerData/PosterData/WildTypeData-{}{}.npy'.format(x,y),allow_pickle=True)
    sampler = Dataset[0]
    Dataset = sampler.get_chain(discard=250,thin = 2,flat=True)
    W = Dataset[:,0]
    B = Dataset[:,2]
    OBWT = (W * B)
    return OBWT



distance = [25,50,75,100,125,150,175]
time = [5,10,30,50]
ob_readings = {}
for i in range(len(time)):
    for j in range(len(distance)):
        ObBias = Bias_persistance(j,i)
        mean = np.mean(ObBias)
        std = np.std(ObBias)
        ob_readings[(distance[j], time[i])] = (mean,std)



"""
This declares the type of wound in which the attractant if a diffusion type is not chosen,
the assumed choice is the continious production.
diffusion choices can be set by setting the dynamics variable =:
0 = Continious production between t and 𝛕
1 = 𝛿 function initial condition
"""
wound = PointWound(position=np.array([0, 0]))
inferer_con= AttractantInferer(ob_readings, wound=wound, t_units='minutes',dynamics = 0)

Posterior_con = inferer_con.Ensembleinfer(nwalkers,niter)
Post_Save_con = np.save('/Users/danieltudor/Desktop/cont_inference',Posterior_con)


inferer_del = AttractantInferer(ob_readings, wound=wound, t_units='minutes',dynamics = 1)

Posterior_del = inferer_del.Ensembleinfer(nwalkers,niter)
Post_Save_Del = np.save('/Users/danieltudor/Desktop/Del_inference',Posterior_del)
