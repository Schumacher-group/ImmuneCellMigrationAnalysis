import os
import sys
sys.path.append(os.path.abspath('..'))
from in_silico.sources import CellsOnWoundMargin, PointWound, CellsInsideWound
import numpy as np
import matplotlib.pyplot as plt
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



#Chose wound locations
wound_centre = PointWound(position=np.array([0, 0]))
# This imports the numpy arrays for each model
ProdTime = np.load('/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/data/DiffProdTime.npy',allow_pickle=True)
ProdTime = DiffProdTime[0]
ProdTimeSamples = ProdTime.get_chain(discard=5000,thin =20, flat=True)

DeltaSpike = np.load('/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/data/DiffNoTime.npy',allow_pickle=True)
DeltaSpikesampler = DiffNoProd[0]
DeltaSpikeSamples = DeltaSpikesampler.get_chain(discard=5000,thin =20, flat=True)

# This calculates the HM for each  model type
from inference.attractant_inference import AttractantInferer
inferer_prodtime = AttractantInferer(ob_readings, wound=wound_centre, t_units='minutes')
pi_hat_1 = []
for theta in ProdTimeSamples[np.random.randint(len(ProdTimeSamples), size=500)]:
    pi_hat_1.append(1 / (inferer_prodtime.log_likelihood(params=theta)))
HME_model1 = 1 / (np.mean(pi_hat_1))



from inference.attractant_inference_diff import AttractantInferer
inferer_delta = AttractantInferer(ob_readings, wound=wound_centre, t_units='minutes')
pi_hat = []
for theta in DeltaSpikeSamples[np.random.randint(len(DeltaSpikeSamples), size=500)]:
    pi_hat.append(1 / (inferer_delta.log_likelihood(params=theta)))
HME = 1 / (np.mean(pi_hat))
HME_model2 = HME



Bayes_Factor = HME_model1 / HME_model2


print("Bayes Factor = ", Bayes_Factor)
print("HME for model 1 = {} and model 2 = {}".format(HME_model1,HME_model2))
print("pi_hat for model 2:", pi_hat[:20])
print("pi_hat for model 1:", pi_hat_1[:20])
