#allow imports from the LM package
import os
import sys
sys.path.append(os.path.abspath('..'))

import datetime

from inference.walker_inference_mixed_persistence import BiasedPersistentInferer as BiasedPersistentInfererMix

from in_silico.walkers import BP_Leukocyte
from in_silico.sources import PointSource

# instantiate a walker with parameters w=0.5, p=0.6, b=0.7
import numpy as np

source = PointSource(position=np.array([0, 0]))

t = np.arange(30, 330, 30)

np.random.seed(100)
w1, p1, b1 = 0.5, 0.8, 0.05
params1 = np.array([w1, p1, b1])
walker1 = BP_Leukocyte(params1, source)

w2, p2, b2 = 0.5, 0.9, 0.05
params2 = np.array([w2, p2, b2])
walker2 = BP_Leukocyte(params2, source)

p1_paths_list = []
p2_paths_list = []
total_paths_list = []
for step in t:
    X01s = np.random.uniform(-5, 5, size=(step, 2))
    X02s = np.random.uniform(-5, 5, size=(step, 2))

    paths_p1 = walker1.walk(X01s, T=60)
    paths_p2 = walker2.walk(X02s, T=60)

    p1_paths_list.append(paths_p1)
    p2_paths_list.append(paths_p2)

total_paths_list = [np.concatenate((p1_paths_list[i], p2_paths_list[i]), axis = 2 ) for i in range(len(p1_paths_list))]


niter = 12000 # number of MCMC iterations
nwalkers = 60 # number of
iters = 0
for i in range(len(t)):
    inferer_t = BiasedPersistentInfererMix(total_paths_list[i], source)
    total_inf = inferer_t.ensembleinfer(nwalkers, niter)
    np.save(f'../data/Mixed_persistence_accuracy/total_list_vals{i}_for_Yi', total_inf)
    print(f"{i}/{len(t)}")


datetime_object = datetime.datetime.now()

file = open(f'../data/Mixed_persistence_accuracy/Param_values_{datetime_object}', 'w')
file.write('Parameter values')
file.write('----------------')
file.write(f'biases = {b1,b2}, persistence 1 = {p1}, persistence 2 = {p2} , weights = {w1,w2}')
file.write('Length of walk = 60')
file.write(f'Number of walkers = {t}')
file.close()
