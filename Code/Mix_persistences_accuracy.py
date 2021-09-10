#allow imports from the LM package
import os
import sys
sys.path.append(os.path.abspath('..'))

import datetime

from inference.walker_inference_mixed_persistence import BiasedPersistentInferer
import matplotlib.pyplot  as plt
from in_silico.walkers import BP_Leukocyte
from in_silico.sources import PointSource

# instantiate a walker with parameters w=0.5, p=0.6, b=0.7
import numpy as np

source = PointSource(position=np.array([0, 0]))

t = np.arange(15, 165, 15)

np.random.seed(100)
w1, p1, b1 = 0.5, 0.7, 0.7
params1 = np.array([w1, p1, b1])
walker1 = BP_Leukocyte(params1, source)

p2 = 0.3
params2 = np.array([w1, p2, b1])
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

total_paths = [np.concatenate((p1_paths_list[i], p2_paths_list[i]), axis = 2 ) for i in range(len(p1_paths_list))]


niter = 18000 # number of MCMC iterations
nwalkers = 90 # number of
iters = 0
for i in range(len(t)):
    inferer_total = BiasedPersistentInferer(total_paths[i], source)
    niter = 2000  # number of MCMC iterations
    nwalkers = 140  # number of
    EM_total = inferer_total.ensembleinfer(nwalkers, niter)

    sampler = EM_total[0]
    p_total = sampler.get_chain(discard=1000, thin=1, flat=True)
    P1 = p_total[:, 2]
    P2 = p_total[:, 3]
    np.save(f"/Users/danieltudor/Documents/GitHub/ImmuneCellMigrationAnalysis/data/Mixed_persistence_accuracy/EM_total{i}",
            EM_total)
    plt.hist(P1, label='$P_1$ = {:.2f} $\pm$ {:.2f}'.format(np.mean(P1), np.std(P1)), bins=100, alpha=0.6, density=True)
    plt.hist(P2, label='$P_2$ = {:.2f} $\pm$ {:.2f}'.format(np.mean(P2), np.std(P2)), bins=100, alpha=0.6, density=True)

    plt.axvline(0.3, color='black', ls='--', label="True value: 0.3")
    plt.axvline(0.7, color='black', ls='--', label="True value: 0.7")
    plt.xlabel("Persistence parameter", fontsize=16)
    plt.ylabel("Probability Density", fontsize=16)
    plt.legend()
    plt.savefig(
        f"/Users/danieltudor/Documents/GitHub/ImmuneCellMigrationAnalysis/data/Mixed_persistence_accuracy/Persistence_total_mix_w2_p1_p2_{i}.pdf")
    plt.show()


datetime_object = datetime.datetime.now()

file = open(f'../data/Mixed_persistence_accuracy/Param_values_{datetime_object}', 'w')
file.write('Parameter values')
file.write('----------------')
file.write(f'biases = {b1}, persistence 1 = {p1}, persistence 2 = {p2} , weights = {w1}')
file.write('Length of walk = 60')
file.write(f'Number of walkers = {t}')
file.close()
