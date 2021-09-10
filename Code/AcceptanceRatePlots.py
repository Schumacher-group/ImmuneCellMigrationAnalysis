import os
import sys
sys.path.append(os.path.abspath('..'))
import matplotlib.pyplot as plt

import numpy as np


acceptance_rate_production = np.load('../data/Synthetic_Data/acceptance_rate_production_deltamodel_60walkers_large.npy', allow_pickle=True)
acceptance_rate_delta = np.load('../data/Synthetic_Data/acceptance_rate_delta_deltamodel_60walkers_large.npy', allow_pickle=True)
n_walkers = np.load('../data/Synthetic_Data/number_of_walkers.npy', allow_pickle=True)
BayesFactor = np.load('../data/Synthetic_Data/BayesFactor.npy', allow_pickle=True)
mean_production = np.empty(len(acceptance_rate_production))
mean_production.fill(np.mean(acceptance_rate_production))
mean_delta = np.empty(len(acceptance_rate_production))
mean_delta.fill(np.mean(acceptance_rate_delta))
plt.plot(acceptance_rate_production,'o-', label='Production model')
plt.plot(acceptance_rate_delta,'o-', label='Delta model')
plt.plot(mean_production, 'k--', label='Mean production accept rate')
plt.plot(mean_delta, 'k-.', label='Mean delta accept rate')
plt.title("Observed bias data from delta model")
plt.xlabel('Iterations')
plt.ylabel('Acceptance rate')
plt.legend()
plt.savefig(f'../data/Synthetic_Data/60_walkers_average_acceptance_deltamodel.pdf', format='pdf')

plt.show()

"""
plt.plot(n_walkers, BayesFactor)
plt.xlabel('Number of walkers')
plt.ylabel('Bayes Factor')
plt.show()

"""

