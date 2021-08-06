import os
import sys
sys.path.append(os.path.abspath('..'))
import matplotlib.pyplot as plt

import numpy as np


acceptance_rate_production = np.load('../data/Synthetic_Data/acceptance_rate_production.npy', allow_pickle=True)
acceptance_rate_delta = np.load('../data/Synthetic_Data/acceptance_rate_delta.npy', allow_pickle=True)
n_walkers = np.load('../data/Synthetic_Data/number_of_walkers.npy', allow_pickle=True)
BayesFactor = np.load('../data/Synthetic_Data/BayesFactor.npy', allow_pickle=True)

plt.plot(n_walkers, acceptance_rate_production,'o-', label='Production model')
plt.plot(n_walkers, acceptance_rate_delta,'o-', label='Delta model')

plt.xlabel('Number of walkers')
plt.ylabel('Acceptance rate')
plt.legend()
plt.show()

"""
plt.plot(n_walkers, BayesFactor)
plt.xlabel('Number of walkers')
plt.ylabel('Bayes Factor')
plt.show()
"""


