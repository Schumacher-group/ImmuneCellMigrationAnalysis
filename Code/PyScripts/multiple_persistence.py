# allow imports from the LM package

import numpy as np
from in_silico.walkers import BP_Leukocyte
from in_silico.sources import PointSource
from inference.walker_inference import BiasedPersistentInferer
import os
import sys
sys.path.append(os.path.abspath('..'))

# make a source (for the walkers to migrate towards)
source = PointSource(position=np.array([0, 0]))

# instantiate a walker with parameters w=0.5, p=0.6, b=0.7
w, p, b = 0.4, 0.7, 0.8
params = np.array([w, p, b])
walker = BP_Leukocyte(params, source)


# these are the starting positions of the leukocytes. Size must be of the form
# (N, 2)
X0s = np.random.uniform(-5, 5, size=(200, 2))
paths_p09 = walker.walk(X0s, T=50)

# instantiate a walker with parameters w=0.5, p=0.6, b=0.7
w1, p1, b1 = 0.4, 0.3, 0.8
params1 = np.array([w1, p1, b1])
walker = BP_Leukocyte(params1, source)


# these are the starting positions of the leukocytes.
# Size must be of the form (N, 2)
X0s = np.random.uniform(-5, 5, size=(200, 2))
paths_p02 = walker.walk(X0s, T=50)

total_paths = np.concatenate((paths_p09, paths_p02), axis=2)


inferer_total = BiasedPersistentInferer(total_paths, source)
niter = 1000  # number of MCMC iterations
nwalkers = 800  # number of
EM_total = inferer_total.Ensembleinfer(nwalkers, niter)
np.save('/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/data/persistences', EM_total)
