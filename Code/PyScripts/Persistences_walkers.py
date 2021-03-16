#allow imports from the LM package
import os
import sys
sys.path.append(os.path.abspath('..'))
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
from utils.distributions import Normal
from in_silico.walkers import BP_Leukocyte
from in_silico.sources import PointSource
from utils.plotting import plot_paths

# make a source (for the walkers to migrate towards)
source = PointSource(position=np.array([0, 0]))
import random
# instantiate a walker with parameters w=0.5, p=0.6, b=0.7
w, p, b = 0.4, 0.7, 0.8
params = np.array([w, p, b])
walker = BP_Leukocyte(params,source)

# these are the starting positions of the leukocytes. Size must be of the form (N, 2)
X0s = np.random.uniform(-5, 5, size=(100, 2))
paths_p09 = walker.walk(X0s, T=50)

# instantiate a walker with parameters w=0.5, p=0.6, b=0.7
w, p, b = 0.4, 0.3, 0.8
params = np.array([w, p, b])
walker = BP_Leukocyte(params, source)

# these are the starting positions of the leukocytes. Size must be of the form (N, 2)
X0s = np.random.uniform(-5, 5, size=(100, 2))
paths_p02 = walker.walk(X0s, T=50)

total_paths = np.concatenate((paths_p02, paths_p09), axis=2)

from inference.walker_inference_mixed_persistence import BiasedPersistentInferer
inferer_total = BiasedPersistentInferer(total_paths, source)
niter = 2000 # number of MCMC iterations
nwalkers = 1200 # number of walkers, assumes 100 walkers per parameter (to make sure the param space is covered)
EM_total = inferer_total.Ensembleinfer(nwalkers,niter)

np.save("/home/danieltudor/Documents/ImmuneCellMigrationAnalysis/Mixed_persistence",EM_total)
