"""
Modelling of two persistence model using synthetic data, to check for the required number of movies to accurately
capture the two persistence's within the dataset
"""

import os
import sys

sys.path.append(os.path.abspath('..'))
import numpy as np
from in_silico.walkers import BP_Leukocyte
from in_silico.sources import PointSource
from inference.walker_inference_mixed_persistence import BiasedPersistentInferer

# make a source (for the walkers to migrate towards)
source = PointSource(position=np.array([0, 0]))
"""
Paths for tracks created using persistence = 0.7
"""
num_of_walkers = np.arange(15, 165, 15)
i = 0
for num in num_of_walkers:
    # instantiate a walker with parameters w=0.5, p=0.7, b=0.7
    w, p, b1 = 0.5, 0.7, 0.7
    params = np.array([w, p, b1])
    walker = BP_Leukocyte(params, source)

    # these are the starting positions of the leukocytes. Size must be of the form (N, 2)
    X0s = np.random.uniform(-5, 5, size=(num, 2))
    paths_p07 = walker.walk(X0s, T=50)
    """
    Paths for tracks created using persistence = 0.3
    """

    # instantiate a walker with parameters w=0.5, p=0.3, b=0.7
    w, p, b2 = 0.5, 0.3, 0.7
    params = np.array([w, p, b2])
    walker = BP_Leukocyte(params, source)

    # these are the starting positions of the leukocytes. Size must be of the form (N, 2)
    X0s = np.random.uniform(-5, 5, size=(num, 2))
    paths_p03 = walker.walk(X0s, T=50)

    print('The paths object has shape {}: (T+1, [x, y], n_walkers)'.format(paths_p03.shape))

    # Concatenate the two path arrays together to make one dataset with mixed persistence values

    total_paths = np.concatenate((paths_p03, paths_p07), axis=2)

    # Run inference on the mixed persistence's,  save and plot the results
    inferer_total = BiasedPersistentInferer(total_paths, source)
    niter = 5000  # number of MCMC iterations
    n_walkers = 140  # number of
    EM_total = inferer_total.ensembleinfer(n_walkers, niter)
    np.save(f"../data/Mixed_persistence_accuracy/Yi_data_total{i}_{num + num}",
            EM_total)
    i += 1
