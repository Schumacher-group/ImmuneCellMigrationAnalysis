import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from time import time
from in_silico.walkers import BP_Leukocyte
from in_silico.sources import CellsOnWoundMargin, CellsInsideWound, PointWound, PointSource
from utils.plotting import plot_paths
sys.path.append(os.path.abspath('..'))

# Function to produce the different cell paths depending on length

# parametrise some leukocytes
w, p, b = 0.6, 0.8, 0.3
source = PointSource(position=np.array([0, 0]))
walker = BP_Leukocyte(params=np.array([w, p, b]), source=source, s=0.1)

# Number here relates to the number of paths available to run the inference upon
def get_number_paths(number):
    np.random.seed(0)
    X0s = np.random.uniform(-5, 5, (number, 2))
    paths = walker.walk(X0s, T=120)
    return paths


cell_number = [get_number_paths(i) for i in range(10,510,100)]


from inference.walker_inference import BiasedPersistentInferer, prepare_paths
# Produces a list for the inference to run on
inferer = [BiasedPersistentInferer(cell_number[i], source) for i in range(5)]


niter = 1000
nwalkers = 80
Time_to_run = []
for i in range(2,5):
    start_time = time()
    EM = inferer[i].Ensembleinfer(nwalkers,niter)
    np.save('../../data/Test_path_data/Paths_{}_b{}'.format(i,b),EM)
    end_time = time()
    Time_to_run.append(end_time - start_time)


np.save('../../data/Test_path_data/Time_to_run',Time_to_run)
