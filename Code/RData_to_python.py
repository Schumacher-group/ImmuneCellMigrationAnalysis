import pyreadr
import sys
import os
import numpy as np
sys.path.append(os.path.abspath('..'))

load_data_file = pyreadr.read_r('../data/Comparison_data/140514Wounded/HaemocytesTrajectories.RData')
print(load_data_file.items())
