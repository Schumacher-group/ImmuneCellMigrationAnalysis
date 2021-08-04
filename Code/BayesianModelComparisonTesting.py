import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from Data_cleaning import wound_locations, filelocation, reformat_file, csv_to_dataframe
from inference.attractant_inference import AttractantInferer, observed_bias
from in_silico.sources import CellsOnWoundMargin, PointWound, CellsInsideWound

class TestDataCleaning(unittest.TestCase):
    def test_observed_bias_true(self):
        tau = 28
        R0 = 0.2
        kappa = 0.5
        m = 3
        b0 = 0.001
        wound = PointWound(position=np.array([0, 0]))
        delta_params = np.array([10, 20, R0, kappa, m, b0])
        production_params = np.array([1000, 500, tau, R0, kappa, m, b0])

        r = 25
        tt = 10
        ActualResult = observed_bias(delta_params, r, tt, wound, model="delta")
        ExpectedResult = observed_bias(production_params, r, tt, wound, model="production")
        self.assertNotEqual(ActualResult, ExpectedResult, 'Different models chosen')

if __name__ == '__main__':
    unittest.main()
