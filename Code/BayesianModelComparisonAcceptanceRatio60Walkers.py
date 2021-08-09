import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
from inference.attractant_inference import AttractantInferer
from in_silico.sources import PointWound
from BayesFactor import BayesFactor
from PlotsForPublish import observed_bias_plots


params_delta = [80, 200, 0.2, 0.5, 3, 0.001]
params_production = [600, 400, 35, 0.2, 0.5, 3, 0.001]
wound = PointWound(position=np.array([0, 0]))

ob_readings = observed_bias_plots(25, 300, 10, 140, params_production, wound, 'production', save_fig=True)
Num_walkers  = 60
Num_acceptance_rate_production = []
Num_acceptance_rate_delta = []
Bayes_Factors = []
n_iters = 14000

for num in range(50):
    print(f"I am at  walker number of {num} of 60")
    inferer_prod = AttractantInferer(ob_readings, t_units='minutes', wound=wound, model="production")
    post_production = inferer_prod.ensembleinfer(num, n_iters)
    production_acceptance_fraction = np.mean(post_production[0].acceptance_fraction)
    inferer_delta = AttractantInferer(ob_readings, t_units='minutes', wound=wound, model="delta")
    post_delta = inferer_delta.ensembleinfer(num, n_iters)
    delta_acceptance_fraction = np.mean(post_delta[0].acceptance_fraction)
    Num_acceptance_rate_production.append(production_acceptance_fraction)
    Num_acceptance_rate_delta.append(delta_acceptance_fraction)
    Bayes_Factors.append(BayesFactor(post_production[0], post_delta[0], n_discards=7000))


save_accept_production = np.save('../data/Synthetic_Data/acceptance_rate_production_delta_60walkers',Num_acceptance_rate_production)
save_accept_delta = np.save('../data/Synthetic_Data/acceptance_rate_delta_delta_60walkers', Num_acceptance_rate_delta)
save_bayes_factor = np.save('../data/Synthetic_Data/BayesFactor_delta_60walkers',Bayes_Factors)
