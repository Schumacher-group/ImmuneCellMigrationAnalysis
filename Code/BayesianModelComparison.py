from pathlib import Path
import numpy as np
from inference.attractant_inference import AttractantInferer
from in_silico.sources import PointWound
from BayesFactor import BayesFactor
from PlotsForPublish import observed_bias_plots

data_dir = Path('../data')

params_delta = [200, 200, 0.2, 0.5, 3, 0.001]
params_production = [200, 200, 35, 0.2, 0.5, 3, 0.001]
wound = PointWound(position=np.array([0, 0]))

ob_readings = observed_bias_plots(25, 250, 10, 130, params_production, wound, 'production', save_fig=True)

n_walkers = 50
n_iters = 12000
inferer_prod = AttractantInferer(ob_readings, t_units='minutes', wound=wound, model="production")
post_production = inferer_prod.ensembleinfer(n_walkers, n_iters)

inferer_delta = AttractantInferer(ob_readings, t_units='minutes', wound=wound, model="delta")
post_delta = inferer_delta.ensembleinfer(n_walkers, n_iters)

BayesFactor(post_production[0], post_delta[0], 8000)


production_acceptance_fraction = np.mean(post_production[0].acceptance_fraction)
delta_acceptance_fraction = np.mean(post_delta[0].acceptance_fraction)
print(f"Acceptance fraction for production: {production_acceptance_fraction}"
      f" Acceptance fraction for delta: {delta_acceptance_fraction}")
save_posterior_results_production = np.save('../data/Synthetic_Data/posterior_results_production', post_production)
save_posterior_results_delta = np.save('../data/Synthetic_Data/posterior_results_delta', post_delta)