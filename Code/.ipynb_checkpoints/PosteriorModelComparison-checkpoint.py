import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
from inference.attractant_inference import AttractantInferer
from in_silico.sources import PointWound
from PlotsForPublish import observed_bias_plots


params_delta = [80, 200, 0.2, 0.5, 3, 0.001]
params_production = [500, 300, 30, 0.2, 0.5, 3, 0.001]
wound = PointWound(position=np.array([0, 0]))

ob_readings = observed_bias_plots(20, 320, 5, 85, params_production, wound, 'production', save_fig=True)
Num_walkers = 70
n_iters = 20000

inferer_prod = AttractantInferer(ob_readings, t_units='minutes', wound=wound, model="production")
post_production = inferer_prod.ensembleinfer(Num_walkers, n_iters)
save_posterior_production = np.save('../data/Synthetic_Data/Posterior_production_70_walkers', post_production)
inferer_delta = AttractantInferer(ob_readings, t_units='minutes', wound=wound, model="delta")
post_delta = inferer_delta.ensembleinfer(Num_walkers, n_iters)
save_posterior_delta = np.save('../data/Synthetic_Data/Posterior_delta_70_walkers', post_delta)

"""
production_acceptance_fraction = np.mean(post_production[0].acceptance_fraction)
inferer_delta = AttractantInferer(ob_readings, t_units='minutes', wound=wound, model="delta")
post_delta = inferer_delta.ensembleinfer(Num_walkers, n_iters)

save_posterior_delta = np.save('../data/Synthetic_Data/Posterior_delta_70_walkers', post_delta)


"""
