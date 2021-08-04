from PlotsForPublish import plot_posterior_distributions, plot_posterior_chains, observed_bias_posterior_plots, \
    plot_marginal_joint_dists
import numpy as np
from in_silico.sources import PointWound
"""
Parameters for models, set wound location and load the posterior chains 
"""
params_delta = [200, 200, 0.2, 0.5, 3, 0.001]
params_production = [200, 200, 35, 0.2, 0.5, 3, 0.001]
wound = PointWound(position=np.array([0, 0]))
post_prod = np.load('../data/Synthetic_Data/posterior_results_production.npy', allow_pickle=True)
post_delta = np.load('../data/Synthetic_Data/posterior_results_delta_deltaTrue.npy', allow_pickle=True)
params_prod_name = ['q', 'D', 'τ', 'R0', 'κ', 'm', 'b0']
params_delta_name = ['c0', 'D', 'R0', 'κ', 'm', 'b0']
production = post_prod[0].get_chain(discard=6000, thin=5, flat=True)
delta = post_delta[0].get_chain(discard=6000, thin=5, flat=True)

"""
Plotting for production model; posterior distributions, chains, inferred observed bias and marginal-joint distributions
"""
plot_posterior_distributions(post_prod, params_prod_name, 7, name='Production', save_fig=True)
plot_posterior_chains(post_prod, params_prod_name, name='Production', n_discards=6000, save_fig=True)
plot_marginal_joint_dists(production, params_prod_name, model='production', save_fig=False)
observed_bias_posterior_plots(25, 250, 10, 130, params_production, production, wound, true_model='production',
                              post_model='production', save_fig=True)

"""
Plotting for delta model; posterior distributions, chains, inferred observed bias and marginal-joint distributions
"""

plot_posterior_distributions(post_delta, params_delta_name, 6, name='Delta', save_fig=True)
plot_posterior_chains(post_delta, params_delta, name='Delta', n_discards=6000, save_fig=True)
plot_marginal_joint_dists(delta, params_delta_name, model='delta', save_fig=True)
observed_bias_posterior_plots(25, 250, 10, 130, params_production, delta, wound, true_model='production',
                              post_model='delta', save_fig=True)
