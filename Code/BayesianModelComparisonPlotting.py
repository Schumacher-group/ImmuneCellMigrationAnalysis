from PlotsForPublish import plot_posterior_distributions, plot_posterior_chains, observed_bias_posterior_plots
import numpy as np
from in_silico.sources import PointWound
params_delta = [200, 200, 0.2, 0.5, 3, 0.001]
params_production = [200, 200, 35, 0.2, 0.5, 3, 0.001]
wound = PointWound(position=np.array([0, 0]))
post_prod = np.load('../data/Synthetic_Data/posterior_results_production.npy', allow_pickle=True)
post_delta = np.load('../data/Synthetic_Data/posterior_results_delta_deltaTrue.npy', allow_pickle=True)


params_prod = ['q', 'D', 'τ', 'R0', 'κ', 'm', 'b0']
plot_posterior_distributions(post_prod, params_prod, 7, name='Production_delTrue', save_fig=True)
plot_posterior_chains(post_prod, params_prod, name='Production_delTrue', n_discards=6000, save_fig=True)


params_delta = ['c0', 'D', 'R0', 'κ', 'm', 'b0']
plot_posterior_distributions(post_delta, params_delta, 6, name='Delta_delTrue', save_fig=True)
plot_posterior_chains(post_delta, params_delta, name='Delta_delTrue', n_discards=6000, save_fig=True)


production = post_prod[0].get_chain(discard=6000, thin=5, flat=True)
delta = post_delta[0].get_chain(discard=6000, thin=5, flat=True)
observed_bias_posterior_plots(25, 250, 10, 130, params_production, production, wound, true_model='production',
                         post_model='production', save_fig=True)
observed_bias_posterior_plots(25, 250, 10, 130, params_production, delta, wound, true_model='production',
                         post_model='delta', save_fig=True)
