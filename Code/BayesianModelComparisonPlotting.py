from PlotsForPublish import plot_posterior, plot_posterior_chains
import numpy as np


post_prod = np.load('../data/Synthetic_Data/posterior_results_production.npy',allow_pickle=True)
params_prod = ['q', 'D', 'τ', 'R0', 'κ', 'm', 'b0']
plot_posterior(post_prod, params_prod, 7, name='Production', save_fig=True)
plot_posterior_chains(post_prod, params_prod, name='Production', n_discards=6000, save_fig=True)


post_delta = np.load('../data/Synthetic_Data/posterior_results_delta.npy', allow_pickle=True)
params_delta = ['c0', 'D', 'R0', 'κ', 'm', 'b0']
plot_posterior(post_delta, params_delta, 6, name='Delta', save_fig=True)
plot_posterior_chains(post_delta, params_delta, name='Delta', n_discards=6000, save_fig=True)


