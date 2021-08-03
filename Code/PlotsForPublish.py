import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import numpy as np
from pathlib import Path

data_dir = Path('../data')


def observed_bias_plots(x_start, x_stop, t_start, t_stop, param, wound_loc, model, save_fig=False, **kwargs):
    # noinspection DuplicatedCode
    from inference.attractant_inference import observed_bias

    # instantiate a point wound

    # where to measure observed bias
    r_points = np.arange(x_start, x_stop, x_start)
    r = np.linspace(x_start, x_stop + x_start, 100)
    t = np.arange(t_start, t_stop, 20)
    fig, axes = plt.subplots(ncols=1, nrows=len(t), sharex=True)

    for ax, p in zip(axes, t):
        ax.set_ylabel('$t={}$'.format(p), rotation=0, size='large', labelpad=35)

    # plot the points
    lines = []
    scatters = []
    for i, tt in enumerate(t):
        col = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        lines.append(
            axes[i].plot(r, observed_bias(param, r, tt, wound_loc, model=model), color=col, linewidth=1)[0])
        scatters.append(
            axes[i].plot(r_points, observed_bias(param, r_points, tt, wound_loc, model=model),
                         color=col,
                         marker='o', linewidth=0, markersize=4)[0])

        axes[i].set_ylim(0, 0.3)

    axes[0].set_title(f'Observed bias from {model} attractant model')
    axes[-1].set_xlabel('Distance, microns')
    plt.tight_layout()
    if save_fig == True:
        plt.savefig('../data/Synthetic_Data/observed_bias.pdf', format='pdf')
    plt.show()

    def scatter(scatters):
        ob_readings = {}
        for T, ob in zip(t, scatters):
            mus = ob.get_ydata()
            rs = ob.get_xdata()
            for r, mu in zip(rs, mus):
                ob_readings[(r, T)] = (mu, np.random.uniform(0.01, 0.03))

        return ob_readings

    return scatter(scatters)


def plot_posterior(sampler, variables, num_vars, name, save_fig=False, **kwargs):
    var_names = variables
    emcee_data = az.from_emcee(sampler[0], var_names=var_names).sel(draw=slice(100, None))
    az.plot_posterior(emcee_data, var_names=var_names[:])

    if save_fig == True:
        plt.savefig(f'../data/Synthetic_Data/posterior_plots_{name}.pdf', format='pdf')
    plt.show()


def plot_posterior_chains(sampler, params, name, save_fig=False, **kwargs):
    fig, axes = plt.subplots(nrows=1, ncols=len(params), figsize=(12, 5), sharex='col')
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for j in range(len(params)):
        axes[j].set_title(params[j])
        axes[j].set_yticks([])
        axes[j].plot(sampler[:, j], color=cols[j])

    plt.tight_layout()
    if save_fig == True:
        plt.savefig(f'../data/Synthetic_Data/posterior_plots_chains{name}.pdf', format='pdf')
    plt.show()
