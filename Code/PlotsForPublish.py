import matplotlib.pyplot as plt
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
        plt.savefig(f'../data/Synthetic_Data/observed_bias_{model}.pdf', format='pdf')
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


def observed_bias_posterior_plots(x_start, x_stop, t_start, t_stop, true_param, post_param, wound_loc, true_model,post_model, save_fig=False,
                             **kwargs):
    # noinspection DuplicatedCode
    from inference.attractant_inference import observed_bias

    # instantiate a point wound

    # where to measure observed bias
    r_points = np.arange(x_start, x_stop, x_start)

    r = np.linspace(x_start, x_stop + x_start, 100)
    t = np.arange(t_start, t_stop, 20)
    samples = post_param
    fig, axes = plt.subplots(ncols=1, nrows=len(t), sharex=True)

    for ax, p in zip(axes, t):
        ax.set_ylabel('$t={}$'.format(p), rotation=0, size='large', labelpad=35)

    # plot the points
    scatters = []
    for i, tt in enumerate(t):
        col = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        scatters.append(
            axes[i].plot(r_points, observed_bias(true_param, r_points, tt, wound_loc, model=true_model),
                         color=col,
                         marker='o', linewidth=0, markersize=6, label = 'True observed bias')[0])
        for theta in samples[np.random.randint(len(samples), size=50)]:
            axes[i].plot(r, observed_bias(theta, r, tt, wound_loc, model=post_model), color="r", alpha=0.1)

        axes[i].set_ylim(0, 0.3)

    axes[0].set_title(f'Observed bias from {post_model} attractant model')
    axes[-1].set_xlabel('Distance, microns')
    axes[len(t)-1].legend()
    plt.tight_layout()
    if save_fig == True:
        plt.savefig(f'../data/Synthetic_Data/observed_bias_{post_model}_posterior.pdf', format='pdf')
    plt.show()


def plot_posterior_distributions(sampler, variables, num_vars, name, save_fig=False, **kwargs):
    var_names = variables
    emcee_data = az.from_emcee(sampler[0], var_names=var_names).sel(draw=slice(100, None))
    plt.suptitle(f"Posterior distributions from {name}")

    az.plot_posterior(emcee_data, var_names=var_names[:])
    if save_fig == True:
        plt.savefig(f'../data/Synthetic_Data/posterior_plots_{name}.pdf', format='pdf')
    plt.show()


def plot_posterior_chains(sampler, params, name, n_discards, save_fig=False, **kwargs):
    samples = sampler[0].get_chain(flat=True, discard=n_discards)
    fig, axes = plt.subplots(nrows=len(params), ncols=1, figsize=(12, 5), sharex='col')
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for j in range(len(params)):
        axes[j].set_title(params[j])
        axes[j].plot(samples[:, j], color=cols[j])

    plt.tight_layout()
    if save_fig == True:
        plt.savefig(f'../data/Synthetic_Data/posterior_plots_chains{name}.pdf', format='pdf')
    plt.show()
