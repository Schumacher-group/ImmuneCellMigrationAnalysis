import numpy as np
import pandas as pd
from fractions import Fraction
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from subprocess import check_output
import os
import time
from in_silico.sources import Source
from tqdm import tqdm



def make_gif(array: np.ndarray, save_as: str, delay: int = 10,
             time_first: bool = True, v_bounds: tuple = (None, None),
             dpi: int = None, cells: list = None, paths: list = None,
             extent: list = None, origin: str = None):
    """

    Make a gif file from a numpy array, using Matplotlib imshow over the x-y
    coordinates. Useful for plotting cell tracks and heat equation plots.
    Relies on the linux command line tool 'convert'.

    Parameters
    ----------
    array         A 3D numpy array of shape (T, X, Y) or (X, Y, T).
    save_as       String file name or path to save gif to. No file extension.
    delay         Time delay in ms between gif frames
    time_first    Boolean specifying whether time dimension appears first
    v_bounds      tuple of (vmin, vmax) to be passed to imshow.
    dpi           Resolution in dots per inch

        If working with pictures of cells:

    add_LOG       Whether to add a Laplacian of Gaussian circles to identify cells
    threshold     The threshold to use with LOG
    add_paths     Whether to plot particle paths

    """

    # check if the t-dimension is in index 0
    if not time_first:
        array = array.transpose((2, 0, 1))

    # useful for keeping heat equation time frames consistent
    vmin, vmax = v_bounds
    T, by, bx = array.shape

    # use a system of binary sequences to name the files (helps keep them in order ;P)
    bits = int(np.ceil(np.log2(T)))

    path_to_gif = '/'.join(save_as.split('/')[:-1]) + '/'
    if path_to_gif == '/':
        path_to_gif = './'

    if not os.path.exists(path_to_gif + 'tmp'):
        os.mkdir(path_to_gif + 'tmp')

    path_to_png = path_to_gif + 'tmp/'

    T0 = time.time()

    for t in tqdm(range(T), desc='Plotting images'):
        # sort name stuff
        name = bin(t)[2:]
        name = '0' * (bits - len(name)) + name
        # plot the image
        image = array[t, :, :]
        fig, ax = plt.subplots()
        plt.imshow(image, vmin=vmin, vmax=vmax, extent=extent, origin=origin)
        plt.title('t={:.2f}'.format(t))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([0, bx])
        ax.set_ylim([0, by])

        # add Laplacian of Gaussians
        if cells is not None:
            for x, y, r in cells[t]:
                c = plt.Circle((x, y), r, color='red', linewidth=0.5, fill=False)
                ax.add_patch(c)

        if paths is not None:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            i = 0
            for path in paths:
                if path.shape[0] > 1:
                    path_to_plot = path[path[:, 0] < t]
                    plt.plot(path_to_plot[:, 1], path_to_plot[:, 2], color=colors[i % len(colors)], linewidth=1,
                             alpha=0.75)
                    i += 1
            # plt.show()

        # save each png file
        plt.savefig(path_to_png + '{}.png'.format(name), dpi=dpi)
        plt.close()

    time.sleep(0.1)
    print('Creating gif')

    # convert pngs to gif
    check_output(['convert', '-delay', '{}'.format(delay), path_to_png + '*.png', save_as + '.gif'])

    for file_name in os.listdir(path_to_png):
        if '.png' in file_name:
            os.remove(path_to_png + file_name)

    os.rmdir(path_to_png)

    print('Done')


def latex(fraction: Fraction) -> str:
    """
    Convert a fraction.Fraction, in multiples of pi, to a nice
    latex representation
    """
    fraction = str(fraction)
    if '/' in fraction:
        l = fraction.split('/')
        l[0] = l[0].replace('1', '')
        if '-' in l[0]:
            fraction = '-\\' + 'frac{' + l[0][1:] + '\pi}{' + l[1] + '}'
        else:
            fraction = '\\' + 'frac{' + l[0] + '\pi}{' + l[1] + '}'
    elif fraction == '1':
        fraction = '\pi'
    elif fraction == '-1':
        fraction = '-\pi'
    elif fraction == '0':
        pass
    else:
        fraction += '\pi'
    return '${}$'.format(fraction)


def get_pi_ticks(between=(-np.pi, np.pi), step=np.pi / 4):
    """
    Get the positions and labels (string, latexed) for an axis
    in multiples of pi
    """
    start, stop = between
    ticks = np.array(list(np.arange(start, stop, step)) + [stop])
    labels = [latex(Fraction(number).limit_denominator()) for number in ticks / np.pi]
    return ticks, labels


def add_pi_ticks(ax, between=(-np.pi, np.pi), step=np.pi / 4, axis='x'):
    ticks, labels = get_pi_ticks(between=between, step=step)
    if axis == 'x':
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    elif axis == 'y':
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
    elif axis == 'both':
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)


def plot_wpb_dist(params: np.array,
                  title: str = None,
                  add_kde: bool = False,
                  y_max: float = None,
                  save_as: str = None,
                  ax=None,
                  legend=True):
    cols = {'w': '#1f77b4', 'p': '#ff7f0e', 'b': '#2ca02c'}

    if ax is None:
        fig, ax = plt.subplots()

    stds = np.std(params, axis=0)
    means = np.mean(params, axis=0)

    for i, typ in enumerate(['w', 'p', 'b']):
        ax.hist(params[:, i],
                label='${}$ = {:.2f} $\pm$ {:.2f}'.format(typ, means[i], stds[i]),
                bins=100,
                alpha=0.6,
                density=True,
                color=cols[typ])

    if add_kde:
        y_w = gaussian_kde(params[:, 0])
        y_p = gaussian_kde(params[:, 1])
        y_b = gaussian_kde(params[:, 2])
        x = np.linspace(0, 1, 250)
        ax.plot(x, y_w, color=cols['w'])
        ax.plot(x, y_p, color=cols['p'])
        ax.plot(x, y_b, color=cols['b'])

    if legend:
        plt.legend()

    plt.xlim(0, 1)

    if y_max is not None:
        plt.ylim(0, 40)

    plt.title(title)

    if save_as is not None:
        if '.' in save_as:
            plt.savefig(save_as[:save_as.index('.')] + '.pdf')
        else:
            plt.savefig(save_as + '.pdf')

    plt.show()


def set_source():
    import skimage
    from matplotlib.patches import Circle

    path = r'/media/ed/DATA/Datasets/Leukocytes/Control wounded 1hr/'
    tif_file = path + r'Pupae 1 concatenated ubi-ecad-GFP, srpGFP; srp-3xH2Amch x white-1HR.tif'
    frames = skimage.io.imread(tif_file)[:, 0, :, :]

    fig, ax = plt.subplots()

    T, by, bx = frames.shape
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([0, bx])
    ax.set_ylim([0, by])

    class SourceSetter:

        def __init__(self, fig, ax):
            self.fig, self.ax = fig, ax

            self.circ = Circle((None, None), 30, fill=False, linewidth=1, color='red')
            self.ax.add_artist(self.circ)
            self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        def on_click(self, event):
            if event.inaxes is None:
                return
            self.circ.center = event.xdata, event.ydata
            self.fig.canvas.draw()
            print(event.xdata, event.ydata)

    ax.imshow(frames[0, :, :])
    t = 1

    ss = SourceSetter(fig, ax)
    while True:
        ax.imshow(frames[t % T, :, :], origin='lower')
        plt.pause(0.2)
        del ax.images[0]
        t += 1

    # plt.show()


def plot_paths(paths: np.ndarray, source: Source):
    source_x, source_y = source.position
    T, _, N = paths.shape

    fig, ax = plt.subplots()

    max_x, min_x = paths[:, 0, :].max(), paths[:, 0, :].min()
    max_y, min_y = paths[:, 1, :].max(), paths[:, 1, :].min()
    x_diff = max_x - min_x
    y_diff = max_y - min_y
    max_x += 0.05 * x_diff
    min_x -= 0.05 * x_diff
    max_y += 0.05 * y_diff
    min_y -= 0.05 * y_diff

    step = (max_x - min_x) / 500

    x_space = np.arange(min_x, max_x, step)
    y_space = np.arange(min_y, max_y, step)
    X, Y = np.meshgrid(x_space, y_space)
    Rs = (X - source_x) ** 2 + (Y - source_y) ** 2
    Z = np.exp(- Rs)

    plt.imshow(Z, extent=[min_x, max_x, min_y, max_y], origin='lower')
    ax.set_aspect('equal')

    for n in range(N):
        plt.plot(paths[:, 0, n], paths[:, 1, n], linewidth=0.5)

    plt.show()


def plot_AD_param_dist(dist: np.ndarray, priors: list = None):
    """
    Plot the output distibution of the attractant dynamics parameters

    Parameters
    ----------
    dist        The output MCMC trails (from AttractantInferer().infer())
    priors      A list of the prior distributions (from AttractantInferer().priors)

    """

    fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(17, 5), sharex='col')
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

    names = ['$q$ [mol min$^{-1}$]', '$D$ [$\mu m^{2}$ min$^{-1}$]', 'τ [min]', '$R_0$ [mol $\mu m^{-2}$]',
             '$\kappa_d$ [mol $\mu m^{-2}$]', '$m$ [$\mu m^{2}$ mol$^{-1}$]', '$b_0$ [unitless]']

    for j in range(7):
        axes[j].set_title(names[j])
        axes[j].set_yticks([])
        axes[j].hist(dist[:, j], bins=50, color=cols[j], alpha=0.6, density=True)
        if priors is not None:
            axes[j].axvline(priors[j], color='black', ls='--', label=f"True value: {priors[j]}")
            axes[j].legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_find_wound_location(dataframe: pd.DataFrame):
    trajectory_group = dataframe.groupby('Track_ID')
    final_data = (
        pd.concat([trajectory_group.tail(1)]).drop_duplicates().sort_values('Track_ID').reset_index(drop=True))
    final_data['x'] = final_data['x'].astype(float)
    final_data['y'] = final_data['y'].astype(float)
    x = final_data['x'].tolist()
    y = final_data['y'].tolist()

    plt.hist2d(x, y)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.title("Final t distribution of tracks")
    plt.show()


# Plot of temporal bins of the trajectory data
def plotxy_time_bins(dataframe: pd.DataFrame):
    trajectory = dataframe
    t20 = trajectory[(trajectory['t'] >= (0*60) )& (trajectory['t'] <= (20 * 60))]
    t35 = trajectory[(trajectory['t'] >= (20 * 60)) & (trajectory['t'] <= (35 * 60))]
    t50 = trajectory[(trajectory['t'] >= (35 * 60)) & (trajectory['t'] <= (50 * 60))]
    t65 = trajectory[(trajectory['t'] >= (50 * 60)) & (trajectory['t'] <= (65 * 60))]
    t90 = trajectory[(trajectory['t'] >= (65 * 60)) & (trajectory['t'] <= (90 * 60))]
    t125 = trajectory[(trajectory['t'] >= (90 * 60)) & (trajectory['t'] <= (125 * 60))]

    for ID, tracks in t125.groupby('Track_ID'):
        t125, = plt.plot(tracks['x'], tracks['y'], color='xkcd:rust', lw=1)
    for ID, tracks in t90.groupby('Track_ID'):
        t90, = plt.plot(tracks['x'], tracks['y'], color='xkcd:pine green', lw=1)
    for ID, tracks in t65.groupby('Track_ID'):
        t65, = plt.plot(tracks['x'], tracks['y'], color='xkcd:salmon', lw=1)
    for ID, tracks in t50.groupby('Track_ID'):
        t50, = plt.plot(tracks['x'], tracks['y'], color='xkcd:sky blue', lw=1)
    for ID, tracks in t35.groupby('Track_ID'):
        t35, = plt.plot(tracks['x'], tracks['y'], color='xkcd:sage', lw=1)
    for ID, tracks in t20.groupby('Track_ID'):
        t20, = plt.plot(tracks['x'], tracks['y'], color='xkcd:cobalt', lw=1)
    plt.legend(handles=[t125, t90, t65, t50, t35, t20],
               labels=["90 - 125 mins", "65 - 90 mins",
                       "50 - 65 mins", "35 - 50 mins",
                       "20 - 35 mins ", "15 - 20 mins"], title="Time Bins", loc=[1, 0.5])
    plt.xlabel("X-distance ($\\mu m$")
    plt.ylabel("Y-distance ($\\mu m$)")
    plt.title("Time binning for immune cell trajectories")
    plt.figure(figsize=(8, 6), dpi=80)
    plt.tight_layout()
    plt.show()


def plotxy_space_bins(dataframe: pd.DataFrame):
    trajectory = dataframe
    s25 = trajectory[(trajectory['r'] >= 0) & (trajectory['r'] <= 70)]
    s50 = trajectory[(trajectory['r'] >= 70) & (trajectory['r'] <= 140)]
    s75 = trajectory[(trajectory['r'] >= 140) & (trajectory['r'] <= 250)]
    s100 = trajectory[(trajectory['r'] >= 250) & (trajectory['r'] <= 360)]
    s125 = trajectory[(trajectory['r'] >= 360) & (trajectory['r'] <= 500)]

    for ID, tracks in s125.groupby('Track_ID'):
        s125, = plt.plot(tracks['x'], tracks['y'], color='xkcd:rust', lw=1)
    for ID, tracks in s100.groupby('Track_ID'):
        s100, = plt.plot(tracks['x'], tracks['y'], color='xkcd:pine green', lw=1)
    for ID, tracks in s75.groupby('Track_ID'):
        s75, = plt.plot(tracks['x'], tracks['y'], color='xkcd:salmon', lw=1)
    for ID, tracks in s50.groupby('Track_ID'):
        s50, = plt.plot(tracks['x'], tracks['y'], color='xkcd:sky blue', lw=1)
    for ID, tracks in s25.groupby('Track_ID'):
        s25, = plt.plot(tracks['x'], tracks['y'], color='xkcd:sage', lw=1)
    plt.legend(handles=[s125, s100, s75, s50, s25],
               labels=["360 - 500$\\mu m$", "250 - 360$\\mu m$",
                       "140 - 250$\\mu m$", "70 - 140$\\mu m$",
                       "0 - 70$\\mu m$"], title="Spatial Bins", loc=[1, 0.5])
    plt.xlabel("X-distance ($\\mu m$)")
    plt.ylabel("Y-distance ($\\mu m$)")
    plt.title("Spatial binning for immune cell trajectories")
    plt.tight_layout()
    plt.show()

"""
def plot_observed_bias(parameters, wound):
    params = np.array(parameters)
    fig, axes = plt.subplots(ncols=1, nrows=5, sharex=True)

    # instantiate a point wound

    # where to measure observed bias
    r_points = np.array([25, 50, 75, 100, 125, 150, 175, 200, 225, 250])  # , 175, 200, 225, 250
    r = np.linspace(25, 250, 100)
    t = np.array([10, 30, 50, 80, 120])

    for ax, p in zip(axes, t):
        ax.set_ylabel('$t={}$'.format(p), rotation=0, size='large', labelpad=35)

    # plot the points
    lines = []
    scatters = []
    for i, tt in enumerate(t):
        col = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        lines.append(axes[i].plot(r, observed_bias(params, r, tt, wound), color=col, linewidth=1)[0])
        scatters.append(
            axes[i].plot(r_points, observed_bias(params, r_points, tt, wound), color=col, marker='o', linewidth=0,
                         markersize=4)[0])
        axes[i].set_ylim(0, 0.3)

    axes[0].set_title('A point wound: observed bias as a function of distance')
    axes[-1].set_xlabel('Distance, microns')
    plt.tight_layout()

    ob_readings = {}
    for T, ob in zip(t, scatters):
        mus = ob.get_ydata()
        rs = ob.get_xdata()
        for r, mu in zip(rs, mus):
            ob_readings[(r, T)] = (mu, 0.02)

    return fig, ob_readings
"""
