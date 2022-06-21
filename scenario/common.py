# filename "common.py"
# Global methods: contains general methods used everywhere

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os
from datetime import date
import tikzplotlib
from scipy.constants import c, k
from scipy.stats import rice, norm
from tqdm import tqdm
import sys

# Global dictionaries
cluster_shapes = {'box', 'circle', 'semicircle'}
channel_stats = {'LoS', 'fading'}
node_labels = {'BS': 0, 'UE': 1, 'RIS': 2}

# The following are defined for graphic purpose only
node_color = {'BS': '#DC2516',  'UE': '#36F507', 'RIS': '#0F4EEA'}
node_mark = {'BS': 'o', 'UE': 'x', 'RIS': '^'}

# The supported channel types are the following.
channel_types = {'LoS', 'No', 'AWGN', 'Rayleigh', 'Rice', 'Shadowing'}

# Custom distributions
def circular_uniform(n: int, r_outer: float, r_inner: float = 0, rng: np.random.RandomState = None):
    """Generate n points uniform distributed on an annular region. The output is in polar coordinates.

    Parameters
    ----------
    :param n: int, number of points.
    :param r_outer: float, outer radius of the annular region.
    :param r_inner: float, inner radius of the annular region.
    :param rng: np.random.RandomState, random generator needed for reproducibility

    Returns
    -------
    rho: np.ndarray, distance of each point from center of the annular region.
    phi: np.ndarray, azimuth angle of each point.
    """
    if rng is None:
        rho = np.sqrt((r_outer ** 2 - r_inner ** 2) * np.random.rand(n, 1) + r_inner ** 2)
        phi = 2 * np.pi * np.random.rand(n, 1)
    else:
        rho = np.sqrt((r_outer ** 2 - r_inner ** 2) * rng.rand(n, 1) + r_inner ** 2)
        phi = 2 * np.pi * rng.rand(n, 1)
    return np.hstack((rho, phi))


def semicircular_uniform(n: int, r_outer: float, r_inner: float = 0, rng: np.random.RandomState = None):
    """Generate n points uniform distributed on an semi-annular region. The outputs is in polar coordinates.

    Parameters
    ----------
    :param n: int, number of points.
    :param r_outer: float, outer radius of the annular region.
    :param r_inner: float, inner radius of the annular region.
    :param rng: np.random.RandomState, random generator needed for reproducibility

    Returns
    -------
    rho: np.ndarray, distance of each point from center of the annular region.
    phi: np.ndarray, azimuth angle of each point.
    """
    if rng is None:
        rho = np.sqrt((r_outer ** 2 - r_inner ** 2) * np.random.rand(n, 1) + r_inner ** 2)
        phi = np.pi * np.random.rand(n, 1)
    else:
        rho = np.sqrt((r_outer ** 2 - r_inner ** 2) * rng.rand(n, 1) + r_inner ** 2)
        phi = np.pi * rng.rand(n, 1)
    return np.hstack((rho, phi))

# Fading channel
def fading(typ: str, dim: tuple = (1,),
           shape: float = 6, seed: int = None) -> np.ndarray:
    """Create a sampled fading channel from a given distribution and given
    dimension.

    Parameters
    __________
    typ : str in dic.channel_types,
        type of the fading channel to be used.
    dim : tuple,
        dimension of the resulting array.
    shape : float [dB],
        shape parameters used in the rice distribution modeling the power of
        the LOS respect to the NLOS rays.
    seed : int,
        seed used in the random number generator to provide the same arrays if
        the same is used.
    """
    if typ not in channel_types:
        raise ValueError(f'Type can only be in {channel_types}')
    elif typ == 'AWGN' or typ == 'LoS':
        return np.ones(dim)
    elif typ == 'No':
        return np.zeros(dim)
    elif typ == "Rayleigh":
        vec = norm.rvs(size=2 * np.prod(dim), random_state=seed)
        return (vec[0:1] + 1j * vec[1:2]).reshape(dim) / np.sqrt(2)  # TODO FIXING THIS BUG
    elif typ == "Rice":
        return rice.rvs(10 ** (shape / 10), size=np.prod(dim), random_state=seed).reshape(dim)  # TODO: FIX ALSO THIS
    elif typ == "Shadowing":
        return norm.rvs(scale=10 ** (shape / 10), random_state=seed)


# Physical noise
def thermal_noise(bandwidth, noise_figure=3, t0=293):
    """Compute the noise power [dBm] according to bandwidth and ambient temperature.

    :param bandwidth : float, receiver total bandwidth [Hz]
    :param noise_figure: float, noise figure of the receiver [dB]
    :param t0: float, ambient temperature [K]

    :return: power of the noise [dBm]
    """
    return watt2dbm(k * bandwidth * t0) + noise_figure  # [dBm]


# Utilities
def lin2dB(lin):
    return 10 * np.log10(lin)

def db2lin(db):
    return 10 ** (db / 10)

def dbm2watt(dbm):
    """Simply converts dBm to Watt"""
    return 10 ** (dbm / 10 - 3)


def watt2dbm(watt):
    """Simply converts Watt to dBm"""
    with np.errstate(divide='ignore'):
        return 10 * np.log10(watt * 1e3)

def np_ceil(a, precision=0):
    return np.round(a + 0.5 * 10 ** (-precision), precision)

def standard_bar(total_iteration):
    return tqdm(total_iteration, file=sys.stdout, leave=False, ncols=60, ascii=True)


# Coordinate system
def cart2cyl(pos: np.array):
    """ Transformation from cartesian to cylindrical coordinates.

    :param pos: np.array (n,3), position to be transformed
    :return: np.array (n,3), cartesian coordinate
    """
    rho = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    z = pos[:, 2]
    return np.vstack((rho, phi, z)).T


def cyl2cart(pos: np.array):
    """ Transformation from cylinder to cartesian coordinates.

    :param pos: np.array (n,3), position to be transformed
    :return: np.array (n,3), polar coordinate
    """
    x = pos[:, 0] * np.cos(pos[:, 1])
    y = pos[:, 0] * np.sin(pos[:, 1])
    z = pos[: , 2]
    return np.vstack((x, y, z)).T


def cart2spher(pos: np.array):
    """ Transformation from cartesian to cylindrical coordinates.

    :param pos: np.array (n,3), position to be transformed
    :return: np.array (n,3), cartesian coordinate
    """
    rho = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    theta = np.arccos(pos[:, 2] / rho)
    return np.vstack((rho, theta, phi)).T


def spher2cart(pos: np.array):
    """ Transformation from cylinder to cartesian coordinates.

    :param pos: np.array (n,3), position to be transformed
    :return: np.array (n,3), polar coordinate
    """
    x = pos[:, 0] * np.sin(pos[:, 1]) * np.cos(pos[:, 2])
    y = pos[:, 0] * np.sin(pos[:, 1]) * np.sin(pos[:, 2])
    z = pos[:, 0] * np.cos(pos[:, 1])
    return np.vstack((x, y, z)).T

class Position:
    # TODO: see if you can subclass numpy
    """Wrapper for position vectors. Generally a K x 3 vector, where K is the number of points."""
    def __init__(self, cartesian: np.array = None, cylindrical: np.array = None, spherical: np.array = None):
        """Constructor of the class.

        :param cartesian: np.array (K,3) representing the 3D Cartesian coordinates of K points.
        :param cylindrical: np.array (K,3) representing the 3D cylindrical coordinates of K points.
        :param spherical: np.array (K,3) representing the 3D spherical coordinates of K points.
        """
        if cartesian is not None:
            self._pos = cartesian
        elif cylindrical is not None:
            self._pos = cyl2cart(cylindrical)
        elif spherical is not None:
            self._pos = spher2cart(spherical)
        else:
            raise ValueError('No input has been provided.')

    @property
    def cart(self):
        """ Output cylindrical coordinates."""
        return self._pos

    @property
    def cyl(self):
        """ Output cylindrical coordinates."""
        rho = np.sqrt(self._pos[:, 0] ** 2 + self._pos[:, 1] ** 2)
        phi = np.arctan2(self._pos[:, 1], self._pos[:, 0])
        z = self._pos[:, 2]
        return np.vstack((rho, phi, z)).T

    @property
    def sph(self):
        """ Output spherical coordinates."""
        rho = np.sqrt(self._pos[:, 0] ** 2 + self._pos[:, 1] ** 2)
        phi = np.arctan2(self._pos[:, 1], self._pos[:, 0])
        theta = np.arccos(self._pos[:, 2] / rho)
        return np.vstack((rho, theta, phi)).T

    @property
    def norm(self):
        return np.linalg.norm(self._pos, axis=-1)

    @property
    def cartver(self):
        return (self.cart.T / self.norm).T    # transpose operation needed to obtain K x 3 vector

    @property
    def cylver(self):
        return (self.cyl.T / self.norm).T  # transpose operation needed to obtain K x 3 vector

    @property
    def sphver(self):
        return (self.sph.T / self.norm).T  # transpose operation needed to obtain K x 3 vector


    def __repr__(self):
        return self._pos.__repr__()


# Print scenarios
def printplot(fig: plt.Figure,
              ax: plt.Axes or np.array,
              render: bool = False,
              filename: str = '',
              dirname: str = '',
              title: str = None,
              labels: list = None):
    if isinstance(ax, plt.Axes):
        ax.grid(axis='both', color='#E9E9E9', linestyle='--', linewidth = 0.8)
        try:
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_ylabel(labels[2])
        except (TypeError, IndexError):
            pass
        if ax.get_legend_handles_labels()[1]:   # Is there is at least a label, print the legend
            ax.legend()
        if not render:
            ax.set_title(title)
            fig.show()
        else:
            filename = os.path.join(dirname, filename)
            ax.set_title(title)
            fig.savefig(filename + '.jpg', dpi=300)
            ax.set_title('')
            # try:
            #     tikzplotlib.clean_figure()
            # except ValueError:
            #     pass
            tikzplotlib.save(filename + '.tex')
            return
    else:
        if ax[0].get_legend_handles_labels()[1]:  # Is there is at least a label, print the legend
            ax[0].legend()
        for a in ax:
            a.grid(axis='both')
        try:
            ax[-1].set_xlabel(labels[0])
            for i, a in enumerate(ax):
                a.set_ylabel(labels[i+1])
        except (TypeError, IndexError):
            pass
        if not render:
            ax[0].set_title(title)
            fig.show()
        else:
            filename = os.path.join(dirname, filename)
            ax[0].set_title(title)
            fig.savefig(filename + '.jpg', dpi=300)
            ax[0].set_title('')
            # try:
            #     tikzplotlib.clean_figure()
            # except ValueError:
            #     pass
            tikzplotlib.save(filename + '.tex')



def standard_output_dir(subdirname: str) -> str:
    subdir = os.path.join(os.path.expanduser('~'), 'OneDrive/plots', subdirname)
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    output_dir = os.path.join(subdir, str(date.today()))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    return output_dir