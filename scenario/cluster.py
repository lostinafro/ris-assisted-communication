#!/usr/bin/env python3
# filename "cells.py"

import numpy as np
import scenario.common as common
from scenario.common import cluster_shapes, circular_uniform, semicircular_uniform, cyl2cart, cart2cyl, fading
from scenario.nodes import UE, BS, RIS
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib import rc
from scipy.constants import speed_of_light


class Cluster:
    """Creates a scenario defined by an area where UEs, BSs and RISs coexist.
    The coordinate system is centered in the center of the cluster.
    """

    def __init__(self,
                 shape: str,
                 x_size: float,
                 y_size: float = None,
                 carrier_frequency: float = 3e9,
                 bandwidth: float = 180e3,
                 direct_channel: str = 'LoS',
                 reflective_channel: str = 'LoS',
                 pl_exponent: float = 2,
                 frequencies: int = 1,
                 symbols: int = 1,
                 rng: np.random.RandomState = None):

        """Constructor of class.

        Parameters
        __________
        :param x_size: float, x size of the cluster (radius if a circle, max side if box, etc.)
        :param y_size: float, y size of the cluster (same as x for circle)
        :param carrier_frequency: float, central frequency in Hertz.
        :param bandwidth: float, bandwidth in Hertz.
        :param rng: random nuber generator for reproducibility purpose
        """
        # Input check
        assert shape in cluster_shapes, f'Cluster kind not supported. Possible are: {cluster_shapes}'
        if y_size is None:
            y_size = x_size
        else:
            assert x_size >= y_size, f'Error in size definition: min_size must be <= max_size'

        # Physical attributes
        self.shape = shape
        self.x_size = x_size
        self.y_size = y_size

        # Channel attributes
        self.direct_channel = direct_channel
        self.reflective_channel = reflective_channel
        self.pl_exponent = pl_exponent

        # Bandwidth available
        self.fc = carrier_frequency
        self.wavelength = speed_of_light / carrier_frequency
        self.wavenumber = 2 * np.pi / self.wavelength
        self.bw = bandwidth

        # Random State generator
        self.rng = np.random.RandomState() if rng is None else rng

        # Nodes
        self.bs = None
        self.ue = None
        self.ris = None

        # Channels
        self.h_dir = None
        self.h_ris = None
        # TODO: transform the following into an attribute of ris class
        self.ris_array_factor = None

    def random_positioning(self, n: int):
        """Generate n positions in cartesian coordinate depending on the shape.

        :param n: int, number of points
        :return: np.array (n,3), position in cartesian coordinate of n points
        """
        if self.shape == 'circle':
            return cyl2cart(np.hstack((circular_uniform(n, self.x_size, rng=self.rng), np.zeros((n, 1)))))
        elif self.shape == 'semicircle':
            return cyl2cart(np.hstack((semicircular_uniform(n, self.x_size, rng=self.rng), np.zeros((n, 1)))))
        elif self.shape == 'box':
            x = self.x_size * self.rng.rand(n, 1) - self.x_size / 2
            y = self.y_size * self.rng.rand(n, 1) - self.y_size / 2
            return np.hstack((x, y, np.zeros(n, 1)))

    def check_position(self, pos: np.array):
        """Check if position is inside the cluster.

        :param pos: np.array (n,3), position to be tested
        :return: np.array (n), boolean indicating with True is the position is inside the cluster
        """
        if self.shape == 'circle':
            return cart2cyl(pos)[:, 0] <= self.x_size
        elif self.shape == 'semicircle':
            cyl_pos = cart2cyl(pos)
            return (cyl_pos[:, 0] <= self.x_size) & (cyl_pos[:, 1] <= np.pi) & (cyl_pos[:, 1] >= 0)
        elif self.shape == 'box':
            return (np.abs(pos[:, 0]) <= self.x_size/2) & (np.abs(pos[:, 1]) <= self.y_size/2)

    def place_bs(self,
                 n: int = 1,
                 position: np.array = None,
                 gain: float = None,
                 max_pow: float = None):
        """Place BS in the scenario. If a new set of BSs is set the old one is canceled.

        Parameters
        ----------
        :param n: number of BS to place,
        :param position: np.array shape (n,3), position of the bs in cartesian coordinate.
        :param gain: float, BS antenna gain G_b.
        :param max_pow: float, maximum power available at the BS.
        """
        # Input check
        if not isinstance(n, int) or (n <= 0):  # Cannot add a negative number of nodes
            raise ValueError('n must be int >= 0.')
        if position is None:
            position = self.random_positioning(n)
        elif not np.any(self.check_position(position)):
            raise ValueError(f'Error in positioning for BS, recheck please.')

        # Append BS
        self.bs = BS(n=n, pos=position, gain=gain, max_pow=max_pow)

    def place_ue(self,
                 n: int,
                 position: np.array = None,
                 gain: float = None,
                 max_pow: float = None):
        """Place a predefined number n of UEs in the box. If a new set of UE is set the old one is canceled.

        Parameters
        ----------
        :param n: int, number of UEs to be placed.
        :param position: np.array shape (n,3), position of the UE in cartesian coordinate.
        :param gain: float, UE antenna gain G_k.
        :param max_pow: float, maximum power available at each UE.
        """
        # Input check
        if not isinstance(n, int) or (n <= 0):  # Cannot add a negative number of nodes
            raise ValueError('n must be int >= 0.')
        if position is None:
            position = self.random_positioning(n)
        elif not np.any(self.check_position(position)):
            raise ValueError(f'Error in positioning for UEs, recheck please.')

        # Append UEs
        self.ue = UE(n=n, pos=position, gain=gain, max_pow=max_pow)

    def place_ris(self,
                  n: int,
                  position: np.ndarray = None,
                  num_els_x: int = None,
                  dist_els_x: float = None,
                  num_els_z: int = None,
                  dist_els_z: float = None):
        """Place a set RIS in the scenario. If a new set is set the old one is canceled.

        Parameters
        ----------
        :param n: int, number of RIS to be placed.
        :param position : ndarray of shape (3,), position of the RIS in rectangular coordinates.
        :param num_els_x : int, number of elements along x-axis.
        :param dist_els_x: float, distance between elements on x-axis. Default: wavelength
        :param num_els_z: int, number of elements along z-axis. Default: as num_els_x
        :param dist_els_z: float, distance between elements on x-axis. Default: as dist_els_x
        """
        # Input check
        if not isinstance(n, int) or (n <= 0):  # Cannot add a negative number of nodes
            raise ValueError('n must be int >= 0.')
        if position is None:
            position = np.array([0, 0, 0])
        elif not np.any(self.check_position(position)):
            raise ValueError(f'Error in positioning for RIS, recheck please.')
        if num_els_x is None:
            num_els_x = 8
        if dist_els_x is None:
            dist_els_x = self.wavelength

        # Append RIS
        self.ris = RIS(n=n, pos=position, num_els_h=num_els_x, dist_els_h=dist_els_x, num_els_v=num_els_z, dist_els_v=dist_els_z)
        # Configure codebook RIS
        self.ris.init_std_configurations(self.wavelength)

    def get_channel_model(self):
        """Get Downlink (DL) and Uplink (UL) channel gain.

        Returns
        -------
        channel_gains_dl : ndarray of shape (num_configs, num_ues)
            Downlink channel gain between the BS and each UE for each RIS configuration.

        channel_gains_ul : ndarray of shape (num_configs, num_ues)
            Uplink channel gain between the BS and each UE for each RIS configuration.

        """
        # Compute DL pathloss component of shape (num_ues, )
        num = self.bs.gain * self.ue.gain * self.ris.dist_els_h ** 2
        den = (4 * np.pi * self.bs.distance * self.ue.distances)**2

        pathloss_dl = (num / den) * np.cos(self.bs.angles) ** 2

        # Compute UL pathloss component of shape (num_ues, )
        pathloss_ul = (num / den) * np.cos(self.ue.angles)**2

        # Compute constant phase component of shape (num_ues, )
        distances_sum = (self.bs.distance + self.ue.distances)
        disagreement = (np.sin(self.bs.angles) - np.sin(self.ue.angles)) * ((self.ris.num_els_h + 1) / 2) * self.ris.dist_els_h

        phi = - self.wavenumber * (distances_sum - disagreement)

        # Compute array factor of shape (num_configs, num_ues)
        enumeration_num_els_x = np.arange(1, self.ris.num_els_h + 1)
        sine_differences = (np.sin(self.ue.angles[np.newaxis, :, np.newaxis]) - np.sin(self.ris.configs[:, np.newaxis, np.newaxis]))

        argument = self.wavenumber * sine_differences * enumeration_num_els_x[np.newaxis, np.newaxis, :] * self.ris.dist_els_h

        array_factor_dl = self.ris.num_els_v * np.sum(np.exp(+1j * argument), axis=-1)
        array_factor_ul = array_factor_dl.conj()

        # Compute channel gains of shape (num_configs, num_ues)
        channel_gains_dl = np.sqrt(pathloss_dl[np.newaxis, :]) * np.exp(+1j * phi[np.newaxis, :]) * array_factor_dl
        channel_gains_ul = np.sqrt(pathloss_ul[np.newaxis, :]) * np.exp(-1j * phi[np.newaxis, :]) * array_factor_ul

        return channel_gains_dl, channel_gains_ul

    def get_channel_model_slotted_aloha(self):
        """Get Downlink (DL) and Uplink (UL) channel gain.

        Returns
        -------
        channel_gains_dl : ndarray of shape (1, num_ues)
            Downlink channel gain between the BS and each UE.

        channel_gains_ul : ndarray of shape (1, num_ues)
            Uplink channel gain between the BS and each UE.

        """
        # Compute DL pathloss component of shape (num_ues, )
        distance_bs_ue = np.linalg.norm(self.ue.pos - self.bs.pos, axis=-1)

        num = self.bs.gain * self.ue.gain
        den = (4 * np.pi * distance_bs_ue)**2

        pathloss_dl = num / den

        # Compute UL pathloss component of shape (num_ues, )
        pathloss_ul = pathloss_dl

        # Compute constant phase component of shape (num_ues, )
        phi = - self.wavenumber * distance_bs_ue

        # Compute channel gains of shape (num_configs, num_ues)
        channel_gains_dl = np.sqrt(pathloss_dl[np.newaxis, :]) * np.exp(+1j * phi[np.newaxis, :])
        channel_gains_ul = np.sqrt(pathloss_ul[np.newaxis, :]) * np.exp(-1j * phi[np.newaxis, :])

        return channel_gains_dl, channel_gains_ul

    def build_channels(self):
        """Build channels depending"""
        # Common values
        gain = self.bs.gain + self.ue.gain
        # Direct channel
        d_bu = np.linalg.norm(self.bs.pos - self.ue.pos, axis=1)
        # Path loss
        pl_bu = 20 * np.log10(4 * np.pi / self.wavelength) - gain + 10 * self.pl_exponent * np.log10(d_bu)
        fad = fading(typ=self.direct_channel, dim=pl_bu.shape)
        self.h_dir = fad * np.sqrt(10 ** (-pl_bu / 10)) * np.exp(- 1j * self.wavenumber * d_bu)

        # reflective channel
        d_br = np.linalg.norm(self.bs.pos - self.ris.pos, axis=1)
        d_ru = np.linalg.norm(self.ris.pos - self.ue.pos, axis=1)
        # Path loss
        pl_ru = 20 * np.log10(4 * np.pi) - gain - 20 * np.log10(self.ris.dist_els_h * self.ris.dist_els_v) + 20 * np.log10(d_br * d_ru)
        # TODO: what happens if RIS is not in the origins?
        # positioning versors
        r_k = (self.ue.pos.T/d_ru).T    # transpose operation needed to obtain K x 3 vector
        r_b = (self.bs.pos.T/d_br).T    # transpose operation needed to obtain B x 3 vector
        # phases
        pos_factor = np.sin(self.ue.az_angle) * np.sin(self.bs.el_angle)
        delta_psi_ue = - self.wavenumber * (d_ru[np.newaxis].T - r_k @ self.ris.el_pos)
        delta_psi_bs = - self.wavenumber * (d_br - r_b @ self.ris.el_pos)
        # delta_psi_ue = self.wavenumber * (r_k @ self.ris.el_pos)
        # delta_psi_bs = self.wavenumber * (r_b @ self.ris.el_pos)
        self.ris_array_factor = np.sum(self.ris.actual_conf * np.exp(1j * (delta_psi_ue + delta_psi_bs)), axis=1)
        # Overall
        fad = fading(typ=self.reflective_channel, dim=pl_ru.shape)
        self.h_ris = self.ris_array_factor * pos_factor * fad * np.sqrt(10 ** (-pl_ru / 10))
        return self.h_dir + self.h_ris


    def plot_scenario(self, render: bool = False, *args):
        """This method will plot the scenario of communication
        """
        # LaTeX type definitions
        rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
        rc('text', usetex=True)
        # Open axes
        _, ax = plt.subplots()

        # Displacement value
        delta = self.x_size / 100
        # Box positioning
        if self.shape == 'box':
            shape = plt.Rectangle((-self.x_size / 2, -self.y_size / 2), self.x_size, self.y_size, ec="black", ls="--", lw=1, fc='#45EF0605')
            ax.set_ylim(- self.y_size - delta, self.y_size + delta)
            ax.set_xlim(- self.x_size - delta, self.x_size + delta)
            ax.axis('equal')
        elif self.shape == 'circle':
            shape = plt.Circle((0, 0), self.x_size, ec="black", ls="--", lw=1, fc='#45EF0605')
            ax.set_ylim(- self.x_size - delta, self.x_size + delta)
            ax.set_xlim(- self.x_size - delta, self.x_size + delta)
            ax.axis('equal')
        else: # or self.shape == 'semicircle':
            shape = plt.Circle((0, 0), self.x_size, ec="black", ls="--", lw=1, fc='#45EF0605')
            ax.set_ylim(- delta, self.x_size + delta)
            ax.set_xlim(- self.x_size - delta, self.x_size + delta)
            ax.axis('scaled')
        ax.add_patch(shape)
        # User positioning
        # BS
        plt.scatter(self.bs.pos[:, 0], self.bs.pos[:, 1], c=common.node_color['BS'], marker=common.node_mark['BS'], label='BS')
        # plt.text(self.bs.pos[:, 0], self.bs.pos[:, 1] + delta, s='BS', fontsize=10)
        # UE
        plt.scatter(self.ue.pos[:, 0], self.ue.pos[:, 1], c=common.node_color['UE'], marker=common.node_mark['UE'], label='UE')
        for k in np.arange(self.ue.n):
            plt.text(self.ue.pos[k, 0], self.ue.pos[k, 1] + delta, s=f'{k}', fontsize=10)
        # RIS
        plt.scatter(self.ris.pos[:, 0], self.ris.pos[:, 1], c=common.node_color['RIS'], marker=common.node_mark['RIS'], label='RIS')
        # plt.text(self.ris.pos[:, 0], self.ris.pos[:, 1] + delta, s='RIS', fontsize=10)
        # Set axis
        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$y$ [m]')
        # Legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        # Finally
        plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)
        common.printplot(render, filename=args[0], dirname=args[1])
