from scenario.cluster import Cluster
import scenario.common as cmn
import numpy as np
from scipy.constants import c
import argparse


# GLOBAL STANDARD PARAMETERS
OUTPUT_DIR = cmn.standard_output_dir('ris-channel')
# Set parameters
NUM_EL_X = 8
CARRIER_FREQ = 1.8e9
BANDWIDTH = 180e3
PRBS = 275 * 3


# Class
class RISEnvironment2D(Cluster):
    def __init__(self,
                 radius: float,
                 bs_position: np.array,
                 ue_position: np.array,
                 ris_num_els: int = NUM_EL_X,
                 carrier_frequency: float = CARRIER_FREQ,
                 bandwidth: float = BANDWIDTH,
                 rbs: int = 400,
                 symbols: int = 1,
                 rng: np.random.RandomState = None):
        # Init parent class
        super().__init__(shape='semicircle',
                         x_size=radius,
                         carrier_frequency=carrier_frequency,
                         bandwidth=bandwidth,
                         direct_channel='LoS',
                         reflective_channel='LoS',
                         rbs=rbs,
                         rng=rng)
        self.place_bs(1, bs_position)
        self.place_ue(ue_position.shape[0], ue_position)
        self.place_ris(1, np.array([[0, 0, 0]]), num_els_x=ris_num_els, dist_els_x=self.wavelength/2)

    def set_std_configuration(self, index: int):
        return self.ris.set_std_configuration(self.wavenumber, index, self.bs.pos)

    def set_configuration(self, angle: float):
        return self.ris.set_configuration(self.wavenumber, angle, self.bs.pos)

    def best_configuration_selection(self):
        """Find the best configuration considering the azimuth angle only"""
        return np.abs(self.ris.std_config_angles[np.newaxis].T - self.ue.pos.sph[:, 2]).argmin(axis=0)

    def best_frequency_selection(self):
        """Find the best frequency for the transmission (residual phase shift on f0 MUST BE NULL)"""
        dist_triangle = self.ue.pos.norm + self.bs.pos.norm - np.linalg.norm(self.ue.pos.cart - self.bs.pos.cart[0], axis=1)
        integer_multiplier = np.floor(- self.f0 / c * dist_triangle) + np.arange(0, -self.RBs, -1)[np.newaxis].T
        return np.round((- (c * integer_multiplier) / dist_triangle - self.f0) / self.BW).T
        # Old version
        # ru = env.ue.pos.norm[u]
        # rub = np.linalg.norm(env.ue.pos.cart[u] - env.bs.pos.cart[0])
        # triangle = ru + rb - rub
        # # for i in range(env.ris.num_std_configs)[:-1]:
        # i = best_configurations[u]
        # Set configuration
        # _, varphi_x, varphi_z = env.set_std_configuration(best_configurations[u])
        # compute the right periodicity of the phase vs frequency
        # phase_sum = 0  # - (env.ris.num_els_h + 1) / 2 * env.ris.dist_els_h * varphi_x - (env.ris.num_els_v + 1) / 2 * env.ris.dist_els_v * varphi_z
        # k_min = np.floor(env.f0 / c * (phase_sum - triangle))
        # k = np.arange(k_min, k_min - 400, -1)
        # f = np.round(((env.f0 * phase_sum - c * k) / triangle - env.f0) / env.BW)


def command_parser():
    """Parse command line using arg-parse and get user data to run the render.

        :return: the parsed arguments
    """
    # Parse depending on the boolean watch flag
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--render", action="store_true", default=False)
    args: dict = vars(parser.parse_args())
    return list(args.values())



