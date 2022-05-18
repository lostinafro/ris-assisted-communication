from scenario.cluster import Cluster
from scenario.common import cart2cyl, cyl2cart, printplot, standard_output_dir
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import date

# Class
class TestEnvironment(Cluster):
    def __init__(self,
                 radius: float,
                 bs_position: np.array,
                 ue_position: np.array,
                 ris_num_els: int,
                 carrier_frequency: float = 3e9,
                 bandwidth: float = 180e3,
                 symbols: int = 1,
                 rng: np.random.RandomState = None):
        # Init parent class
        super().__init__(shape='semicircle',
                         x_size=radius,
                         carrier_frequency=carrier_frequency,
                         bandwidth=bandwidth,
                         direct_channel='LoS',
                         reflective_channel='LoS',
                         rng=rng)
        self.place_bs(1, bs_position)
        self.place_ue(ue_position.shape[0], ue_position)
        self.place_ris(1, np.array([[0, 0, 0]]), num_els_x=ris_num_els, dist_els_x=self.wavelength/2)

    def set_std_configuration(self, index: int):
        return self.ris.set_std_configuration(self.wavenumber, index, self.bs.pos)

    def set_configuration(self, angle: float):
        return self.ris.set_configuration(self.wavenumber, angle, self.bs.pos)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # rendering
    render = False
    output_dir = standard_output_dir('ris-channel')
    # Set parameters
    seed = 0
    N_x = 8
    ue_dist = 20
    # radius of cell
    rad = 50
    # Angles
    users = 3
    angles = np.linspace(25, 155, users)
    # ue pos in cylindrical
    ue_pos = np.vstack((np.repeat(ue_dist, users), np.deg2rad(angles), np.zeros(users))).T #, [ue_dist, np.deg2rad(90), 0]])#, [ue_dist, np.deg2rad(150), 0]])


    # Varying bs positioning
    bs_angles = [90] # np.arange(45, 180, 45)
    for bs_angle in bs_angles:
        bs_pos = np.array([[45, np.deg2rad(bs_angle), 0]])
        # Build environment
        env = TestEnvironment(rad, cyl2cart(bs_pos), cyl2cart(ue_pos), N_x, rng=np.random.RandomState(seed))
        # env.plot_scenario(False, 'scenario', output_dir)

        # RIS evaluation can be done only once
        if bs_angle == bs_angles[0]:

            # Plotting specific configurations
            conf_angles = np.arange(30, 180, 30)
            for conf_angle in conf_angles:
                _, ax = plt.subplots()
                env.set_configuration(np.deg2rad(conf_angle))
                env.build_channels()
                a = env.ris_array_factor / env.ris.num_els
                ax.plot(angles, np.abs(a) ** 2, label=r'$\phi_c$ =' + f'{conf_angle:.0f}°')
                plt.ylabel(r'$|A|^2$')
                plt.xlabel(r'$\phi_k$')
                ax.grid()
                ax.legend()
                title = f'Array factor having $\phi_c$ = {conf_angle} [deg]'
                filename = f'conf_dir_angle_{conf_angle}'
                printplot(render, title, filename, output_dir)

            # Plotting array factor for standard configurations
            _, ax = plt.subplots()
            for i in range(env.ris.num_std_configs)[:-1]:
                env.set_std_configuration(i)
                h = env.build_channels()
                a = env.ris_array_factor / env.ris.num_els
                ax.plot(angles, np.abs(a) ** 2, label=r'$\phi_c$ =' + f'{np.around(np.rad2deg(env.ris.std_config_angles[i])):.0f}°')
            plt.ylabel(r'$|A|^2$')
            plt.xlabel(r'$\phi_k$')
            ax.grid()
            ax.legend()
            title = f'Array factor using standard $\phi_c$'
            filename = f'array_factor'
            printplot(render, title, filename, output_dir)

            # Plotting h_ris for standard configurations
            _, ax = plt.subplots()
            for i in range(env.ris.num_std_configs)[:-1]:
                env.set_std_configuration(i)
                h = env.build_channels()
                ax.plot(angles, np.abs(env.h_ris) ** 2, label=r'$\phi_c$ =' + f'{np.around(np.rad2deg(env.ris.std_config_angles[i])):.0f}°')
            plt.ylabel(r'$|h_{ris}|^2$')
            plt.xlabel(r'$\phi_k$')
            ax.grid()
            ax.legend()
            title = f'RIS channel using standard $\phi_c$'
            filename = f'ris_channel'
            printplot(render, title, filename, output_dir)

        # Suffix for title and filenames
        filename_suffix = f'_bs_angle_{bs_angle}'
        title_suffix = f' ($\phi_b$ = {bs_angle} [deg])'
        # Plotting h_ris + h_dir for standard configurations
        _, ax = plt.subplots()
        for i in range(env.ris.num_std_configs)[:-1]:
            env.set_std_configuration(i)
            h = env.build_channels()
            ax.plot(angles, np.abs(env.h_dir + env.h_ris) ** 2, label=r'$\phi_c$ =' + f'{np.around(np.rad2deg(env.ris.std_config_angles[i])):.0f}°')
        plt.ylabel(r'$|h_{ris} + h_{dir}|^2$')
        plt.xlabel(r'$\phi_k$')
        ax.grid()
        ax.legend()
        title = f'RIS + direct channel using standard $\phi_c$' + title_suffix
        filename = f'total_channel' + filename_suffix
        printplot(render, title, filename, output_dir)


        # Plotting h_dir for standard configurations
        _, ax = plt.subplots()
        for i in range(env.ris.num_std_configs)[:1]:
            env.set_std_configuration(i)
            h = env.build_channels()
            ax.plot(angles, np.abs(env.h_dir) ** 2)
        plt.ylabel(r'$|h_{ris} + h_{dir}|^2$')
        plt.xlabel(r'$\phi_k$')
        ax.grid()
        # ax.legend()
        title = f'direct channel using standard $\phi_c$' + title_suffix
        filename = f'dir_channel' + filename_suffix
        printplot(render, title, filename, output_dir)
