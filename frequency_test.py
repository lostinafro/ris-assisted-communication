from scenario.cluster import Cluster
from scenario.common import cart2cyl, cyl2cart, printplot, standard_output_dir, db2lin, printplot
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
import argparse
import os
from datetime import date

# Class
class TestEnvironment(Cluster):
    def __init__(self,
                 radius: float,
                 bs_position: np.array,
                 ue_position: np.array,
                 ris_num_els: int,
                 carrier_frequency: float = 1.8e9,
                 bandwidth: float = 180e3,
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


def command_parser():
    """Parse command line using arg-parse and get user data to run the render.

        :return: the parsed arguments
    """
    # Parse depending on the boolean watch flag
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--render", action="store_true", default=False)
    args: dict = vars(parser.parse_args())
    return list(args.values())

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # rendering
    render = command_parser()[0]
    output_dir = standard_output_dir('ris-channel')
    # Set parameters
    seed = 0
    N_x = 8
    rng = np.random.RandomState(seed)

    # radius of cell
    rad = 100

    # UE under tests
    users = 180
    angles = np.linspace(0, 180, users)
    ue_dist = 20 * np.ones(users)  # (1 +  rng.rand(users))
    # ue pos in cylindrical
    ue_pos = np.vstack((ue_dist, np.deg2rad(angles), np.zeros(users))).T #, [ue_dist, np.deg2rad(90), 0]])#, [ue_dist, np.deg2rad(150), 0]])

    # Varying bs positioning
    bs_angles = np.arange(45, 180, 45)
    for bs_angle in bs_angles:
        bs_pos = np.array([[100, np.deg2rad(bs_angle), 0]])
        # Build environment
        env = TestEnvironment(rad, cyl2cart(bs_pos), cyl2cart(ue_pos), N_x, rng=rng)
        # env.plot_scenario(False, 'scenario', output_dir)

        # if bs_angle == bs_angles[0]:
            # Plotting specific configurations
            # conf_angles = np.arange(30, 180, 30)
            # for conf_angle in conf_angles:
            #     title = f'Array factor having $\phi_c$ = {conf_angle} [deg]'
            #     _, ax = plt.subplots()
            #     env.set_configuration(np.deg2rad(conf_angle))
            #     env.build_channels()
            #     a = env.ris_array_factor / env.ris.num_els
            #     ax.plot(angles, np.abs(a[0]) ** 2, label=r'$\phi_c$ =' + f'{conf_angle:.0f}°')
            #     plt.ylabel(r'$|A|^2$')
            #     plt.xlabel(r'$\phi_k$')
            #     ax.grid()
            #     ax.legend()
            #     filename = f'conf_dir_angle_{conf_angle}'
            #     printplot(render, title, filename, output_dir)

        # Suffix for title and filenames
        filename_suffix = f'_bs_angle_{bs_angle}'
        title_suffix = f' ($\phi_b = {bs_angle}$°)'
        # Figures
        af_fig, af_ax = plt.subplots()
        h_ris_fig, h_ris_ax = plt.subplots(2)
        h_dir_fig, h_dir_ax = plt.subplots(2)
        h_fig, h_ax = plt.subplots()
        for i in range(env.ris.num_std_configs):
            env.set_std_configuration(i)
            h = env.build_channels()
            a = env.ris_array_factor / env.ris.num_els
            # Plots
            legend_label = r'$\phi_c \simeq' + f'{np.around(np.rad2deg(env.ris.std_config_angles[i])):.0f}$°'
            if bs_angle == bs_angles[0]:
                af_ax.plot(angles, np.abs(a[0]) ** 2, label=legend_label)
                h_ris_ax[0].plot(angles, np.abs(env.h_ris[0]) ** 2, label=legend_label)
                h_ris_ax[1].plot(angles, np.angle(env.h_ris[0]), label=legend_label)
            if i == 0:
                h_dir_ax[0].plot(angles, np.abs(env.h_dir[0]) ** 2)
                h_dir_ax[1].plot(angles, np.angle(env.h_dir[0]))
            h_ax.plot(angles, np.abs(h[0]) ** 2, label=legend_label)
            # h_ax[1].plot(angles, np.angle(h[0]), label=legend_label)

        # RIS evaluation is done only once
        if bs_angle == bs_angles[0]:
            # Plotting array factor for standard configurations
            af_file = f'array_factor'
            printplot(af_fig, af_ax, render, af_file, output_dir, title=f'Array factor using standard $\phi_c$', labels=[r'$\phi_k$', r'$|A|^2$'])
            # Plotting h_ris for standard configurations
            h_ris_file = f'ris_channel'
            printplot(h_ris_fig, h_ris_ax, render, h_ris_file, output_dir, title=f'RIS channel using standard $\phi_c$', labels=[r'$\phi_k$', r'$|g_k|^2$', r'$\angle{g_k}$'])

        # Plotting h_dir
        h_dir_file = f'dir_channel' + filename_suffix
        title = f'direct channel using standard $\phi_c$' + title_suffix
        printplot(h_dir_fig, h_dir_ax, render, h_dir_file, output_dir, title=title, labels=[r'$\phi_k$', r'$|h_k|^2$', r'$\angle{h_k}$'])

        # Plotting h for standard configurations
        h_file = f'total_channel' + filename_suffix
        title = f'RIS + direct channel using standard $\phi_c$' + title_suffix
        printplot(h_fig, h_ax, render, h_file, output_dir, title=title, labels=[r'$\phi_k$', r'$|h_k + g_k|^2$', r'$\angle{h_k + g_k}$'])

        # Plotting the whole channel for a single configuration vs frequency
        # choose users
        angle_u = np.arange(30, 175, 60)
        colors = ['b', 'g', 'r', 'c', 'm']
        rb = np.linalg.norm(env.bs.pos[0])
        hf_fig, hf_ax = plt.subplots()
        for j, ang in enumerate(angle_u):
            u = np.argmin(np.abs(ang - angles))
            ru = np.linalg.norm(env.ue.pos[u])
            rub = np.linalg.norm(env.ue.pos[u] - env.bs.pos[0])
            triangle = ru + rb - rub
            # for i in range(env.ris.num_std_configs)[:-1]:
            i = np.abs(np.rad2deg(env.ris.std_config_angles) - ang).argmin()
            legend_label = r'$\phi_c \simeq' + f'{np.around(np.rad2deg(env.ris.std_config_angles[i])):.0f}$°'
            # Set configuration
            _, varphi_x, varphi_z = env.set_std_configuration(i)
            # compute the right periodicity of the phase vs frequency
            phase_sum = 0  # - (env.ris.num_els_h + 1) / 2 * env.ris.dist_els_h * varphi_x - (env.ris.num_els_v + 1) / 2 * env.ris.dist_els_v * varphi_z
            k_min = np.floor(env.f0 / c * (phase_sum - triangle))
            k = np.arange(k_min, k_min - 400, -1)
            f = np.round(((env.f0 * phase_sum - c * k) / triangle - env.f0) / env.BW)
            h = env.build_channels()
            # Plot
            hf_ax.plot(env.freqs, np.abs(h[:, u] / env.h_dir[:, u]) ** 2, label=r'$\phi_k =' + f'{np.around(ang):.0f}$° ' + legend_label, color=colors[j], linestyle='dotted')
            hf_ax.vlines(x=env.f0 + env.BW * f[:10], ymin=0.75, ymax=1.25, linewidth=.5, linestyle='dashed', color=colors[j])

        title = f'Composite channel' + title_suffix
        hf_file = f'total_channelVSfreqs' + filename_suffix
        hf_ax.set_xlim(left=env.freqs[0], right=env.freqs[-1])
        printplot(hf_fig, hf_ax, render, hf_file, output_dir, title=title, labels=[r'$f$', r'$|h|^2 / |h_k|^2$'])



