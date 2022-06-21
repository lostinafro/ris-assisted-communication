from environment import RISEnvironment2D, command_parser, OUTPUT_DIR, NUM_EL_X, CARRIER_FREQ
import scenario.common as cmn
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # rendering
    render = command_parser()[0]
    prefix = 'varRho_'
    # Set parameters
    seed = 0
    rng = np.random.RandomState(seed)

    # radius of cell
    rad = 100
    rad_min = np.ceil(NUM_EL_X ** 2 / 2 * c / CARRIER_FREQ)  # ff distance
    # UE under tests
    users = 80
    angles = 90 * np.ones(users)
    ue_dist = np.linspace(rad_min, rad/2, users)
    # ue pos in cylindrical
    ue_pos = np.vstack((ue_dist, np.deg2rad(angles), np.zeros(users))).T #, [ue_dist, np.deg2rad(90), 0]])#, [ue_dist, np.deg2rad(150), 0]])

    # Varying bs positioning
    bs_angles = np.arange(45, 180, 45)
    for bs_angle in bs_angles:
        bs_pos = np.array([[100, np.deg2rad(bs_angle), 0]])
        # Build environment
        env = RISEnvironment2D(rad, cmn.cyl2cart(bs_pos), cmn.cyl2cart(ue_pos), rng=rng)
        # env.plot_scenario(False, 'scenario', OUTPUT_DIR)
        best_configurations = env.best_configuration_selection()
        best_frequencies = env.best_frequency_selection()

        # Suffix for title and filenames
        filename_suffix = f'_bs_angle_{bs_angle}'
        title_suffix = f' ($\phi_b = {bs_angle}$°)'
        # Figures
        h_ris_fig, h_ris_ax = plt.subplots(2)
        h_dir_fig, h_dir_ax = plt.subplots(2)
        h_fig, h_ax = plt.subplots()
        # Best configuration is the same for all the users
        env.set_std_configuration(best_configurations[0])
        h = env.build_channels()
        # a = env.ris_array_factor / env.ris.num_els
        # Plots
        conf_label = r'$\phi_c \simeq' + f'{np.around(np.rad2deg(env.ris.std_config_angles[best_configurations[0]])):.0f}$°'
        if bs_angle == bs_angles[0]:
            h_ris_ax[0].plot(ue_dist, np.abs(env.h_ris[0]) ** 2)
            h_ris_ax[1].plot(ue_dist, np.angle(env.h_ris[0]))
        h_dir_ax[0].plot(ue_dist, np.abs(env.h_dir[0]) ** 2)
        h_dir_ax[1].plot(ue_dist, np.angle(env.h_dir[0]))
        h_ax.plot(ue_dist, np.abs(h[0]) ** 2)
        # h_ax[1].plot(angles, np.angle(h[0]), label=legend_label)

        # RIS evaluation is done only once
        if bs_angle == bs_angles[0]:
            # Plotting h_ris for standard configurations
            h_ris_file = prefix + f'ris_channel'
            cmn.printplot(h_ris_fig, h_ris_ax, render, h_ris_file, OUTPUT_DIR, title=f'RIS channel using ' + conf_label, labels=[r'$r_k$ [m]', r'$|g_k|^2$', r'$\angle{g_k}$'])

        # Plotting h_dir
        h_dir_file = prefix + f'dir_channel' + filename_suffix
        title = f'direct channel using using ' + conf_label + title_suffix
        cmn.printplot(h_dir_fig, h_dir_ax, render, h_dir_file, OUTPUT_DIR, title=title, labels=[r'$r_k$ [m]', r'$|h_k|^2$', r'$\angle{h_k}$'])

        # Plotting h for standard configurations
        h_file = prefix + f'total_channel' + filename_suffix
        title = f'RIS + direct channel using ' + conf_label + title_suffix
        cmn.printplot(h_fig, h_ax, render, h_file, OUTPUT_DIR, title=title, labels=[r'$r_k$ [m]', r'$|h_k + g_k|^2$', r'$\angle{h_k + g_k}$'])

        # Plotting the whole channel for a single configuration vs frequency
        # choose users
        rho_u = np.linspace(ue_dist[0], ue_dist[-1], 5)
        colors = ['b', 'g', 'r', 'c', 'm']
        rb = env.bs.pos.norm[0]
        hf_fig, hf_ax = plt.subplots()
        for j, rho in enumerate(rho_u):
            u = np.argmin(np.abs(rho - ue_dist))
            f = best_frequencies[u]
            env.set_std_configuration(best_configurations[u])
            h = env.build_channels()
            # Plot
            hf_ax.plot(env.freqs, np.abs(h[:, u] / env.h_dir[:, u]) ** 2, label=r'$r_k =' + f'{np.around(rho):.0f}$ [m]', color=colors[j])
            hf_ax.vlines(x=env.f0 + env.BW * f[:10], ymin=0.4, ymax=2, linewidth=.5, linestyle='dotted', color=colors[j])

        title = f'Composite channel' + title_suffix
        hf_file = prefix + f'total_channelVSfreqs' + filename_suffix
        hf_ax.set_xlim(left=env.freqs[0], right=env.freqs[-1])
        cmn.printplot(hf_fig, hf_ax, render, hf_file, OUTPUT_DIR, title=title, labels=[r'$f$', r'$|h|^2 / |h_k|^2$'])



