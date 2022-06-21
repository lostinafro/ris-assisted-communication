from environment import RISEnvironment2D, command_parser, OUTPUT_DIR
import scenario.common as cmn
import matplotlib.pyplot as plt
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # rendering
    render = command_parser()[0]
    prefix = 'varPhi_'
    # Set parameters
    seed = 0
    rng = np.random.RandomState(seed)

    # radius of cell
    rad = 100

    # UE under tests
    users = 180
    angles = np.linspace(0, 180, users)
    ue_dist = 15 * np.ones(users)  # (1 +  rng.rand(users))
    # ue pos in cylindrical
    ue_pos = np.vstack((ue_dist, np.deg2rad(angles), np.zeros(users))).T #, [ue_dist, np.deg2rad(90), 0]])#, [ue_dist, np.deg2rad(150), 0]])

    # Varying bs positioning
    bs_angles = np.arange(45, 180, 45)
    for bs_angle in bs_angles:
        bs_pos = np.array([[100, np.deg2rad(bs_angle), 0]])
        # Build environment
        env = RISEnvironment2D(rad, cmn.cyl2cart(bs_pos), cmn.cyl2cart(ue_pos), rng=rng)
        env.plot_scenario(False, 'scenario', OUTPUT_DIR)
        best_configurations = env.best_configuration_selection()
        best_frequencies = env.best_frequency_selection()

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
            af_file = prefix + f'array_factor'
            cmn.printplot(af_fig, af_ax, render, af_file, OUTPUT_DIR, title=f'Array factor using standard $\phi_c$', labels=[r'$\phi_k$ [deg]', r'$|A|^2 / N^2$'])
            # Plotting h_ris for standard configurations
            h_ris_file = prefix + f'ris_channel'
            cmn.printplot(h_ris_fig, h_ris_ax, render, h_ris_file, OUTPUT_DIR, title=f'RIS channel using standard $\phi_c$', labels=[r'$\phi_k$ [deg]', r'$|g_k|^2$', r'$\angle{g_k}$'])

        # Plotting h_dir
        h_dir_file = prefix + f'dir_channel' + filename_suffix
        title = f'direct channel using standard $\phi_c$' + title_suffix
        cmn.printplot(h_dir_fig, h_dir_ax, render, h_dir_file, OUTPUT_DIR, title=title, labels=[r'$\phi_k$ [deg]', r'$|h_k|^2$', r'$\angle{h_k}$'])

        # Plotting h for standard configurations
        h_file = prefix + f'total_channel' + filename_suffix
        title = f'RIS + direct channel using standard $\phi_c$' + title_suffix
        cmn.printplot(h_fig, h_ax, render, h_file, OUTPUT_DIR, title=title, labels=[r'$\phi_k$ [deg]', r'$|h_k + g_k|^2$', r'$\angle{h_k + g_k}$'])

        # Plotting the whole channel for a single configuration vs frequency
        # choose users
        angle_u = np.arange(30, 175, 60)
        colors = ['b', 'g', 'r', 'c', 'm']
        rb = env.bs.pos.norm[0]
        hf_fig, hf_ax = plt.subplots()
        for j, ang in enumerate(angle_u):
            u = np.argmin(np.abs(ang - angles))
            legend_label = r'$\phi_c \simeq' + f'{np.around(np.rad2deg(env.ris.std_config_angles[best_configurations[u]])):.0f}$°'
            f = best_frequencies[u]
            env.set_std_configuration(best_configurations[u])
            h = env.build_channels()
            # Plot
            hf_ax.plot(env.freqs, np.abs(h[:, u] / env.h_dir[:, u]) ** 2, label=r'$\phi_k =' + f'{np.around(ang):.0f}$° ' + legend_label, color=colors[j])
            hf_ax.vlines(x=env.f0 + env.BW * f[:10], ymin=0.75, ymax=1.25, linewidth=.5, linestyle='dotted', color=colors[j])

        title = f'Composite channel' + title_suffix
        hf_file = prefix + f'total_channelVSfreqs' + filename_suffix
        hf_ax.set_xlim(left=env.freqs[0], right=env.freqs[-1])
        cmn.printplot(hf_fig, hf_ax, render, hf_file, OUTPUT_DIR, title=title, labels=[r'$f$', r'$|h|^2 / |h_k|^2$'])



