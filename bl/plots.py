import matplotlib.pyplot as plt

def _plot_profiles(results, methods, legends: bool):
    # Simplest hook: plot normalized velocity profiles for requested x_locs
    for res in results:
        for x_loc, profiles in res.velocity_profiles.items():
            plt.figure()
            for prof in profiles:
                plt.plot(prof['u_normalized'], prof['s_normalized'],
                         label=f"x={x_loc:.3f}, node={prof['node_index']}")
            plt.xlabel("u / u_e")
            plt.ylabel("s / δ")
            plt.title(f"Velocity profiles at x = {x_loc:.3f}")
            if legends:
                plt.legend()
    plt.show()

def plot_bl_params_multi(results, methods):
    # Simple multi-case overlay of BL thickness
    plt.figure()
    for res in results:
        for method in methods:
            plt.plot(res.x, res.delta[method], label=f"{res.x[:3]}… ({method})")
    plt.xlabel("x")
    plt.ylabel("δ")
    plt.legend()
    plt.show()

def plot_velocity_profiles_multi(results, methods):
    # No‐op or similar to _plot_profiles
    _plot_profiles(results, methods, legends=True)
