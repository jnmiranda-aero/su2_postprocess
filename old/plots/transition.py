# transition.py
import matplotlib.pyplot as plt

def plot_transition_map(df):
    if "Turb_index" not in df.columns:
        print("Warning: 'Turb_index' not found — skipping transition plot.")
        return None

    fig, ax = plt.subplots()
    ti = df["Turb_index"].clip(0, 1)
    sc = ax.scatter(df["x"], df["y"], c=ti, cmap='viridis', s=8)
    ax.set_title("Transition Indicator Map")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, label="Turbulence Index")
    return fig

