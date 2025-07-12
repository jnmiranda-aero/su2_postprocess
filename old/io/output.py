from pathlib import Path

def save_plot(fig, directory, name, format=['svg','png'], dpi=300):
    """
    Save a matplotlib figure to the specified directory with given format and DPI.
    """
    directory = Path(directory) / "plots"
    directory.mkdir(parents=True, exist_ok=True)

    filename = directory / f"{name}.{format}"
    fig.savefig(filename, dpi=dpi, format=format, bbox_inches="tight")
    print(f"[INFO] Saved: {filename}")



# import matplotlib.pyplot as plt

# def save_plot(fig, case_path, name):
#     out_dir = case_path / "plots"
#     out_dir.mkdir(exist_ok=True, parents=True)
#     for ext in ['.png', '.svg']:
#         fig.savefig(out_dir / f"{name}{ext}", bbox_inches='tight', dpi=600)
#     plt.close(fig)
