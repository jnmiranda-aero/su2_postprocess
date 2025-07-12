
import matplotlib.pyplot as plt

def plot_cp_cf(df):
    fig1, ax1 = plt.subplots()
    ax1.plot(df['x'], df['Pressure_Coefficient'])
    ax1.set_xlabel('x')
    ax1.set_ylabel('Cp')
    ax1.set_title('Pressure Coefficient')
    ax1.invert_yaxis()
    #ax1.grid(True)

    fig2, ax2 = plt.subplots()
    ax2.plot(df['x'], df['Skin_Friction_Coefficient_x'])
    ax2.set_xlabel('x')
    ax2.set_ylabel('Cf')
    ax2.set_title('Skin Friction Coefficient')
    #ax2.grid(True)

    return fig1, fig2
