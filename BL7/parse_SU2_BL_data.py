# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy.signal import savgol_filter

# # # Function to read data from the file
# # def read_data(file_path):
# #     data = np.loadtxt(file_path, skiprows=2)  # Skip header lines
# #     return data[:, 0], data[:, 1]  # Return x and value arrays

# # # Read data from your files
# # x_bl, bl_thickness = read_data('output/bl_thickness_delta.dat')
# # x_disp, disp_thickness = read_data('output/bl_displacement_thickness_deltaStar.dat')
# # x_mom, mom_thickness = read_data('output/bl_momentum_thickness_theta.dat')
# # x_shape, shape_factor = read_data('output/bl_shape_factor_H.dat')
# # x_edge, edge_velocity = read_data('output/bl_edge_velocity_ue.dat')
# # x, y = read_data('output/airfoil_nodes.dat')

# # # Normalize the data
# # U_inf = 1.0  # Replace with your freestream velocity if known
# # y_bl_normalized = bl_thickness / bl_thickness.max()
# # U_normalized = edge_velocity / U_inf

# # # Smoothing the data
# # window_length = 51  # Choose an odd number for the window length
# # polyorder = 3  # Polynomial order for the Savitzky-Golay filter

# # bl_thickness_smooth = savgol_filter(bl_thickness, window_length, polyorder)
# # # disp_thickness_smooth = savgol_filter(disp_thickness, window_length, polyorder)
# # # mom_thickness_smooth = savgol_filter(mom_thickness, window_length, polyorder)
# # # shape_factor_smooth = savgol_filter(shape_factor, window_length, polyorder)

# # # Create the plot
# # fig, ax = plt.subplots()  # Create figure and axes
# # ax.plot(x_bl, mom_thickness / bl_thickness.max(), label=r'Momentum Thickness, $\theta$', color='black', linestyle=':')
# # ax.plot(x_bl, disp_thickness / bl_thickness.max(), label=r'Displacement Thickness, $\delta^*$', color='black', linestyle='-')
# # ax.plot(x_bl, bl_thickness_smooth / bl_thickness.max(), label=r'Boundary Layer Thickness, $\delta$', color='black', linestyle='--')
# # ax.set_xlabel(r'x/c', fontsize=12)
# # ax.set_ylabel(r'$y/\delta_{BL_{max}}$', fontsize=12)
# # ax.set_title('Boundary Layer Parameters', fontsize=14)
# # ax.legend()
# # # ax.grid()

# # # Create inset for airfoil coordinates
# # inset_ax = fig.add_axes([0.22, 0.40, 0.2, 0.2])  # [left, bottom, width, height]
# # inset_ax.plot(x, y, color='black')  # Plot airfoil coordinates
# # inset_ax.axis('equal')  # Make inset axes equal

# # # Remove labels and title from inset
# # inset_ax.set_xticks([])  # Remove x-axis ticks
# # inset_ax.set_yticks([])  # Remove y-axis ticks
# # inset_ax.text(0.5, 0.25, r'$SE^2A$ Airfoil', fontsize=10, ha='center', va='center', transform=inset_ax.transAxes)
# # # Set the aspect ratio of the main plot to be equal
# # ax.axis('square')

# # # Use tight_layout to adjust spacing
# # plt.tight_layout()

# # profiles = []  # Replace with your actual profiles data
# # u_ue, y_deltaBL = read_data('output2/velocity_*.dat')
# # for idx, (u_profile, y_profile) in enumerate(zip(u_ue, y_deltaBL)):
# #     plt.figure()
# #     for profile in profiles:
# #         node_idx = profile['node_index']
# #         s_norm = profile['s_normalized']  # Normalized distance from the wall
# #         u_norm = profile['u_normalized']  # Normalized velocity

# #         plt.plot(u_norm, s_norm, label=f'Node {node_idx}')  # Plot each profile
    
# #     # Additional plot settings
# #     plt.xlabel('Normalized Velocity (u/u_e)', fontsize=12)
# #     plt.ylabel('Normalized Distance (s/δ)', fontsize=12)
# #     plt.title(f'Normalized Velocity Profiles at x = {idx}', fontsize=14)  # Modify title to include x location
# #     plt.legend()
# #     plt.grid(True)
# #     plt.gca().invert_yaxis()  # Optional: Invert y-axis if desired
# #     plt.tight_layout()  # Adjust layout for better fitting

# # # Show the plot
# # plt.show()
















import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import glob

# Function to read data from the file
def read_data(file_path):
    data = np.loadtxt(file_path, skiprows=2)  # Skip header lines
    return data[:, 0], data[:, 1]  # Return x and value arrays

# Read data from your files
x_bl, bl_thickness = read_data('/home/jnm8/DAAL/HPC2/SE2A_NEW/RANS/SA-LM/SA-noft2/MEDIDA-BAEDER/M0.4/Re1.6e+07/cl0.4/BL/bl_thickness_delta.dat')
x_disp, disp_thickness = read_data('/home/jnm8/DAAL/HPC2/SE2A_NEW/RANS/SA-LM/SA-noft2/MEDIDA-BAEDER/M0.4/Re1.6e+07/cl0.4/BL/bl_displacement_thickness_deltaStar.dat')
x_mom, mom_thickness = read_data('/home/jnm8/DAAL/HPC2/SE2A_NEW/RANS/SA-LM/SA-noft2/MEDIDA-BAEDER/M0.4/Re1.6e+07/cl0.4/BL/bl_momentum_thickness_theta.dat')
x_shape, shape_factor = read_data('/home/jnm8/DAAL/HPC2/SE2A_NEW/RANS/SA-LM/SA-noft2/MEDIDA-BAEDER/M0.4/Re1.6e+07/cl0.4/BL/bl_shape_factor_H.dat')
x_edge, edge_velocity = read_data('/home/jnm8/DAAL/HPC2/SE2A_NEW/RANS/SA-LM/SA-noft2/MEDIDA-BAEDER/M0.4/Re1.6e+07/cl0.4/BL/bl_edge_velocity_ue.dat')
# x, y = read_data('/home/jnm8/DAAL/HPC2/SE2A_NEW/RANS/SA-LM/SA-noft2/MEDIDA-BAEDER/M0.4/Re1.6e+07/Sensitivity/L3_v2/aoa+0.00/BL/airfoil_nodes.dat')

# Normalize the data
U_inf = 1.0  # Replace with your freestream velocity if known
y_bl_normalized = bl_thickness / bl_thickness.max()
U_normalized = edge_velocity / U_inf

# Smoothing the data
window_length = 51  # Choose an odd number for the window length
polyorder = 3  # Polynomial order for the Savitzky-Golay filter
bl_thickness_smooth = savgol_filter(bl_thickness, window_length, polyorder)

# Create the main plot for boundary layer parameters
fig, ax = plt.subplots()  # Create figure and axes
ax.plot(x_bl, mom_thickness / bl_thickness.max(), label=r'Momentum Thickness, $\theta$', color='black', linestyle=':')
ax.plot(x_bl, disp_thickness / bl_thickness.max(), label=r'Displacement Thickness, $\delta^*$', color='black', linestyle='-')
ax.plot(x_bl, bl_thickness_smooth / bl_thickness.max(), label=r'Boundary Layer Thickness, $\delta$', color='black', linestyle='--')
ax.set_xlabel(r'x/c', fontsize=12)
ax.set_ylabel(r'$y/\delta_{BL_{max}}$', fontsize=12)
ax.set_title('Boundary Layer Parameters', fontsize=14)
ax.legend()
ax.axis('square')  # Set the aspect ratio of the main plot to be equal

# Create inset for airfoil coordinates
# inset_ax = fig.add_axes([0.22, 0.40, 0.2, 0.2])  # [left, bottom, width, height]
# inset_ax.plot(x, y, color='black')  # Plot airfoil coordinates
# inset_ax.axis('equal')  # Make inset axes equal

# Remove labels and title from inset
# inset_ax.set_xticks([])  # Remove x-axis ticks
# inset_ax.set_yticks([])  # Remove y-axis ticks
# inset_ax.text(0.5, 0.2, r'$SE^2A$ Airfoil', fontsize=10, ha='center', va='center', transform=inset_ax.transAxes)

# Show the boundary layer plot
plt.tight_layout()  # Adjust layout for better fitting
# plt.show()

# Function to read data from the file
# Function to read data from the file
def read_data(file_path):
    data = np.loadtxt(file_path, skiprows=2)  # Skip header lines
    print(f"Successfully read {file_path}: {data.shape} (should be 2 columns)")
    return data[:, 0], data[:, 1]  # Return normalized velocity and normalized distance arrays

# Create a new figure for velocity profiles
velocity_files = glob.glob('output2/velocity_profile_x_*.dat')  # Use glob to find all relevant files

# Check if any files were found
if not velocity_files:
    print("No velocity files found matching the pattern 'output2/velocity_profile_x_*.dat'.")
else:
    profiles = []

    # Load and store normalized velocity profiles from files
    for file_path in velocity_files:
        u_ue, y_deltaBL = read_data(file_path)  # Load each velocity profile file

        # Normalize y values
        y_deltaBL_normalized = y_deltaBL # Normalize distance values

        # Append profiles with their corresponding normalized values
        profiles.append((u_ue, y_deltaBL_normalized))

# Create subplots
num_profiles = len(profiles)
n_rows = 2
n_cols = (num_profiles + 1) // n_rows  # Calculate number of columns needed

# Create the figure with subplots
fig, axs = plt.subplots(n_rows, n_cols)  # Create a subplot grid

# Flatten the axs array for easy indexing
axs = axs.flatten()

# Plot each normalized velocity profile in a separate subplot
for idx, (u_profile, y_profile) in enumerate(profiles):
    axs[idx].plot(y_profile, u_profile)#, label=f'Profile from {velocity_files[idx].split("/")[-1]}')  # Use filename for label
    axs[idx].set_xlabel('u/u_e', fontsize=12)
    axs[idx].set_ylabel('y/δ', fontsize=12)
    # axs[idx].set_title(f'Normalized Velocity Profile {idx + 1}', fontsize=14)
    # axs[idx].grid(True)
    # axs[idx].invert_yaxis()  # Invert y-axis if desired
    # axs[idx].invert_xaxis()  # Invert y-axis if desired
    axs[idx].legend()
    axs[idx].axis('square')  # Set the aspect ratio of the main plot to be equal


# Hide any empty subplots
for j in range(num_profiles, n_rows * n_cols):
    fig.delaxes(axs[j])  # Remove unused axes

# plt.tight_layout()  # Adjust layout for better fitting

# Show the velocity profiles plot
plt.show()

################################################################



# #!/usr/bin/env python3
# """
# parse_SU2_BL_data.py

# Integrates boundary-layer extraction (MeshParser) with multi-case comparison plotting.
# Regenerates BL/*.dat via MeshParser and then plots δ, δ*, θ, and velocity profiles.
# """
# import os
# import glob
# import argparse
# import logging

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter
# from scipy.optimize import brentq
# from scipy.interpolate import NearestNDInterpolator
# from scipy.spatial import KDTree
# from joblib import Parallel, delayed
# from collections import defaultdict

# # Logging configuration for MeshParser
# logging.basicConfig(
#     filename='mesh_parser_debug.log',
#     filemode='a',  # append
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     level=logging.DEBUG
# )

# def compute_bl_thickness_for_node(args):
#     i, surface_node, normal, points, V_mag, Vorticity, methods, threshold, max_steps, step_size, verbose = args
#     try:
#         vel_interp = NearestNDInterpolator(points, V_mag)
#         vort_interp = NearestNDInterpolator(points, Vorticity)

#         bl_t = {}
#         disp_t = {}
#         mom_t = {}
#         H_t   = {}
#         ue_t  = {}

#         V_s = vel_interp(surface_node)
#         vort_s = vort_interp(surface_node)
#         if np.isnan(V_s):
#             for m in methods:
#                 bl_t[m]=disp_t[m]=mom_t[m]=H_t[m]=ue_t[m]=np.nan
#             return (i, bl_t, disp_t, mom_t, H_t, ue_t)

#         for method in methods:
#             if method == 'edge_velocity':
#                 def f(s):
#                     p0 = surface_node + normal*s
#                     p1 = surface_node + normal*(s+step_size)
#                     v0 = vel_interp(p0); v1 = vel_interp(p1)
#                     if np.isnan(v0) or np.isnan(v1): return 1e6
#                     return v0 - threshold*v1
#                 a,b = 0.0, 0.5
#                 fa, fb = f(a), f(b)
#                 if fa*fb>0:
#                     bl_t[method]=disp_t[method]=mom_t[method]=H_t[method]=ue_t[method]=np.nan
#                     continue
#                 s_root = brentq(f, a, b)
#                 bl_t[method] = s_root
#                 Ue = vel_interp(surface_node + normal*s_root)
#                 ue_t[method] = Ue
#                 if s_root<=1e-6:
#                     disp_t[method]=mom_t[method]=H_t[method]=np.nan
#                     continue
#                 # integrate
#                 ss = np.linspace(0, s_root, 500)
#                 int_disp = [(1 - vel_interp(surface_node + normal*s)/Ue) if Ue else 0 for s in ss]
#                 int_mom  = [((vel_interp(surface_node + normal*s)/Ue)*(1 - vel_interp(surface_node + normal*s)/Ue)) if Ue else 0 for s in ss]
#                 disp_val = np.trapz(int_disp, ss)
#                 mom_val  = np.trapz(int_mom, ss)
#                 disp_t[method]=disp_val
#                 mom_t[method]=mom_val
#                 H_t[method]  = (disp_val/mom_val) if mom_val else np.nan
#             elif method == 'vorticity_threshold':
#                 s=0; found=False
#                 for _ in range(int(max_steps)):
#                     s += step_size
#                     vort_c = vort_interp(surface_node + normal*s)
#                     if np.isnan(vort_c) or np.isnan(vort_s): break
#                     if abs(vort_c/vort_s)<=1e-4:
#                         bl_t[method]=s
#                         Ue=vel_interp(surface_node + normal*s)
#                         ue_t[method]=Ue
#                         found=True; break
#                 if not found or s<=1e-6:
#                     bl_t[method]=disp_t[method]=mom_t[method]=H_t[method]=ue_t[method]=np.nan
#                     continue
#                 ss = np.linspace(0, s, 500)
#                 int_disp = [(1 - vel_interp(surface_node + normal*s0)/ue_t[method]) for s0 in ss]
#                 int_mom  = [((vel_interp(surface_node + normal*s0)/ue_t[method])*(1 - vel_interp(surface_node + normal*s0)/ue_t[method])) for s0 in ss]
#                 disp_val = np.trapz(int_disp, ss)
#                 mom_val  = np.trapz(int_mom, ss)
#                 disp_t[method]=disp_val
#                 mom_t[method]=mom_val
#                 H_t[method]  = (disp_val/mom_val) if mom_val else np.nan
#             else:
#                 bl_t[method]=disp_t[method]=mom_t[method]=H_t[method]=ue_t[method]=np.nan
#         return (i, bl_t, disp_t, mom_t, H_t, ue_t)
#     except Exception as e:
#         logging.error(f"Node {i} error: {e}")
#         return (i, {},{}, {},{}, {})

# class MeshParser:
#     def __init__(self, surface_file, flow_file, output_dir='BL', x_locations=None):
#         self.surface_file = surface_file
#         self.flow_file    = flow_file
#         self.output_dir   = output_dir
#         self.x_locations  = x_locations or []
#         os.makedirs(self.output_dir, exist_ok=True)

#     def load_su2(self, path, reorder=True):
#         # parse TECPLOT-like .dat into data dict, nodes, elements, neighbors
#         lines = open(path,'r').read().splitlines()
#         headers = [h.strip() for h in lines[1].split('=')[1].replace('"','').split(',')]
#         nc = int(lines[2].split('NODES=')[1].split(',')[0])
#         ec = int(lines[2].split('ELEMENTS=')[1].split(',')[0])
#         zt = lines[2].split('ZONETYPE=')[1].strip()
#         data, nodes, elems, nbrs = {h:[] for h in headers},[],[],defaultdict(set)
#         idx=3
#         for L in lines[idx:idx+nc]:
#             vals=L.split()
#             if len(vals)!=len(headers): continue
#             for i,h in enumerate(headers): data[h].append(float(vals[i]))
#             nodes.append([float(vals[headers.index('x')]), float(vals[headers.index('y')])])
#         idx+=nc
#         for L in lines[idx:idx+ec]:
#             inds=[int(x)-1 for x in L.split()]
#             elems.append(inds)
#             for i in inds:
#                 for j in inds:
#                     if i!=j: nbrs[i].add(j)
#         nodes=np.array(nodes)
#         return {h:np.array(v) for h,v in data.items()}, np.array(elems), nodes, nbrs, zt

#     def compute_normals(self):
#         # reorder surface by FELINESEG connectivity
#         data_s, elems_s, nodes_s, nbrs_s, zt_s = self.load_su2(self.surface_file)
#         order=[elems_s[0][0]]
#         while len(order)<len(nodes_s):
#             curr=order[-1]
#             for e in elems_s:
#                 if curr in e:
#                     nxt=e[1] if e[0]==curr else e[0]
#                     if nxt not in order:
#                         order.append(nxt)
#                         break
#             else: break
#         self.surface_nodes = nodes_s[order]
#         # compute normals
#         nlist=[]
#         for i in range(len(self.surface_nodes)-1):
#             t=self.surface_nodes[i+1]-self.surface_nodes[i]
#             n=np.array([-t[1],t[0]]);
#             norm=np.linalg.norm(n)
#             nlist.append(n/norm if norm else n)
#         nlist.append(nlist[-1]); self.surface_normals=np.array(nlist)
#         # flow data
#         self.flow_data, elems_v, self.flow_nodes, _, _ = self.load_su2(self.flow_file, reorder=False)

#     def run(self, methods=['vorticity_threshold'], threshold=0.99, max_steps=1e6, step_size=1e-7, n_jobs=-1, verbose=False):
#         self.compute_normals()
#         X = self.flow_data['x']; Y=self.flow_data['y']
#         U = self.flow_data.get('Velocity_x'); V=self.flow_data.get('Velocity_y')
#         vort=self.flow_data.get('Vorticity')
#         pmag=np.sqrt(U**2+V**2)
#         pts=np.column_stack((X,Y))
#         valid=~np.isnan(pmag)&~np.isnan(vort)
#         pts,pmag,vort=pts[valid],pmag[valid],vort[valid]
#         args=[(i,node,n,pts,pmag,vort,methods,threshold,max_steps,step_size,verbose)
#               for i,(node,n) in enumerate(zip(self.surface_nodes,self.surface_normals))]
#         out=Parallel(n_jobs=n_jobs)(delayed(compute_bl_thickness_for_node)(a) for a in args)
#         out=sorted(out,key=lambda x:x[0])
#         # collect
#         for m in methods:
#             setattr(self, f'bl_{m}', [r[1].get(m,np.nan) for r in out])
#             setattr(self, f'disp_{m}',[r[2].get(m,np.nan) for r in out])
#             setattr(self, f'mom_{m}', [r[3].get(m,np.nan) for r in out])
#             setattr(self, f'H_{m}',   [r[4].get(m,np.nan) for r in out])
#             setattr(self, f'ue_{m}',  [r[5].get(m,np.nan) for r in out])
#         # save to BL/
#         xcoords=self.surface_nodes[:,0]
#         for m in methods:
#             self._save_dat('bl_thickness_delta',      xcoords, getattr(self,f'bl_{m}'))
#             self._save_dat('bl_displacement_deltaStar',xcoords, getattr(self,f'disp_{m}'))
#             self._save_dat('bl_momentum_theta',       xcoords, getattr(self,f'mom_{m}'))
#             self._save_dat('bl_shape_H',              xcoords, getattr(self,f'H_{m}'))
#             self._save_dat('bl_edge_ue',              xcoords, getattr(self,f'ue_{m}'))
#         # extract velocity profiles
#         self.velocity_profiles=defaultdict(list)
#         interp=NearestNDInterpolator(pts,pmag)
#         for xloc in self.x_locations:
#             idx=np.argmin(np.abs(self.surface_nodes[:,0]-xloc))
#             smax=self.bl_vorticity_threshold[idx]
#             Ue=self.ue_vorticity_threshold[idx]
#             ss=np.linspace(0,smax,100)
#             pos=self.surface_nodes[idx]+np.outer(ss,self.surface_normals[idx])
#             vel=interp(pos)
#             valid=~np.isnan(vel)
#             ss,vel=ss[valid],vel[valid]
#             ssn,un=ss/smax,vel/Ue
#             un_s=savgol_filter(un,11,3)
#             self.velocity_profiles[xloc].append((ssn,un_s,idx))
#         # save profiles
#         for xloc,pl in self.velocity_profiles.items():
#             for ssn,un,node in pl:
#                 fname=f"velocity_profile_x_{xloc:.4f}_node_{node}.dat"
#                 self._save_dat(fname, ssn, un)

#     def _save_dat(self,fname,x,y):
#         out=os.path.join(self.output_dir,fname+'.dat')
#         with open(out,'w') as f:
#             f.write('VARIABLES = "x","value"\n')
#             f.write('ZONE T="BL Data"\n')
#             for xi,yi in zip(x,y): f.write(f"{xi:.6e} {yi:.6e}\n")

# # Helper I/O and plotting routines

# def read_dat(path, skiprows=2):
#     data=np.loadtxt(path,skiprows=skiprows)
#     return data[:,0], data[:,1]

# def gather_bl_data(case_dir):
#     bl_dir=os.path.join(case_dir,'BL')
#     keys=['bl_thickness_delta','bl_displacement_deltaStar','bl_momentum_theta','bl_shape_H','bl_edge_ue']
#     out={}
#     for k in keys:
#         p=os.path.join(bl_dir,k+'.dat')
#         if not os.path.isfile(p): raise FileNotFoundError(p)
#         out[k]=read_dat(p)
#     return out

# def gather_velocity_profiles(case_dir):
#     files=sorted(glob.glob(os.path.join(case_dir,'BL','velocity_profile_x_*.dat')))
#     profiles=[]
#     for f in files:
#         u,y=read_dat(f)
#         lbl=os.path.basename(f).replace('.dat','')
#         profiles.append((u,y,lbl))
#     return profiles

# def plot_bl_parameters(cases_data, labels):
#     plt.figure(figsize=(8,6))
#     for data,lbl in zip(cases_data,labels):
#         x,δ=data['bl_thickness_delta']
#         _,δs=data['bl_displacement_deltaStar']
#         _,θ =data['bl_momentum_theta']
#         dmax=δ.max()
#         plt.plot(x,θ/dmax,':',label=f"{lbl} θ")
#         plt.plot(x,δs/dmax,'-',label=f"{lbl} δ*")
#         plt.plot(x,δ/dmax,'--',label=f"{lbl} δ")
#     plt.xlabel('x/c'); plt.ylabel('y/δ_max')
#     plt.legend(fontsize='small'); plt.axis('square'); plt.grid() ; plt.tight_layout(); plt.show()

# def plot_velocity_profiles(all_profiles, labels):
#     for profs,lbl in zip(all_profiles,labels):
#         n=len(profs)
#         if not n: continue
#         fig,axs=plt.subplots(1,n,figsize=(4*n,4),squeeze=False)[0]
#         for ax,(u,y,l) in zip(axs,profs):
#             ax.plot(u,y); ax.set_xlabel('u/u_e'); ax.set_ylabel('y/δ'); ax.set_title(f"{lbl} @ {l}"); ax.axis('square'); ax.grid()
#         plt.tight_layout(); plt.show()

# def build_case_list(args):
#     if args.cases:
#         dirs=[]; labels=[]
#         for c in args.cases:
#             d=c if os.path.isabs(c) else os.path.join(args.root,c)
#             dirs.append(d); labels.append(os.path.basename(d))
#     else:
#         dirs=[os.path.join(args.root,d) for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root,d))]
#         labels=[os.path.basename(d) for d in dirs]
#     return dirs, labels

# # Main entry
# if __name__ == '__main__':
#     p=argparse.ArgumentParser()
#     p.add_argument('--root')
#     p.add_argument('--cases',nargs='+')
#     p.add_argument('--x-locs',nargs='+',type=float,default=[0.1,0.4,0.65,0.9])
#     p.add_argument('--compute-bl',action='store_true')
#     p.add_argument('--methods',nargs='+',default=['vorticity_threshold'])
#     p.add_argument('--jobs',type=int,default=-1)
#     p.add_argument('--step',type=float,default=1e-7)
#     args=p.parse_args()
#     case_dirs,labels=build_case_list(args)
#     bl_list, vp_list=[],[]
#     for d in case_dirs:
#         if args.compute_bl:
#             mp=MeshParser(os.path.join(d,'flow_surf_.dat'), os.path.join(d,'flow_vol_.dat'), output_dir=os.path.join(d,'BL'), x_locations=args.x_locs)
#             mp.run(methods=args.methods, step_size=args.step, n_jobs=args.jobs, verbose=True)
#         try:
#             bl_list.append(gather_bl_data(d))
#             vp_list.append(gather_velocity_profiles(d))
#         except Exception as e:
#             print(f"Skip {d}: {e}")
#     if bl_list: plot_bl_parameters(bl_list, labels)
#     if vp_list: plot_velocity_profiles(vp_list, labels)









