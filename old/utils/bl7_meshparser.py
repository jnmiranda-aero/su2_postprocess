import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.optimize import brentq
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import KDTree
from scipy.signal import savgol_filter  # Import Savitzky-Golay filter
from joblib import Parallel, delayed
from collections import defaultdict

# Set up logging for debugging
logging.basicConfig(
    filename='mesh_parser_debug.log',
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # Capture all levels
)

def compute_bl_thickness_for_node(args):
    i, surface_node, normal, points, V_mag, Vorticity, methods, threshold, max_steps, step_size, verbose = args

    try:
        # Build the interpolators
        vel_interpolator = NearestNDInterpolator(points, V_mag)
        vorticity_interpolator = NearestNDInterpolator(points, Vorticity)

        bl_thicknesses = {}
        displacement_thicknesses = {}
        momentum_thicknesses = {}
        shape_factors = {}
        edge_velocities = {}

        # Initialize variables
        V_surface = vel_interpolator(surface_node)
        vorticity_surface = vorticity_interpolator(surface_node)

        if verbose:
            logging.debug(f"Node {i}: V_surface={V_surface}, vorticity_surface={vorticity_surface}")

        if np.isnan(V_surface):
            # Velocity at surface node is NaN
            logging.warning(f"Node {i}: V_surface is NaN.")
            for method in methods:
                bl_thicknesses[method] = np.nan
                displacement_thicknesses[method] = np.nan
                momentum_thicknesses[method] = np.nan
                shape_factors[method] = np.nan
                edge_velocities[method] = np.nan
            return (i, bl_thicknesses, displacement_thicknesses, momentum_thicknesses, shape_factors, edge_velocities)
        elif V_surface == 0:
            # Velocity at surface node is zero
            logging.info(f"Node {i}: V_surface is zero.")
            # Decide whether to proceed or skip based on physical context
            # For now, proceeding with computations

        for method in methods:
            if method == 'edge_velocity':
                # Define the function for root finding
                def func(s):
                    current_pos = surface_node + normal * s
                    next_pos = surface_node + normal * (s + step_size)
                    V_current = vel_interpolator(current_pos)
                    V_next = vel_interpolator(next_pos)

                    if np.isnan(V_current) or np.isnan(V_next):
                        return 1e6  # Out of bounds

                    return V_current - threshold * V_next

                # Define the search interval
                a = 0.0
                b = 0.5  # Adjust as needed

                if verbose:
                    logging.debug(f"Node {i} ({method}): Starting brentq search between a={a} and b={b}.")

                try:
                    # Check if func(a) and func(b) have opposite signs
                    fa = func(a)
                    fb = func(b)
                    if fa * fb > 0:
                        logging.warning(f"Node {i} ({method}): brentq requires func(a) and func(b) to have opposite signs. fa={fa}, fb={fb}.")
                        bl_thicknesses[method] = np.nan
                        displacement_thicknesses[method] = np.nan
                        momentum_thicknesses[method] = np.nan
                        shape_factors[method] = np.nan
                        edge_velocities[method] = np.nan
                        continue

                    s_root = brentq(func, a, b)
                    bl_thicknesses[method] = s_root
                    current_pos = surface_node + normal * s_root
                    U_e = vel_interpolator(current_pos)
                    edge_velocities[method] = U_e

                    if verbose:
                        logging.info(f"Node {i} ({method}): Edge found at s={s_root}, U_e={U_e}")
                except ValueError as ve:
                    logging.warning(f"Node {i} ({method}): brentq failed with ValueError: {ve}.")
                    bl_thicknesses[method] = np.nan
                    displacement_thicknesses[method] = np.nan
                    momentum_thicknesses[method] = np.nan
                    shape_factors[method] = np.nan
                    edge_velocities[method] = np.nan
                    continue

                # Check if s_root is too small
                if s_root <= 1e-6:
                    logging.warning(f"Node {i} ({method}): s_root={s_root} is too small.")
                    displacement_thicknesses[method] = np.nan
                    momentum_thicknesses[method] = np.nan
                    shape_factors[method] = np.nan
                    continue

                # Compute displacement and momentum thicknesses
                num_steps = 1000
                s_values = np.linspace(0, s_root, num_steps)

                integrand_displacement = []
                integrand_momentum = []

                for s_val in s_values:
                    current_pos = surface_node + normal * s_val
                    V_current = vel_interpolator(current_pos)

                    if np.isnan(V_current) or U_e == 0:
                        integrand_displacement.append(0.0)
                        integrand_momentum.append(0.0)
                    else:
                        u_ratio = V_current / U_e
                        integrand_displacement.append(1 - u_ratio)
                        integrand_momentum.append(u_ratio * (1 - u_ratio))

                # Convert to NumPy arrays and ensure they are 1D
                integrand_displacement = np.array(integrand_displacement).squeeze()
                integrand_momentum = np.array(integrand_momentum).squeeze()
                s_values = s_values.squeeze()

                # Ensure arrays have at least two points
                if len(s_values) < 2 or len(integrand_displacement) < 2:
                    logging.warning(f"Node {i} ({method}): Insufficient points for integration.")
                    displacement_thicknesses[method] = np.nan
                    momentum_thicknesses[method] = np.nan
                    shape_factors[method] = np.nan
                    continue

                # Perform integration
                try:
                    displacement_thickness = np.trapz(integrand_displacement, s_values)
                    momentum_thickness = np.trapz(integrand_momentum, s_values)

                    displacement_thicknesses[method] = displacement_thickness
                    momentum_thicknesses[method] = momentum_thickness
                    shape_factors[method] = (displacement_thickness / momentum_thickness) if momentum_thickness != 0 else np.nan

                    if verbose:
                        logging.debug(f"Node {i} ({method}): Displacement Thickness={displacement_thickness}, Momentum Thickness={momentum_thickness}, Shape Factor={shape_factors[method]}")
                except Exception as e:
                    logging.error(f"Node {i} ({method}): Integration failed with error: {e}")
                    displacement_thicknesses[method] = np.nan
                    momentum_thicknesses[method] = np.nan
                    shape_factors[method] = np.nan

            elif method == 'vorticity_threshold':
                s = 0.0
                found_edge = False

                for step in range(int(max_steps)):
                    s += step_size
                    current_pos = surface_node + normal * s
                    vorticity_current = vorticity_interpolator(current_pos)

                    if np.isnan(vorticity_current) or np.isnan(vorticity_surface):
                        logging.debug(f"Node {i} ({method}): Encountered NaN at s={s}.")
                        break

                    ratio = abs(vorticity_current / vorticity_surface)

                    if ratio <= 1e-4:
                        bl_thicknesses[method] = s
                        U_e = vel_interpolator(current_pos)
                        edge_velocities[method] = U_e
                        found_edge = True
                        if verbose:
                            logging.info(f"Node {i} ({method}): Edge found at s={s}, U_e={U_e}")
                        break

                if not found_edge or s <= 1e-6:
                    logging.warning(f"Node {i} ({method}): Edge not found or s={s} is too small.")
                    bl_thicknesses[method] = np.nan
                    displacement_thicknesses[method] = np.nan
                    momentum_thicknesses[method] = np.nan
                    shape_factors[method] = np.nan
                    edge_velocities[method] = np.nan
                    continue

                # Compute displacement and momentum thicknesses
                num_steps = 1000
                s_values = np.linspace(0, s, num_steps)

                integrand_displacement = []
                integrand_momentum = []

                for s_val in s_values:
                    current_pos = surface_node + normal * s_val
                    V_current = vel_interpolator(current_pos)

                    if np.isnan(V_current) or U_e == 0:
                        integrand_displacement.append(0.0)
                        integrand_momentum.append(0.0)
                    else:
                        u_ratio = V_current / U_e
                        integrand_displacement.append(1 - u_ratio)
                        integrand_momentum.append(u_ratio * (1 - u_ratio))

                # Convert to NumPy arrays and ensure they are 1D
                integrand_displacement = np.array(integrand_displacement).squeeze()
                integrand_momentum = np.array(integrand_momentum).squeeze()
                s_values = s_values.squeeze()

                # Ensure arrays have at least two points
                if len(s_values) < 2 or len(integrand_displacement) < 2:
                    logging.warning(f"Node {i} ({method}): Insufficient points for integration.")
                    displacement_thicknesses[method] = np.nan
                    momentum_thicknesses[method] = np.nan
                    shape_factors[method] = np.nan
                    continue

                # Perform integration
                try:
                    displacement_thickness = np.trapz(integrand_displacement, s_values)
                    momentum_thickness = np.trapz(integrand_momentum, s_values)

                    displacement_thicknesses[method] = displacement_thickness
                    momentum_thicknesses[method] = momentum_thickness
                    shape_factors[method] = (displacement_thickness / momentum_thickness) if momentum_thickness != 0 else np.nan

                    if verbose:
                        logging.debug(f"Node {i} ({method}): Displacement Thickness={displacement_thickness}, Momentum Thickness={momentum_thickness}, Shape Factor={shape_factors[method]}")
                except Exception as e:
                    logging.error(f"Node {i} ({method}): Integration failed with error: {e}")
                    displacement_thicknesses[method] = np.nan
                    momentum_thicknesses[method] = np.nan
                    shape_factors[method] = np.nan

            else:
                # Handle other methods if any
                pass

    # return (i, bl_thicknesses, displacement_thicknesses, momentum_thicknesses, shape_factors, edge_velocities)
        return (i, bl_thicknesses, displacement_thicknesses, momentum_thicknesses, shape_factors, edge_velocities)

    except Exception as e:
        logging.error(f"Error processing node {i}: {e}")
        # Return default values or handle the error as needed
        return (i, {}, {}, {}, {}, {})

class MeshParser:
    def __init__(self, surface_file, flow_file, output_dir='output', x_locations=None):
        self.flow_data         = {}
        self.surface_data      = {}
        self.flow_nodes        = None
        self.flow_elements     = None
        self.surface_nodes     = None
        self.flow_zone_type    = None
        self.surface_elements  = None
        self.surface_zone_type = None
        self.flow_file         = flow_file
        self.output_dir        = output_dir
        self.surface_file      = surface_file
        self.original_flow_nodes = None
        self.surface_normals   = None
        self.bl_thicknesses    = {}
        self.displacement_thicknesses = {}
        self.momentum_thicknesses = {}
        self.shape_factors     = {}
        self.edge_velocities   = {}
        self.surface_reordered_indices = None
        self.x_locations = x_locations if x_locations is not None else []
        self.velocity_profiles = defaultdict(list)  # Dictionary to store multiple velocity profiles per x-location

        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        try:
            self.surface_data, _, self.surface_nodes, self.surface_elements, self.node_neighbors, self.surface_zone_type = self.parse_data(self.surface_file)
            self.flow_data, _, self.flow_nodes, self.flow_elements, self.node_neighbors, self.flow_zone_type = self.parse_data(self.flow_file, reorder_surface=False)
            if self.flow_nodes is not None:
                self.original_flow_nodes = self.flow_nodes.copy()
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def parse_data(self, file_path, reorder_surface=True):
        try:
            print(f'Parsing file {file_path}...')
            data = {}
            nodes = []
            elements = []
            node_neighbors = defaultdict(set)

            with open(file_path, 'r') as file:
                lines = file.readlines()

            headers_line = lines[1].strip()
            headers = [h.strip() for h in headers_line.split('=')[1].strip().replace('"', '').split(',')]
            for header in headers:
                data[header] = []

            node_count_line = lines[2].strip()
            node_count = int(node_count_line.split('NODES=')[1].split(',')[0])
            element_count = int(node_count_line.split('ELEMENTS=')[1].split(',')[0])
            zone_type = node_count_line.split('ZONETYPE=')[1].strip()

            current_line = 3
            for line in lines[current_line:current_line + node_count]:
                try:
                    values = line.strip().split()
                    if len(values) != len(headers):
                        continue
                    for i, header in enumerate(headers):
                        data[header].append(float(values[i]))
                    if 'z' in data:
                        nodes.append([float(values[i]) for i in range(3)])  # x, y, z
                    else:
                        nodes.append([float(values[i]) for i in range(2)])  # x, y
                except ValueError:
                    continue
            current_line += node_count

            for header in data:
                data[header] = np.array(data[header])
            nodes = np.array(nodes)

            # Read element connectivity
            for line in lines[current_line:current_line + element_count]:
                try:
                    element = [int(index) for index in line.strip().split()]
                    if zone_type == 'FEQUADRILATERAL' and len(element) == 4:
                        elements.append(element)
                    elif zone_type == 'FELINESEG' and len(element) == 2:
                        elements.append(element)
                    for i in range(len(element)):
                        for j in range(len(element)):
                            if i != j:
                                node_neighbors[element[i]].add(element[j])
                except ValueError:
                    continue
            current_line += element_count

            elements = np.array(elements) - 1  # Adjust for zero-based indexing

            for key in node_neighbors:
                node_neighbors[key] = list(node_neighbors[key])

            # Reorder surface nodes
            if reorder_surface and zone_type == 'FELINESEG':
                nodes, reordered_indices = self.reorder_surface_nodes(nodes, elements, zone_type)
                self.surface_nodes = nodes
                self.surface_reordered_indices = reordered_indices

            # Reorder volume nodes
            if zone_type == 'FEQUADRILATERAL' and self.surface_nodes is not None:
                reordered_volume_nodes = self.reorder_volume_nodes(nodes)
                nodes = reordered_volume_nodes

            return data, headers, nodes, elements, node_neighbors, zone_type

        except Exception as e:
            logging.error(f"Error parsing data from {file_path}: {e}")
            raise

    def reorder_surface_nodes(self, nodes, elements, zone_type):
        try:
            original_surface_nodes = nodes.copy()

            if len(elements) == 0:
                return nodes, list(range(len(nodes)))

            if zone_type == 'FELINESEG':
                start_node = elements[0][0]
                ordered_nodes = [start_node]
                current_node = start_node
                while len(ordered_nodes) < len(nodes):
                    found = False
                    for element in elements:
                        if current_node in element:
                            next_node = element[1] if element[0] == current_node else element[0]
                            if next_node not in ordered_nodes:
                                ordered_nodes.append(next_node)
                                current_node = next_node
                                found = True
                                break
                    if not found:
                        break  # Prevent infinite loop if no next node is found
                reordered_surface_nodes = np.array([nodes[node] for node in ordered_nodes])
            else:
                reordered_surface_nodes = nodes

            self.plot_debug_nodes(original_surface_nodes, reordered_surface_nodes, title="Surface Nodes", is_surface=True)
            return reordered_surface_nodes, ordered_nodes

        except Exception as e:
            logging.error(f"Error reordering surface nodes: {e}")
            raise

    def reorder_volume_nodes(self, volume_nodes):
        try:
            kdtree = KDTree(volume_nodes[:, :2])
            reordered_volume_indices = []
            for surface_node in self.surface_nodes[:, :2]:
                _, idx = kdtree.query(surface_node)
                reordered_volume_indices.append(idx)

            reordered_volume_nodes = volume_nodes[reordered_volume_indices]
            return reordered_volume_nodes

        except Exception as e:
            logging.error(f"Error reordering volume nodes: {e}")
            raise

    def plot_debug_nodes(self, original_nodes, reordered_nodes, title, is_surface=False):
        try:
            plt.figure()
            if original_nodes is not None:
                plt.plot(original_nodes[:, 0], original_nodes[:, 1], '-', label='Original Nodes')
            plt.plot(reordered_nodes[:, 0], reordered_nodes[:, 1], '-', label='Reordered Nodes')
            plt.title(title)
            plt.legend()
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.grid(False)
            plt.gca().set_aspect('equal')
            #plt.show()
        except Exception as e:
            logging.error(f"Error plotting debug nodes: {e}")

    def compute_surface_normals(self):
        try:
            normals = []
            for i in range(len(self.surface_nodes) - 1):
                p1 = self.surface_nodes[i, :2]
                p2 = self.surface_nodes[i + 1, :2]
                tangent = p2 - p1
                normal = np.array([-tangent[1], tangent[0]])
                norm = np.linalg.norm(normal)
                if norm != 0:
                    normal /= norm
                else:
                    normal = np.array([0.0, 0.0])
                normals.append(normal)
            normals.append(normals[-1])
            self.surface_normals = np.array(normals)
        except Exception as e:
            logging.error(f"Error computing surface normals: {e}")
            raise

    def plot_surface_normals(self, scale=0.05):
        try:
            plt.figure()
            plt.plot(self.surface_nodes[:, 0], self.surface_nodes[:, 1], '-', label='Surface Nodes')
            for i in range(len(self.surface_nodes)):
                plt.arrow(
                    self.surface_nodes[i, 0],
                    self.surface_nodes[i, 1],
                    self.surface_normals[i, 0] * scale,
                    self.surface_normals[i, 1] * scale,
                    head_width=0.005,
                    head_length=0.01,
                    fc='r',
                    ec='r',
                    label='Normals' if i == 0 else "_nolegend_"
                )
            plt.title("Surface Nodes with Normals")
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.grid(False)
            plt.gca().set_aspect('equal')
            plt.legend()
            #plt.show()
        except Exception as e:
            logging.error(f"Error plotting surface normals: {e}")

    def plot_surface_and_flow_data(self):
        try:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.flow_data['x'], self.flow_data['y'], s=1, c='blue', label='Flow Data')
            plt.plot(self.surface_nodes[:, 0], self.surface_nodes[:, 1], 'r-', linewidth=2, label='Surface Nodes')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.title('Surface Nodes Overlaid on Flow Data')
            plt.legend()
            plt.grid(False)
            plt.axis('equal')
            #plt.show()
        except Exception as e:
            logging.error(f"Error plotting surface and flow data: {e}")

    def compute_boundary_layer_thickness(self, methods=['edge_velocity', 'vorticity_threshold'], threshold=0.99, max_steps=1e6, step_size=1e-7, tolerance=1e-3, n_jobs=-1, verbose=False):
        try:
            print("Computing boundary layer thickness...")

            # Get positions and velocities from flow data
            X = self.flow_data.get('x')
            Y = self.flow_data.get('y')

            U = self.flow_data.get('Velocity_x')
            V = self.flow_data.get('Velocity_y')
            Vorticity = self.flow_data.get('Vorticity')

            if X is None or Y is None or U is None or V is None or Vorticity is None:
                print("Required data not found in flow data.")
                return

            # Compute velocity magnitude
            V_mag = np.sqrt(U**2 + V**2)

            # Prepare the points array (2D)
            points = np.column_stack((X, Y))

            # Remove any NaN values from the data
            valid_indices = ~np.isnan(V_mag) & ~np.isnan(X) & ~np.isnan(Y) & ~np.isnan(Vorticity)
            points = points[valid_indices]
            V_mag = V_mag[valid_indices]
            Vorticity = Vorticity[valid_indices]

            # Prepare arguments for parallel processing
            args_list = [
                (i, surface_node, normal, points, V_mag, Vorticity, methods, threshold, max_steps, step_size, verbose)
                for i, (surface_node, normal) in enumerate(zip(self.surface_nodes[:, :2], self.surface_normals))
            ]

            # Use joblib to parallelize the computation
            results = Parallel(n_jobs=n_jobs)(
                delayed(compute_bl_thickness_for_node)(args) for args in args_list
            )

            # Collect results
            self.bl_thicknesses = {method: [] for method in methods}
            self.displacement_thicknesses = {method: [] for method in methods}
            self.momentum_thicknesses = {method: [] for method in methods}
            self.shape_factors = {method: [] for method in methods}
            self.edge_velocities = {method: [] for method in methods}

            # Sort results by node index to maintain order
            results.sort(key=lambda x: x[0])

            for (i, bl_thicknesses_node, displacement_thicknesses_node,
                 momentum_thicknesses_node, shape_factors_node, edge_velocities_node) in results:
                for method in methods:
                    self.bl_thicknesses[method].append(bl_thicknesses_node.get(method, np.nan))
                    self.displacement_thicknesses[method].append(displacement_thicknesses_node.get(method, np.nan))
                    self.momentum_thicknesses[method].append(momentum_thicknesses_node.get(method, np.nan))
                    self.shape_factors[method].append(shape_factors_node.get(method, np.nan))
                    self.edge_velocities[method].append(edge_velocities_node.get(method, np.nan))

        except Exception as e:
            logging.error(f"Error computing boundary layer thickness: {e}")
            raise

    def smooth_velocity_profile(self, u_normalized, window_length=11, polyorder=4):
        """
        Smooths the normalized velocity profile using the Savitzky-Golay filter.

        Parameters:
        - u_normalized (np.ndarray): The normalized velocity data to smooth.
        - window_length (int): The length of the filter window (must be odd and >= polyorder + 2).
        - polyorder (int): The order of the polynomial used to fit the samples.

        Returns:
        - np.ndarray: The smoothed normalized velocity data.
        """
        try:
            # Ensure window_length is odd and less than or equal to the size of u_normalized
            if window_length >= len(u_normalized):
                window_length = len(u_normalized) - 1 if len(u_normalized) % 2 == 0 else len(u_normalized)
                if window_length < polyorder + 2:
                    # Cannot apply Savitzky-Golay filter; return original data
                    logging.warning("Window length too long or too short for Savitzky-Golay filter. Returning original data.")
                    return u_normalized
            if window_length % 2 == 0:
                window_length += 1  # Make it odd

            u_smooth = savgol_filter(u_normalized, window_length=window_length, polyorder=polyorder)
            return u_smooth
        except Exception as e:
            logging.error(f"Smoothing failed: {e}")
            return u_normalized  # Return original data if smoothing fails

    def compute_velocity_profiles(self, s_max=0.01, num_points=100, verbose=False, smoothing=True, window_length=11, polyorder=4):
        try:
            if not self.x_locations:
                print("No x_locations specified for velocity profile extraction.")
                return

            # Prepare the interpolator
            points = np.column_stack((self.flow_data['x'], self.flow_data['y']))
            U = self.flow_data.get('Velocity_x')
            V = self.flow_data.get('Velocity_y')
            V_mag = np.sqrt(U**2 + V**2)
            vel_interpolator = NearestNDInterpolator(points, V_mag)

            # Create a KDTree for efficient spatial searches
            kdtree = KDTree(self.surface_nodes[:, :2])

            for x_loc in self.x_locations:
                # Find the index of the closest surface node to x_loc
                x_diffs = np.abs(self.surface_nodes[:, 0] - x_loc)
                closest_idx = np.argmin(x_diffs)
                min_diff = x_diffs[closest_idx]

                # Define a reasonable maximum allowable difference
                max_allowable_diff = 0.01  # Adjust based on your mesh resolution

                if min_diff > max_allowable_diff:
                    if verbose:
                        logging.warning(
                            f"No surface nodes found within {max_allowable_diff} tolerance for x = {x_loc}. "
                            f"Nearest node is at x = {self.surface_nodes[closest_idx, 0]:.4f} with difference {min_diff:.6f}."
                        )
                    continue  # Skip if the nearest node is too far

                surface_node = self.surface_nodes[closest_idx, :2]
                normal = self.surface_normals[closest_idx]

                # Retrieve boundary layer thickness and edge velocity for normalization
                bl_thickness = self.bl_thicknesses.get('vorticity_threshold', [np.nan])[closest_idx]
                U_e = self.edge_velocities.get('vorticity_threshold', [np.nan])[closest_idx]

                if np.isnan(bl_thickness) or U_e == 0 or np.isnan(U_e):
                    if verbose:
                        logging.warning(
                            f"Invalid BL thickness or edge velocity at x = {x_loc}, node = {closest_idx}."
                        )
                    continue  # Skip normalization if data is invalid

                # Adjust s_max for current location to not exceed BL thickness
                s_max_loc = min(s_max, bl_thickness)

                # Sample points along the normal direction up to s_max_loc
                s_values = np.linspace(0, s_max_loc, num_points)
                positions = surface_node + np.outer(s_values, normal)

                # Interpolate velocities at these positions
                velocities = vel_interpolator(positions)

                # Handle NaNs in velocities
                valid = ~np.isnan(velocities)
                s_values = s_values[valid]
                velocities = velocities[valid]

                if len(s_values) == 0:
                    if verbose:
                        logging.warning(
                            f"No valid velocity data for x = {x_loc}, node = {closest_idx}."
                        )
                    continue

                # Normalize the profiles
                s_normalized = s_values / bl_thickness
                u_normalized = velocities / U_e

                # Apply smoothing if enabled
                if smoothing:
                    u_normalized = self.smooth_velocity_profile(u_normalized, window_length=window_length, polyorder=polyorder)

                if verbose:
                    print(
                        f"Extracted and normalized velocity profile at x = {x_loc:.4f} (node {closest_idx}), "
                        f"number of points: {len(s_normalized)}"
                    )

                # Store the profile using the actual x-location and node identifier
                self.velocity_profiles[x_loc].append({
                    'node_index': closest_idx,
                    's_normalized': s_normalized,
                    'u_normalized': u_normalized
                })

        except Exception as e:
            logging.error(f"Error computing velocity profiles: {e}")
            raise

    def plot_velocity_profiles(self):
        try:
            if not self.velocity_profiles:
                print("No velocity profiles to plot.")
                return

            for x_loc, profiles in self.velocity_profiles.items():
                plt.figure()
                for profile in profiles:
                    node_idx = profile['node_index']
                    s_norm = profile['s_normalized']
                    u_norm = profile['u_normalized']
                    plt.plot(u_norm, s_norm,label=f'Node {node_idx}')
                plt.xlabel('Normalized Velocity (u/u_e)')
                plt.ylabel('Normalized Distance (s/δ)')
                plt.title(f'Normalized Velocity Profiles at x = {x_loc}')
                plt.legend()
                plt.grid(False)
                #plt.show()
        except Exception as e:
            logging.error(f"Error plotting velocity profiles: {e}")

    def plot_boundary_layer_quantities(self):
        try:
            x_coords = np.asarray(self.surface_nodes[:, 0], dtype=float).flatten()

            # Apply smoothing
            window_size = 5  # Adjust as needed for smoothing
            methods = list(self.bl_thicknesses.keys())

            # Plot Boundary Layer Thickness vs X Coordinate
            plt.figure()
            for method in methods:
                thicknesses = np.asarray(self.bl_thicknesses[method], dtype=float).flatten()
                thicknesses_smoothed = self.smooth_data(thicknesses, window_size)
                plt.plot(x_coords, thicknesses_smoothed, label=f'BL Thickness ({method})')
            plt.title('Boundary Layer Thickness vs X Coordinate')
            plt.xlabel('X Coordinate')
            plt.ylabel('Boundary Layer Thickness')
            plt.legend()
            plt.grid(False)
            #plt.show()

            # Plot Displacement Thickness vs X Coordinate
            plt.figure()
            for method in methods:
                displacement_list = np.asarray(self.displacement_thicknesses[method], dtype=float).flatten()
                displacement_smoothed = self.smooth_data(displacement_list, window_size)
                plt.plot(x_coords, displacement_smoothed, label=f'Displacement Thickness ({method})')
            plt.title('Displacement Thickness vs X Coordinate')
            plt.xlabel('X Coordinate')
            plt.ylabel('Displacement Thickness')
            plt.legend()
            plt.grid(False)
            #plt.show()

            # Plot Momentum Thickness vs X Coordinate
            plt.figure()
            for method in methods:
                momentum_list = np.asarray(self.momentum_thicknesses[method], dtype=float).flatten()
                momentum_smoothed = self.smooth_data(momentum_list, window_size)
                plt.plot(x_coords, momentum_smoothed, label=f'Momentum Thickness ({method})')
            plt.title('Momentum Thickness vs X Coordinate')
            plt.xlabel('X Coordinate')
            plt.ylabel('Momentum Thickness')
            plt.legend()
            plt.grid(False)
            #plt.show()

            # Plot Shape Factor vs X Coordinate
            plt.figure()
            for method in methods:
                shape_factor_list = np.asarray(self.shape_factors[method], dtype=float).flatten()
                shape_factor_smoothed = self.smooth_data(shape_factor_list, window_size)
                plt.plot(x_coords, shape_factor_smoothed, label=f'Shape Factor ({method})')
            plt.title('Shape Factor vs X Coordinate')
            plt.xlabel('X Coordinate')
            plt.ylabel('Shape Factor')
            plt.legend()
            plt.grid(False)
            #plt.show()

            plt.figure()
            plt.plot(x_coords, momentum_smoothed, x_coords, displacement_smoothed, x_coords, thicknesses_smoothed, label=f'BL ({method})')
            plt.title('BL all vs X Coordinate')
            plt.xlabel('X Coordinate')
            plt.ylabel('BL ALL')
            plt.legend()
            plt.grid(False)

            # Plot Edge Velocity vs X Coordinate
            plt.figure()
            for method in methods:
                edge_velocity_list = np.asarray(self.edge_velocities[method], dtype=float).flatten()
                edge_velocity_smoothed = self.smooth_data(edge_velocity_list, window_size)
                plt.plot(x_coords, edge_velocity_smoothed, label=f'Edge Velocity ({method})')
            plt.title('Edge Velocity vs X Coordinate')
            plt.xlabel('X Coordinate')
            plt.ylabel('Edge Velocity')
            plt.legend()
            plt.grid(False)
            #plt.show()

            # Plot Airfoil Surface with Boundary Layer Thickness Overlaid
            plt.figure()
            x_surface = self.surface_nodes[:, 0]
            y_surface = self.surface_nodes[:, 1]
            plt.plot(x_surface, y_surface, 'k-', label='Airfoil Surface')

            for method in methods:
                thicknesses = np.asarray(self.bl_thicknesses[method], dtype=float).flatten()
                thicknesses_smoothed = self.smooth_data(thicknesses, window_size)
                for i, (x, y, normal, thickness) in enumerate(zip(x_surface, y_surface, self.surface_normals, thicknesses_smoothed)):
                    if not np.isnan(thickness):
                        # Scale the normal vector by the thickness
                        end_point = [x + normal[0] * thickness, y + normal[1] * thickness]
                        plt.plot([x, end_point[0]], [y, end_point[1]], 'r-', alpha=0.5)
            plt.title('Airfoil Surface with Boundary Layer Thickness')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.legend()
            plt.grid(False)
            plt.axis('equal')
            #plt.show()
        except Exception as e:
            logging.error(f"Error plotting boundary layer quantities: {e}")

    def smooth_data(self, data, window_size=5):
        """Smooth data using a moving average filter."""
        try:
            # Convert data to a flat NumPy array of type float
            data = np.asarray(data).astype(float).flatten()
            # Handle NaN values by interpolating or removing
            nan_indices = np.isnan(data)
            if np.all(nan_indices):
                # All values are NaN, return as is
                return data
            elif np.any(nan_indices):
                # Interpolate to fill NaN values
                not_nan_indices = ~nan_indices
                data[nan_indices] = np.interp(
                    np.flatnonzero(nan_indices),
                    np.flatnonzero(not_nan_indices),
                    data[not_nan_indices]
                )
            if window_size < 3:
                return data  # No smoothing needed
            smoothed_data = np.convolve(data, np.ones(int(window_size))/float(window_size), mode='same')
            return smoothed_data
        except Exception as e:
            logging.error(f"Error smoothing data: {e}")
            return data  # Return original data if smoothing fails

    def save_boundary_layer_to_file(self, file_name, x_coords, data):
        try:
            print(f"Saving boundary layer data to {file_name}...")
            with open(os.path.join(self.output_dir, file_name), 'w') as file:
                file.write("VARIABLES = \"x\", \"value\"\n")
                file.write("ZONE T=\"Boundary Layer Data\"\n")
                for x, val in zip(x_coords, data):
                    x = float(x)
                    if isinstance(val, np.ndarray):
                        val = val.item()
                    else:
                        val = float(val)
                    file.write(f"{x} {val}\n")
        except Exception as e:
            logging.error(f"Error saving boundary layer data to {file_name}: {e}")

    def save_velocity_profiles_to_files(self, prefix='velocity_profile'):
        """
        Saves each velocity profile to a separate file with a descriptive filename.

        Parameters:
        - prefix (str): Prefix for the output files.
        """
        try:
            print("Saving velocity profiles to files...")
            for x_loc, profiles in self.velocity_profiles.items():
                for profile in profiles:
                    node_idx = profile['node_index']
                    s_norm = profile['s_normalized']
                    u_norm = profile['u_normalized']
                    # Create a descriptive filename
                    file_name = f'{prefix}_x_{x_loc:.4f}_node_{node_idx}.dat'
                    with open(os.path.join(self.output_dir, file_name), 'w') as f:
                        f.write("VARIABLES = \"s_normalized\", \"u_normalized\"\n")
                        f.write("ZONE T=\"Velocity Profile\"\n")
                        for s, u in zip(s_norm, u_norm):
                            f.write(f"{s:.6f} {u:.6f}\n")
                    print(f"Saved velocity profile at x = {x_loc:.4f}, node {node_idx} to {file_name}")
        except Exception as e:
            logging.error(f"Error saving velocity profiles to files: {e}")

    def run(self, methods=['edge_velocity', 'vorticity_threshold'], threshold=0.99, max_steps=1e6, step_size=1e-7, tolerance=1e-3, n_jobs=-1, verbose=False):
        try:
            self.load_data()
            self.plot_surface_and_flow_data()
            self.compute_surface_normals()
            self.plot_surface_normals()
            self.compute_boundary_layer_thickness(
                methods=methods,
                threshold=threshold,
                max_steps=max_steps,
                step_size=step_size,
                tolerance=tolerance,
                n_jobs=n_jobs,
                verbose=verbose
            )
            self.plot_boundary_layer_quantities()

            # Save boundary layer parameters to files
            for method in methods:
                x_coords = np.asarray(self.surface_nodes[:, 0], dtype=float).flatten()
                # self.save_boundary_layer_to_file(f'bl_thickness_{method}_delta.dat', x_coords, self.bl_thicknesses[method])
                # self.save_boundary_layer_to_file(f'bl_displacement_thickness_{method}_deltaStar.dat', x_coords, self.displacement_thicknesses[method])
                # self.save_boundary_layer_to_file(f'bl_momentum_thickness_{method}_theta.dat', x_coords, self.momentum_thicknesses[method])
                # self.save_boundary_layer_to_file(f'bl_shape_factor_{method}_H.dat', x_coords, self.shape_factors[method])
                # self.save_boundary_layer_to_file(f'bl_edge_velocity_{method}_ue.dat', x_coords, self.edge_velocities[method])

                self.save_boundary_layer_to_file(f'bl_thickness_delta.dat', x_coords, self.bl_thicknesses[method])
                self.save_boundary_layer_to_file(f'bl_displacement_thickness_deltaStar.dat', x_coords, self.displacement_thicknesses[method])
                self.save_boundary_layer_to_file(f'bl_momentum_thickness_theta.dat', x_coords, self.momentum_thicknesses[method])
                self.save_boundary_layer_to_file(f'bl_shape_factor_H.dat', x_coords, self.shape_factors[method])
                self.save_boundary_layer_to_file(f'bl_edge_velocity_ue.dat', x_coords, self.edge_velocities[method])

            # Compute velocity profiles at specified x_locations with smoothing enabled
            self.compute_velocity_profiles(
                verbose=verbose,
                smoothing=True,            # Enable smoothing
                window_length=13,          # Example window length (must be odd)
                polyorder=2               # Example polynomial order
            )

            # Plot the velocity profiles
            self.plot_velocity_profiles()

            # Save velocity profiles to files
            self.save_velocity_profiles_to_files()

        except Exception as e:
            logging.error(f"Error running MeshParser: {e}")
            raise
    def run_bl7(
        surface_file: str,
        flow_file: str,
        output_dir: str,
        x_locations: list[float],
        methods: list[str] = ["vorticity_threshold"],
        threshold: float = 0.99,
        max_steps: float = 1e6,
        step_size: float = 1e-7,
        tolerance: float = 1e-3,
        n_jobs: int = -1,
        verbose: bool = False,
    ):
        parser = MeshParser(
            surface_file=surface_file,
            flow_file=flow_file,
            output_dir=output_dir,
            x_locations=x_locations,
        )
        parser.run(
            methods=methods,
            threshold=threshold,
            max_steps=max_steps,
            step_size=step_size,
            tolerance=tolerance,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        return parser  # allows post-hook use (e.g., plotting)
# if __name__ == "__main__":
#     try:
#         # Define the paths to your data files
# #        flow_file        = '/home/jnm8/DAAL/SU2/NLF0416/AIAA_mesh/RANS_SA_LM/M0.1/Re4e6/sense/grid/2_Medium/MB-correlation/flow_vol_.dat'
# #        surface_file     = '/home/jnm8/DAAL/SU2/NLF0416/AIAA_mesh/RANS_SA_LM/M0.1/Re4e6/sense/grid/2_Medium/MB-correlation/flow_surf_.dat'
#         flow_file = '/home/jnm8/DAAL/HPC2/SE2A_NEW/RANS/SA-LM/SA-noft2/MEDIDA-BAEDER/M0.4/Re1.6e+07/cl0.4/flow_vol_.dat'
#         surface_file = '/home/jnm8/DAAL/HPC2/SE2A_NEW/RANS/SA-LM/SA-noft2/MEDIDA-BAEDER/M0.4/Re1.6e+07/cl0.4/flow_surf_.dat'

#         # Define the output directory and x_locations
#         output_directory = '/home/jnm8/DAAL/HPC2/SE2A_NEW/RANS/SA-LM/SA-noft2/MEDIDA-BAEDER/M0.4/Re1.6e+07/cl0.4/BL'
#         # x_locations = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]  # Example x-locations
#         x_locations = [0.1,0.4,0.655,0.7,0.9]  # Example x-locations

#         # Initialize the MeshParser
#         processor = MeshParser(
#             surface_file=surface_file,
#             flow_file=flow_file,
#             output_dir=output_directory,
#             x_locations=x_locations
#         )

#         # Run the MeshParser with desired parameters
#         processor.run(
#             methods=['vorticity_threshold'],  # Specify desired methods
#             threshold=0.99,                   # Adjust threshold as needed
#             max_steps=1e6,                    # Adjust max_steps as needed
#             step_size=1e-7,                   # Adjust step_size as needed
#             tolerance=1e-3,                   # Adjusted tolerance
#             n_jobs=16,                        # Utilize all CPU cores
#             verbose=True                      # Enable verbose logging for debugging
#         )
#     except SyntaxError as se:
#         print(f"SyntaxError detected: {se}")
#         print("Please ensure that all 'try' blocks are properly closed with 'except' or 'finally' blocks.")
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         print("Check the log file 'mesh_parser_debug.log' for more details.")
#     plt.show()