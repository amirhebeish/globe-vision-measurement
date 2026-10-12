import open3d as o3d
import numpy as np
import copy
import cv2
from scipy.optimize import least_squares

# ground -> 6th floor transformation
x, y, z = 17.2, -2.43, -26.55
yaw, pitch, roll = 2.8274, -1.572, 0

# Camera calibration matrix scaled to the projection img resolution
width, height = 839, 432
fx, fy = 599, 598
cx, cy = 444, 207

K = np.array([[fx,  0, cx],
              [ 0, fy, cy],
              [ 0,  0,  1]])

# Lidar -> camera transformation 
T_lidar_to_cam = np.array([[-0.292, -0.956, 0., 3.],
                           [-0.033, 0.01, -0.999, 0.002],
                           [0.956, -0.292, -0.035, -0.052],
                           [0., 0., 0., 1.]])

def load_data(pcd_path, image_path):

    pcd = o3d.io.read_point_cloud(pcd_path)
    img = cv2.imread(image_path)

    return pcd, img

def merge_pcd_gnd_6th(file_path_6th, file_path_ground, output_path):
    print(f"Loading {file_path_6th} and {file_path_ground}...")
    pcd_base = o3d.io.read_point_cloud(file_path_6th)
    pcd_move = o3d.io.read_point_cloud(file_path_ground)

    if pcd_base.is_empty() or pcd_move.is_empty():
        print("Point clouds could not be loaded.")
        return
    
    # Create rotation matrix following XYZ rotation
    
    R_x = pcd_move.get_rotation_matrix_from_xyz((roll, 0, 0))    
    R_y = pcd_move.get_rotation_matrix_from_xyz((0, pitch, 0))
    R_z = pcd_move.get_rotation_matrix_from_xyz((0, 0, yaw))

    R = R_z @ R_y @ R_x

    # Join R and t into a homogeneous transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[0, 3] = x
    transform_matrix[1, 3] = y
    transform_matrix[2, 3] = z

    # Apply transformation to the point cloud in the ground frame
    pcd_move_transformed = copy.deepcopy(pcd_move)
    pcd_move_transformed.transform(transform_matrix)

    # Merge point clouds into a single one
    pcd_combined = pcd_base + pcd_move_transformed

    print(f"Saving point cloud to {output_path}.\nTotal points: {len(pcd_combined.points)}")
    o3d.io.write_point_cloud(output_path, pcd_combined)

def get_colored_pcd(pcd_path, image_path, output_path, visualise=False):
    
    pcd, img = load_data(pcd_path, image_path)
    
    if pcd.is_empty() or img is None:
        print("Error: Could not load PCD or Image.")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = img.shape

    # Transform point cloud to the camera frame

    points_lidar = np.asarray(pcd.points)
    ones = np.ones((points_lidar.shape[0], 1))
    points_hom = np.hstack((points_lidar, ones)) # Points in homogeneous coordinates
    
    points_cam = (T_lidar_to_cam @ points_hom.T).T
    xyz_cam = points_cam[:, :3] # Points back to Cartesian
    
    z_vals = xyz_cam[:, 2]
    valid_mask = z_vals > 0 # Mask points in front of the camera only
    
    # Normalisation to get u and v in the image for all the points
    uv = np.zeros((len(points_lidar), 2))
    uv[valid_mask, 0] = (K[0,0] * xyz_cam[valid_mask, 0] / z_vals[valid_mask]) + K[0,2]
    uv[valid_mask, 1] = (K[1,1] * xyz_cam[valid_mask, 1] / z_vals[valid_mask]) + K[1,2]
    
    u = uv[:, 0]
    v = uv[:, 1]
    
    in_bounds = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    final_mask = valid_mask & in_bounds
    
    print(f"Projected {np.sum(final_mask)} valid points out of {len(points_lidar)}.")

    # Assign colours to all the points based on their projected u, v position in the image

    colours = np.ones((len(points_lidar), 3)) * 0.5
    u_idx = u[final_mask].astype(int)
    v_idx = v[final_mask].astype(int)

    u_idx = np.clip(u_idx, 0, img_w - 1) # Keep image index within the boundaries
    v_idx = np.clip(v_idx, 0, img_h - 1)

    # Apply colour to the pointcloud
    colours[final_mask] = img[v_idx, u_idx] / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colours)
    print(f"Saving to {output_path}...")
    o3d.io.write_point_cloud(output_path, pcd)

    if visualise:
        # Visualise in open3d
        o3d.visualization.draw_geometries([pcd], window_name="Colored Cloud")

    return pcd

def filter_globe(combined_pcd_path, output_path):

    pcd = o3d.io.read_point_cloud(combined_pcd_path)

    # Boundaries
    min_bound = np.array([14.0, -7.0, -6.0])
    max_bound = np.array([23.0, 2.0, 2.6])

    # 2. Create an Axis Aligned Bounding Box (AABB)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    # 3. Crop
    cropped_pcd = pcd.crop(bbox)

    points = np.asarray(cropped_pcd.points)

    print(f"Globe point cloud with {points.shape[0]} points.")

    print(f"Saving to {output_path}...")
    o3d.io.write_point_cloud(output_path, cropped_pcd)

    return points, cropped_pcd

def fit_spheroid(points):
    
    # Fit the globe's pointcloud to an ellipsoid

    # Initial guess of the centre and main radii
    center_guess = np.mean(points, axis=0)
    r_xy_guess = (np.max(points[:,0]) - np.min(points[:,0])) / 2
    r_z_guess = (np.max(points[:,2]) - np.min(points[:,2])) / 2
    
    x0 = [center_guess[0], center_guess[1], center_guess[2], r_xy_guess, r_z_guess]

    def residuals(params, p):

        
        cx, cy, cz, r_xy, r_z = params
        
        # Avoid division by zero warnings during optimization
        r_xy = max(abs(r_xy), 1e-5)
        r_z = max(abs(r_z), 1e-5)
        
        # Equation of spheroid: (x-cx)^2/r_xy^2 + ... - 1 = 0
        lhs = ((p[:,0] - cx)**2 + (p[:,1] - cy)**2) / (r_xy**2) + \
              ((p[:,2] - cz)**2) / (r_z**2)
        
        return lhs - 1.0 # Current value - target

    # 3. Run Optimization
    # 'soft_l1' loss is good for ignoring outliers (noise in the cloud)
    result = least_squares(residuals, x0, args=(points,), loss='soft_l1', f_scale=0.1)
    
    return result

def visualize_results(original_pcd, center, r_xy, r_z):
    
    # Visualise globe's point cloud and ellipsoid model

    # Ellipsoid creation
    cx, cy, cz = center
    
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=40)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([1, 0.1, 0.1]) 
    mesh_sphere = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_sphere)
    mesh_sphere.paint_uniform_color([1, 0.1, 0.1]) 
    
    # Apply transformation to the point cloud frame
    transform = np.identity(4)
    transform[0, 0] = r_xy
    transform[1, 1] = r_xy
    transform[2, 2] = r_z
    transform[0, 3] = cx
    transform[1, 3] = cy
    transform[2, 3] = cz
    
    mesh_sphere.transform(transform)

    # Visualise Open 3d    
    print("Opening visualization window...")
    o3d.visualization.draw_geometries([original_pcd, mesh_sphere], 
                                      window_name="Globe model")