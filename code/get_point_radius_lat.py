import numpy as np
import open3d as o3d
import cv2
from matplotlib import pyplot as plt

# Camera calibration matrix scaled to the projection img resolution
width, height = 839, 432
fx, fy = 599, 598
cx, cy = 444, 207

K = np.array([[fx,  0, cx],
              [ 0, fy, cy],
              [ 0,  0,  1]])

# Lidar -> camera transformation
# +1° in the camera tilt because of difference between the projection image and the video
T_lidar_to_cam = np.array([
    [-0.292, -0.956,  0.000,  3.000],
    [-0.050,  0.015, -0.998,  0.003],
    [ 0.955, -0.292, -0.052, -0.052],
    [ 0.000,  0.000,  0.000,  1.000]
])

search_radius = 3.0

def get_latitude_radius(frame, globe_pcd_path, CENTRE_OF_ROTATION, globe_center, globe_radius, n_target):

    # Calculate the latitude and radius of rotation for the input points in the globe's surfarface retrieved from the image

    # Scale the input points from the video frame to the image resolution used during calibration
    latitude_radius = np.ones(n_target)
    latitudes = np.ones(n_target)

    h, w, ch = frame.shape
    resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    x_scale = width / w
    y_scale = height / h
    scaled_globe_centre = (int(x_scale * globe_center[0]), int(y_scale * globe_center[1]))
    scaled_globe_radius = int(y_scale * globe_radius)
    original_radius_divisions = np.linspace(0, globe_radius * 2, n_target + 2)
    radius_divisions = np.linspace(0, scaled_globe_radius * 2, n_target + 2)
    original_target_points = (original_radius_divisions + globe_center[1] - globe_radius).astype(np.uint64)
    target_points = (radius_divisions + scaled_globe_centre[1] - scaled_globe_radius).astype(np.uint64)

    # Import the point cloud of the globe
    pcd = o3d.io.read_point_cloud(globe_pcd_path)
    points = np.asarray(pcd.points)

    # Transform the point cloud to the camera frame
    ones = np.ones((points.shape[0], 1))
    points_hom = np.hstack((points, ones))
    points_cam = (T_lidar_to_cam @ points_hom.T).T
    xyz_cam = points_cam[:, :3]

    # Filter points behind the camera
    valid_mask = xyz_cam[:, 2] > 0
    valid_indices = np.where(valid_mask)[0]
    xyz_valid = xyz_cam[valid_mask]

    # Project all points to the image plane in pixels
    u_proj = (fx * xyz_valid[:, 0] / xyz_valid[:, 2]) + cx
    v_proj = (fy * xyz_valid[:, 1] / xyz_valid[:, 2]) + cy

    text_height = 100
    text_colour = np.array([255, 20, 20])
    colour_tuple = text_colour.tolist()

    for i, point in enumerate(target_points[1:-1]): # Avoid mask limits

        # Calculate the closest projected point to the target pixel
        target_u = scaled_globe_centre[0]
        target_v = point
        dists = np.sqrt((u_proj - target_u)**2 + (v_proj - target_v)**2)
        min_dist_idx = np.argmin(dists)
        min_dist = dists[min_dist_idx]

        if min_dist < search_radius:
            
            # Map back to original points array
            original_idx = valid_indices[min_dist_idx]
            point_3d = points[original_idx]

            # Calculate the rotation radius
            radius = np.sqrt((CENTRE_OF_ROTATION[0] - point_3d[0])**2 + (CENTRE_OF_ROTATION[1] - point_3d[1])**2)

            if radius > 0:

                # Get the target pixel latitude hemisphere
                if point_3d[2] < CENTRE_OF_ROTATION[2]: latitude_sign = -1
                else: latitude_sign = 1

                # Calculate the target pixel latitude
                latitudes[i] = np.round(latitude_sign * np.degrees(np.atan(np.abs(CENTRE_OF_ROTATION[2] - point_3d[2]) / radius)),1)                   
                latitude_radius[i] = radius

            else: latitude_radius[i] = -1; latitudes[i] = -1

            text_height += 40
            text_colour[1] += 55
            colour_tuple = text_colour.tolist()

            if latitudes[i] > 0: latitude_text = latitudes[i].copy(); latitude_hemisphere = 'N'
            else: latitude_text = latitudes[i].copy() * -1; latitude_hemisphere = 'S'

            cv2.putText(rgb, f"{latitude_text} {latitude_hemisphere}", (80, text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour_tuple, 2)
            cv2.drawMarker(rgb, (scaled_globe_centre[0], point), color=colour_tuple, markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=2)
            
    return latitude_radius, original_target_points, latitudes, rgb