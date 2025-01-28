import os
import numpy as np
from PIL import Image
import open3d as o3d

def get_intrinsics_from_json(json_path: str):
    """
    Reads the camera intrinsic matrix from a JSON file.
    
    Parameters:
        json_path (str): Path to the JSON file containing the intrinsic matrix.

    Returns:
        np.ndarray: Camera intrinsic matrix.
    """
    with open(json_path, "r") as f:
        camera_matrix = np.loadtxt(f)
    return camera_matrix

def create_point_cloud(rgb_image, depth_image, intrinsics, depth_scale=1000.0, max_depth=5.0):
    """
    Creates a point cloud from an RGB image and depth image using camera intrinsics.

    Parameters:
        rgb_image (np.ndarray): RGB image.
        depth_image (np.ndarray): Depth image.
        intrinsics (np.ndarray): Camera intrinsic matrix.
        depth_scale (float): Scaling factor for depth values.
        max_depth (float): Maximum depth threshold (meters).

    Returns:
        o3d.geometry.PointCloud: The generated point cloud.
    """
    h, w = depth_image.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Create a grid of pixel coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Convert depth values to meters
    z = depth_image.astype(np.float32) / depth_scale
    z[z > max_depth] = 0

    # Back-project to 3D coordinates
    x_3d = (x - cx) * z / fx
    y_3d = (y - cy) * z / fy

    # Stack to form a 3D point cloud
    points = np.stack((x_3d, y_3d, z), axis=-1).reshape(-1, 3)

    # Filter out points with zero depth
    valid_mask = z.reshape(-1) > 0
    points = points[valid_mask]

    # Get corresponding colors
    colors = rgb_image.reshape(-1, 3)[valid_mask] / 255.0

    # Create Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud

if __name__ == "__main__":
    # Define paths
    demo_path = "/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/demo_data/final_scene/plate_pivotrack"
    rgb_paths = sorted([os.path.join(demo_path, 'rgb', file) for file in os.listdir(os.path.join(demo_path, 'rgb')) if file.lower().endswith('.jpg')])
    depth_paths = sorted([os.path.join(demo_path, 'depth', file) for file in os.listdir(os.path.join(demo_path, 'depth')) if file.lower().endswith('.png')])
    
    cam_intrinsics_path = "/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/demo_data/cam_K.txt"

    # Load camera intrinsics
    intrinsics = get_intrinsics_from_json(cam_intrinsics_path)

    IDX = 90
    rgb_path = rgb_paths[IDX]
    depth_path = depth_paths[IDX]

    # Load RGB and depth images
    img_rgb = np.array(Image.open(rgb_path))
    img_depth = np.array(Image.open(depth_path))

    # Create point cloud
    pcd = create_point_cloud(img_rgb, img_depth, intrinsics)

    # Create coordinate frame for visualization
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    # Set camera view to point toward Z-axis
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Point Cloud")
    vis.add_geometry(pcd)
    vis.add_geometry(coordinate_frame)
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, -1])  # Camera pointing toward -Z
    view_control.set_up([0, -1, 0])    # Camera up is -Y
    view_control.set_lookat([0, 0, 0]) # Look at the origin

    vis.run()
    vis.destroy_window()


    # Visualize point cloud
    # o3d.visualization.draw_geometries([pcd], window_name=f"Point Cloud")
