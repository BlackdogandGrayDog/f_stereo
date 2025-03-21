import numpy as np
import cv2
import open3d as o3d

# 🔹 Set file paths
disparity_path = "hamlyn_outputs/rectified5/disp.npy"  # Path to disparity map (.npy)
rgb_path = "hamlyn/rectified5/L_00.jpg"  # Path to original RGB image
K_file = "hamlyn/rectified5/K.txt"  # Path to K.txt containing intrinsics & baseline
output_ply_path = "hamlyn_outputs/rectified5/point_cloud.ply"   # Output colored point cloud

def load_camera_params(K_file):
    """ Reads K.txt to get the intrinsic matrix and baseline. """
    with open(K_file, "r") as f:
        lines = f.readlines()
    
    # Parse intrinsic matrix (first line, 3×3 values)
    K_values = list(map(float, lines[0].strip().split()))
    K = np.array(K_values, dtype=np.float32).reshape(3, 3)

    # Parse baseline (second line)
    baseline = float(lines[1].strip())

    return K, baseline

def disparity_to_depth(disparity, fx, baseline):
    """ Converts disparity map to depth map (Z in meters). """
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid_mask = disparity > 0  # Ignore invalid disparity values
    depth[valid_mask] = (fx * baseline) / disparity[valid_mask]
    return depth

def generate_colored_point_cloud(disparity_file, rgb_file, K_file, output_ply):
    """ Loads disparity, converts to depth, and generates a 3D colored point cloud using original RGB colors. """
    # 🔹 Load K and baseline
    K, baseline = load_camera_params(K_file)
    fx = K[0, 0]  # Focal length in x

    # 🔹 Load disparity map
    disparity = np.load(disparity_file)

    # 🔹 Load RGB image
    rgb = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  # Convert from OpenCV BGR to RGB

    # 🔹 Compute depth map
    depth_map = disparity_to_depth(disparity, fx, baseline)

    # 🔹 Get image dimensions
    h, w = disparity.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))  # Pixel grid
    
    # 🔹 Convert pixel coordinates to normalized camera coordinates
    x = (i - K[0, 2]) / fx
    y = (j - K[1, 2]) / K[1, 1]
    
    # 🔹 Compute 3D points
    X = x * depth_map
    Y = y * depth_map
    Z = depth_map

    # 🔹 Create a mask for valid depth values
    valid_mask = (Z > 0) & (Z < 10)  # Ignore invalid or too far points

    # 🔹 Stack valid points into (N, 3) array
    points = np.stack((X[valid_mask], Y[valid_mask], Z[valid_mask]), axis=-1)

    # 🔹 Extract colors from original RGB image
    colors = rgb.reshape(-1, 3) / 255.0  # Normalize to [0,1]
    colors = colors[valid_mask.flatten()]  # Apply valid mask

    # 🔹 Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)  # Assign original image colors

    # 🔹 Save colored point cloud
    o3d.io.write_point_cloud(output_ply, pcd)
    print(f"✅ Colored point cloud (with original RGB) saved to {output_ply}")

# Run the function
generate_colored_point_cloud(disparity_path, rgb_path, K_file, output_ply_path)
