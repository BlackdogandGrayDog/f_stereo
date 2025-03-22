import numpy as np
from plyfile import PlyData

def compute_scale_from_ply(ply_path):
    # Load the PLY file
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    # Find min and max points
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)

    # Compute Euclidean distance between min and max
    scale = np.linalg.norm(max_coords - min_coords)

    print(f"Scale (distance between farthest points): {scale:.4f}")

if __name__ == "__main__":
    # 🔧 Set your PLY path here
    ply_path = "./simulator_outputs/trj_6/point_cloud.ply"
    
    compute_scale_from_ply(ply_path)
