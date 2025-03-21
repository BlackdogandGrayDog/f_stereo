import cv2
import numpy as np

# 🔹 Manually specify image paths
left_img_path = "./hamlyn/rectified11/L_00.jpg"   # Change this to your left image path
right_img_path = "./hamlyn/rectified11/R_00.jpg"  # Change this to your right image path
output_path = "hamlyn_outputs/rectified11/SGBM_output_disparity.png"  # Change this to save the colorized disparity map

def compute_disparity(left_img_path, right_img_path, output_path):
    # Load stereo images
    imgL = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)  # Left Image
    imgR = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)  # Right Image

    if imgL is None or imgR is None:
        print("Error: Cannot load images. Check paths.")
        return
    
    # 🔹 SGBM Matcher Parameters (Adjust if needed)
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,  # Optimized for 5.49mm baseline
        blockSize=15,
        P1=4 * 3 * 9**2,  
        P2=16 * 3 * 9**2,  
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=16
    )

    # 🔹 Compute disparity
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0  # Normalize disparity

    # 🔹 Normalize disparity to 0-255 range for visualization
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)

    # 🔹 Apply a colormap for better visualization
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)  # Try COLORMAP_TURBO, COLORMAP_HOT, etc.

    # 🔹 Save the colorized disparity map
    cv2.imwrite(output_path, disp_color)
    print(f"✅ Colorized disparity map saved to {output_path}")

# Run the function
compute_disparity(left_img_path, right_img_path, output_path)
