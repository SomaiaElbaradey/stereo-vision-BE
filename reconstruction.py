import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def triangulate_points(pts1, pts2, K, R, T):
    """
    Triangulate 3D points from matched image points using camera poses.
    
    Args:
        pts1: Points from first image (Nx2 array)
        pts2: Corresponding points from second image (Nx2 array)
        K: Camera intrinsic matrix (3x3)
        R: Rotation matrix between cameras (3x3)
        T: Translation vector between cameras (3x1)
        
    Returns:
        points_3d: Triangulated 3D points in world coordinates (Nx3 array)
    """
    # Ensure points are float32 and properly shaped
    pts1 = np.asarray(pts1, dtype=np.float32).reshape(-1, 2)
    pts2 = np.asarray(pts2, dtype=np.float32).reshape(-1, 2)
    
    # Create projection matrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # First camera: [I|0]
    P2 = K @ np.hstack((R, T.reshape(3, 1)))          # Second camera: [R|t]
    
    # Triangulate points (returns 4D homogeneous coordinates)
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    
    # Convert to 3D by dividing by last coordinate
    points_3d = (points_4d[:3] / points_4d[3]).T
    
    return points_3d

def visualize_point_cloud(points_3d, colors=None, title="3D Point Cloud"):
    """
    Visualize the 3D point cloud.
    
    Args:
        points_3d: Nx3 array of 3D points
        colors: Nx3 array of RGB colors (optional)
        title: Plot title
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Filter out invalid points (infinite or NaN)
    valid_mask = np.all(np.isfinite(points_3d), axis=1)
    points_3d = points_3d[valid_mask]
    
    if colors is not None:
        colors = colors[valid_mask]
    
    # Extract coordinates
    xs = points_3d[:, 0]
    ys = points_3d[:, 1]
    zs = points_3d[:, 2]
    
    # Plot points
    if colors is not None and len(colors) == len(points_3d):
        ax.scatter(xs, ys, zs, c=colors/255.0, s=1)
    else:
        ax.scatter(xs, ys, zs, s=1)
    
    # Set labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(title)
    
    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Set initial view
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()

def reconstruct_3d(left_img, right_img, K, R, T, pts1, pts2):
    """
    Complete 3D reconstruction pipeline from matched points.
    
    Args:
        left_img: Left image (for color information)
        right_img: Right image
        K: Camera matrix (3x3)
        R: Rotation matrix (3x3)
        T: Translation vector (3x1)
        pts1: Matched points in left image (shape: Nx1x2 or Nx2)
        pts2: Corresponding points in right image (shape: Nx1x2 or Nx2)
        
    Returns:
        points_3d: Triangulated 3D points (Nx3)
        colors: Corresponding colors for each point (Nx3)
    """
    # Convert points to proper Nx2 format
    pts1 = np.asarray(pts1, dtype=np.float32).reshape(-1, 2)
    pts2 = np.asarray(pts2, dtype=np.float32).reshape(-1, 2)
    
    # Triangulate 3D points
    points_3d = triangulate_points(pts1, pts2, K, R, T)
    
    # Get colors from left image
    colors = []
    for pt in pts1:
        x, y = int(np.round(pt[0])), int(np.round(pt[1]))
        if 0 <= x < left_img.shape[1] and 0 <= y < left_img.shape[0]:
            colors.append(left_img[y, x])
        else:
            colors.append([0, 0, 0])  # Black for out-of-bounds
    
    colors = np.array(colors)
    
    # Filter out points behind cameras (negative Z)
    valid_mask = points_3d[:, 2] > 0
    points_3d = points_3d[valid_mask]
    colors = colors[valid_mask]
    
    return points_3d, colors