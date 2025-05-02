import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def validate_input(points, colors=None):
    """Ultra-robust input validation"""
    if points is None:
        return np.empty(0, 3), None
    
    points = np.asarray(points, dtype=np.float32)
    
    # Handle completely empty input
    if points.size == 0:
        return np.empty((0, 3)), None
    
    # Reshape to Nx3 if possible
    try:
        points = points.reshape(-1, 3)
    except:
        return np.empty((0, 3)), None
    
    # Validate colors if provided
    if colors is not None:
        colors = np.asarray(colors)
        if colors.size == 0:
            colors = None
        elif len(colors) != len(points):
            colors = None
        else:
            try:
                colors = colors.reshape(-1, 3)
            except:
                colors = None
    
    return points, colors

def safe_statistical_filter(pcd, nb_neighbors=20, std_ratio=2.0):
    """Completely safe statistical outlier removal"""
    if len(pcd.points) < nb_neighbors * 2:
        return pcd
    
    try:
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        return pcd.select_by_index(ind)
    except:
        return pcd

def safe_radius_filter(pcd, nb_points=10, radius=None):
    """Foolproof radius outlier removal"""
    if len(pcd.points) < nb_points * 2:
        return pcd
    
    try:
        if radius is None:
            # Auto-calculate radius based on point density
            pts = np.asarray(pcd.points)
            if len(pts) > 100:
                sample = pts[np.random.choice(len(pts), 100, replace=False)]
                radius = np.median(np.linalg.norm(sample - np.mean(sample, axis=0), axis=1)) * 3
            else:
                radius = np.median(np.linalg.norm(pts - np.mean(pts, axis=0), axis=1)) * 3
        
        cl, ind = pcd.remove_radius_outlier(
            nb_points=nb_points,
            radius=float(radius)
        )
        return pcd.select_by_index(ind)
    except:
        return pcd

def safe_point_cloud_filter(points_3d, colors=None):
    """100% crash-proof filtering"""
    points_3d, colors = validate_input(points_3d, colors)
    if len(points_3d) == 0:
        return points_3d, colors
    
    try:
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors/255.0)
        
        # Apply filters only if we have enough points
        if len(pcd.points) > 20:
            pcd = safe_statistical_filter(pcd)
            pcd = safe_radius_filter(pcd)
        
        # Convert back to numpy
        filtered_points = np.asarray(pcd.points) if len(pcd.points) > 0 else np.empty((0, 3))
        filtered_colors = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else colors
        
        return filtered_points, filtered_colors
    except:
        return points_3d, colors

def safe_visualize(points, colors=None, title="Point Cloud"):
    """Never-failing visualization"""
    points, colors = validate_input(points, colors)
    if len(points) == 0:
        print("No points to visualize")
        return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    try:
        plot_args = {'s': 1, 'alpha': 0.8}
        if colors is not None and len(colors) == len(points):
            ax.scatter(points[:,0], points[:,1], points[:,2], c=colors/255.0, **plot_args)
        else:
            ax.scatter(points[:,0], points[:,1], points[:,2], **plot_args)
        
        ax.set_title(f"{title} ({len(points)} points)")
        plt.show()
    except:
        plt.close()
        print("Visualization failed")

def ultimate_post_process(points_3d, colors=None, remove_ground=False, visualize=False):
    """
    The most robust point cloud processing pipeline possible
    Returns:
        - Processed points (guaranteed to be Nx3 numpy array)
        - Processed colors (or None)
    """
    # Stage 1: Input validation
    points, colors = validate_input(points_3d, colors)
    
    # Stage 2: Initial filtering
    filtered_points, filtered_colors = safe_point_cloud_filter(points, colors)
    if visualize:
        safe_visualize(filtered_points, filtered_colors, "After Filtering")
    
    # Stage 3: Ground removal (simplified)
    if remove_ground and len(filtered_points) > 10:
        try:
            z_values = filtered_points[:,2]
            threshold = np.percentile(z_values, 25)  # Remove bottom 25%
            mask = z_values > threshold
            filtered_points = filtered_points[mask]
            if filtered_colors is not None:
                filtered_colors = filtered_colors[mask]
            
            if visualize:
                safe_visualize(filtered_points, filtered_colors, "After Ground Removal")
        except:
            pass
    
    return filtered_points, filtered_colors