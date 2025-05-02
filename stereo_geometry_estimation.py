import cv2
import numpy as np

def estimate_stereo_geometry(keypoints1, keypoints2, good_matches, K):
    """
    Estimate essential matrix and camera poses for same camera system.
    
    Parameters:
        keypoints1: Keypoints from left image
        keypoints2: Keypoints from right image
        good_matches: Filtered feature matches
        K: Camera matrix from calibration
    
    Returns:
        E: Essential matrix
        R: Relative rotation
        t: Relative translation
        pts1, pts2: Filtered matched points
    """
    # Convert matches to points
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find fundamental matrix with RANSAC
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.5, 0.99)
    
    # Filter inliers
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    
    # Compute essential matrix (same K for both cameras)
    E = K.T @ F @ K
    
    # Recover pose (automatically scales translation)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    
    print("\n=== Geometry Estimation Results ===")
    print(f"Essential Matrix:\n{E}")
    print(f"Rotation:\n{R}")
    print(f"Translation:\n{t}")
    
    return E, R, t, pts1, pts2