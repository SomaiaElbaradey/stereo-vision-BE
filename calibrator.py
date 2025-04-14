import cv2
import numpy as np
import base64

def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def calibrate_camera(images, pattern_size=(9, 7), square_size=15.0):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []
    original_images = []
    chessboard_images = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            corners_subpix = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            objpoints.append(objp)
            imgpoints.append(corners_subpix)
            original_images.append(img)

            # Draw corners on original for visual verification
            drawn = img.copy()
            cv2.drawChessboardCorners(drawn, pattern_size, corners_subpix, ret)
            chessboard_images.append(drawn)

    if len(objpoints) < 5:
        raise ValueError("Need at least 5 valid chessboard images for calibration.")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    result_images = []

    for orig, drawn in zip(original_images, chessboard_images):
        h, w = orig.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        undistorted = cv2.undistort(orig, K, dist, None, new_camera_matrix)

        # Crop based on ROI
        x, y, w, h = roi
        undistorted_cropped = undistorted[y:y+h, x:x+w]

        # Side-by-side visualization
        side_by_side = cv2.hconcat([cv2.resize(drawn, (w, h)), cv2.resize(undistorted_cropped, (w, h))])
        result_images.append(encode_image(side_by_side))

    return {
        "K": K.tolist(),
        "dist": dist.tolist(),
        "visual_results": result_images
    }
