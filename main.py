from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from calibrator import calibrate_camera

from fastapi.responses import JSONResponse
import base64
from rectifier import rectificate_images
from matcher import feature_detection_and_matching

from geometry import estimate_stereo_geometry 

from typing import List
from reconstruction import reconstruct_3d 

app = FastAPI()

# Allow frontend (React) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your frontend domain for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload(files: list[UploadFile] = File(...)):
    images = []

    for file in files:
        file_bytes = await file.read()
        np_img = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)

    try:
        result = calibrate_camera(images)
        return result
    except Exception as e:
        return {"error": str(e)}


def read_image_from_upload(upload_file: UploadFile):
    image_data = np.frombuffer(upload_file.file.read(), np.uint8)
    return cv2.imdecode(image_data, cv2.IMREAD_COLOR)

def encode_image_to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


@app.post("/rectify/")
async def rectify_images(
    left: UploadFile = File(...),
    right: UploadFile = File(...),
):
    # Load calibration parameters (from previous calibration)
    data = np.load("camera_calibration.npz", allow_pickle=True)
    K = data["K"]
    dist = data["dist"]
    # For demo purposes, R and T are identity and small baseline; ideally, load yours
    R = np.eye(3)
    T = np.array([[1.0], [0.0], [0.0]])

    left_img = read_image_from_upload(left)
    right_img = read_image_from_upload(right)

    rect_left, rect_right = rectificate_images(left_img, right_img, K, dist, R, T)

    return {
        "left": encode_image_to_base64(rect_left),
        "right": encode_image_to_base64(rect_right),
    }


@app.post("/match-features/")
async def match_features(
    left: UploadFile = File(...),
    right: UploadFile = File(...)
):
    left_img = read_image_from_upload(left)
    right_img = read_image_from_upload(right)

    matched_image, keypoints1, keypoints2 = feature_detection_and_matching(left_img, right_img)

    if matched_image is None:
        return JSONResponse(status_code=400, content={"error": "Insufficient features to match."})

    # Convert matched image to base64
    _, buffer = cv2.imencode(".jpg", matched_image)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    # Extract (x, y) from keypoints
    keypoints1_coords = [kp.pt for kp in keypoints1]
    keypoints2_coords = [kp.pt for kp in keypoints2]

    return {
        "matched_image": img_base64,
        "keypoints1": keypoints1_coords,
        "keypoints2": keypoints2_coords
    }

@app.post("/geometry/")
async def estimate_geometry(
    pts1: list[float] = Form(...),   # Flattened list [x1, y1, x2, y2, ...]
    pts2: list[float] = Form(...),
    k: list[float] = Form(...),      # Flattened 3x3
):
    # Reshape data
    pts1_np = np.array(pts1, dtype=np.float32).reshape(-1, 1, 2)
    pts2_np = np.array(pts2, dtype=np.float32).reshape(-1, 1, 2)
    K = np.array(k, dtype=np.float64).reshape(3, 3)

    # Create dummy keypoints for compatibility
    class DummyKeyPoint:
        def __init__(self, x, y): self.pt = (x, y)

    keypoints1 = [DummyKeyPoint(x, y) for x, y in pts1_np.squeeze()]
    keypoints2 = [DummyKeyPoint(x, y) for x, y in pts2_np.squeeze()]
    good_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(keypoints1))]

    # Estimate geometry
    E, R, T, inlier_pts1, inlier_pts2 = estimate_stereo_geometry(keypoints1, keypoints2, good_matches, K)

    return JSONResponse(content={
        "E": E.tolist(),
        "R": R.tolist(),
        "T": T.tolist(),
        "inlier_pts1": inlier_pts1.reshape(-1, 2).tolist(),
        "inlier_pts2": inlier_pts2.reshape(-1, 2).tolist()
    })


@app.post("/reconstruct/")
async def reconstruct_3d_endpoint(
    left_img_file: UploadFile = File(...),
    right_img_file: UploadFile = File(...),
    pts1: List[float] = Form(...),  # Flattened list [x1, y1, x2, y2, ...]
    pts2: List[float] = Form(...),
    k: List[float] = Form(...),     # Flattened 3x3
    r: List[float] = Form(...),
    t: List[float] = Form(...)
):
    # Decode images
    left_img_bytes = await left_img_file.read()
    right_img_bytes = await right_img_file.read()

    left_img_np = cv2.imdecode(np.frombuffer(left_img_bytes, np.uint8), cv2.IMREAD_COLOR)
    right_img_np = cv2.imdecode(np.frombuffer(right_img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Parse parameters
    pts1_np = np.array(pts1, dtype=np.float32).reshape(-1, 2)
    pts2_np = np.array(pts2, dtype=np.float32).reshape(-1, 2)
    K = np.array(k, dtype=np.float64).reshape(3, 3)
    R = np.array(r, dtype=np.float64).reshape(3, 3)
    T = np.array(t, dtype=np.float64).reshape(3, 1)

    # Run reconstruction
    points_3d, colors = reconstruct_3d(left_img_np, right_img_np, K, R, T, pts1_np, pts2_np)

    # Serialize result for frontend (example: return first 100 points)
    response_data = {
        "points": points_3d[:100].tolist(),
        "colors": colors[:100].tolist()
    }

    return JSONResponse(content=response_data)
