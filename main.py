import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from flask import jsonify, request
import numpy as np
import cv2
from calibrator import calibrate_camera

from fastapi.responses import JSONResponse
import base64
from rectifier import rectificate_images
from matcher import feature_detection_and_matching

from typing import List
from reconstruction import reconstruct_3d 

import io
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, Body

from stereo_geometry_estimation import estimate_stereo_geometry  
from reconstruction import reconstruct_3d                        
from reconstruction import visualize_point_cloud               
from filtering_refining import ultimate_post_process            


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
    left_image: UploadFile = File(...),
    right_image: UploadFile = File(...)
):
    # Read image bytes
    left_img_np = np.frombuffer(await left_image.read(), np.uint8)
    right_img_np = np.frombuffer(await right_image.read(), np.uint8)

    # Decode to OpenCV image
    left_img = cv2.imdecode(left_img_np, cv2.IMREAD_COLOR)
    right_img = cv2.imdecode(right_img_np, cv2.IMREAD_COLOR)

    # Call your function (assuming it's imported or defined)
    matched_img, good_matches, keypoints1, keypoints2 = feature_detection_and_matching(left_img, right_img)

    if matched_img is None:
        return JSONResponse(status_code=400, content={"error": "Not enough features detected"})

    # Encode matched image to base64
    _, buffer = cv2.imencode('.jpg', matched_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Extract keypoint coordinates
    keypoints1_coords = [kp.pt for kp in keypoints1]
    keypoints2_coords = [kp.pt for kp in keypoints2]

    # Extract match indices
    matches_data = [{'queryIdx': m.queryIdx, 'trainIdx': m.trainIdx} for m in good_matches]

    return {
        "matched_image": img_base64,
        "keypoints1": keypoints1_coords,
        "keypoints2": keypoints2_coords,
        "good_matches": matches_data
    }


def encode_fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def read_image_file(upload: UploadFile):
    data = np.frombuffer(upload.file.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

@app.post("/estimate-geometry/")
async def api_estimate_geometry(
    keypoints1: list[list[float]] = Body(...),
    keypoints2: list[list[float]] = Body(...),
    matches: list[dict]           = Body(...),
    K: list[list[float]]          = Body(...)
):
    try:
        # Build cv2.KeyPoint lists with a dummy size of 1
        kps1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in keypoints1]
        kps2 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in keypoints2]

        # Build cv2.DMatch objects
        cv_matches = []
        for m in matches:
            dm = cv2.DMatch()
            dm.queryIdx = int(m["queryIdx"])
            dm.trainIdx = int(m["trainIdx"])
            cv_matches.append(dm)

        Kmat = np.array(K, dtype=np.float64)

        # Call your stereo-geometry function
        E, R, t, in_pts1, in_pts2 = estimate_stereo_geometry(kps1, kps2, cv_matches, Kmat)

        return {
            "E": E.tolist(),
            "R": R.tolist(),
            "t": t.flatten().tolist(),
            "inlier_pts1": in_pts1.reshape(-1, 2).tolist(),
            "inlier_pts2": in_pts2.reshape(-1, 2).tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/reconstruct-3d/")
async def api_reconstruct_3d(
    left: UploadFile   = File(...),
    right: UploadFile  = File(...),
    K: str             = Form(...),
    R: str             = Form(...),
    t: str             = Form(...),
    pts1: str          = Form(...),
    pts2: str          = Form(...),
):
    try:
        # make this async
        async def read_img(u: UploadFile):
            data = await u.read()
            arr  = np.frombuffer(data, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

        left_img  = await read_img(left)
        right_img = await read_img(right)

        # parse your JSON form fields
        Kmat      = np.array(json.loads(K))
        Rmat      = np.array(json.loads(R))
        tvec      = np.array(json.loads(t)).reshape(3,1)
        pts1_arr  = np.array(json.loads(pts1), dtype=np.float32).reshape(-1,1,2)
        pts2_arr  = np.array(json.loads(pts2), dtype=np.float32).reshape(-1,1,2)

        pts3d, colors = reconstruct_3d(left_img, right_img, Kmat, Rmat, tvec, pts1_arr, pts2_arr)

        # render to base64 PNG...
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection="3d")
        valid = np.isfinite(pts3d).all(axis=1)
        p = pts3d[valid]; c = colors[valid]/255.0
        ax.scatter(p[:,0], p[:,1], p[:,2], c=c, s=1); ax.set_axis_off()
        buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")

        return {"sparse_point_cloud": b64}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/visualize-point-cloud/")
async def api_visualize_point_cloud(
    pts3d: list[list[float]] = Body(...),
    colors: list[list[int]] = Body(None)
):
    """
    Return a PNG of any arbitrary point-cloud.
    """
    pts = np.array(pts3d, dtype=np.float32)
    cols = np.array(colors, dtype=np.uint8) if colors is not None else None

    try:
        # use your module’s visualize, but capture the figure
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        mask = np.isfinite(pts).all(axis=1)
        p = pts[mask]
        if cols is not None:
            ax.scatter(p[:,0], p[:,1], p[:,2], c=cols[mask]/255.0, s=1)
        else:
            ax.scatter(p[:,0], p[:,1], p[:,2], s=1)
        ax.set_axis_off()
        img_b64 = encode_fig_to_base64(fig)
        return {"visualization": img_b64}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/post-process/")
async def api_post_process(
    pts3d: list[list[float]],
    colors: list[list[int]] = Body(None),
    remove_ground: bool       = Body(True)
):
    """
    Run your ultimate_post_process and return a PNG of the filtered cloud.
    """
    pts = np.array(pts3d, dtype=np.float32)
    cols = np.array(colors, dtype=np.uint8) if colors is not None else None

    try:
        f_pts, f_cols = ultimate_post_process(pts, cols, remove_ground=remove_ground, visualize=False)
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        if f_cols is not None:
            ax.scatter(f_pts[:,0], f_pts[:,1], f_pts[:,2], c=f_cols/255.0, s=1)
        else:
            ax.scatter(f_pts[:,0], f_pts[:,1], f_pts[:,2], s=1)
        ax.set_axis_off()
        img_b64 = encode_fig_to_base64(fig)
        return {"filtered_point_cloud": img_b64}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/compute-disparity/")
async def api_compute_disparity(
    left:  UploadFile = File(...),
    right: UploadFile = File(...),
    # you can tweak these or expose more params as needed
    num_disparities: int = Form(16),
    block_size:      int = Form(15),
):
    try:
        # async reader that yields a grayscale image
        async def read_gray(u: UploadFile):
            data = await u.read()
            arr  = np.frombuffer(data, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

        left_img  = await read_gray(left)
        right_img = await read_gray(right)

        # make sure num_disparities % 16 == 0 per OpenCV requirements
        num_disp = (num_disparities // 16) * 16
        if num_disp < 16:
            num_disp = 16

        # compute disparity
        stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
        disp   = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

        # normalize for visualization (0–255)
        disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
        disp_uint8 = np.uint8(disp_norm)

        # encode as PNG & base64
        success, buf = cv2.imencode('.png', disp_uint8)
        if not success:
            raise RuntimeError("Failed to encode disparity image")
        b64 = base64.b64encode(buf).decode('utf-8')

        return {"disparity_map": b64}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
