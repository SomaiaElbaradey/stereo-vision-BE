# Stereo Vision Backend Service

This repository contains the backend service for a stereo vision application. Built with FastAPI, the service provides endpoints for camera calibration, image rectification, feature matching, stereo geometry estimation, 3D reconstruction, point cloud visualization, and filtering.

## Table of Contents

* [Features](#features)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Project Structure](#project-structure)
* [Usage](#usage)

  * [Running the Server](#running-the-server)
  * [API Endpoints](#api-endpoints)
* [Modules Overview](#modules-overview)
* [Requirements](#requirements)

## Features

* **Camera Calibration**: Estimate intrinsic parameters and distortion coefficients from chessboard images.
* **Image Rectification**: Align stereo images for disparity computation.
* **Feature Detection & Matching**: Detect SIFT features and match them between image pairs.
* **Stereo Geometry Estimation**: Compute essential matrix, relative pose, and filter inlier matches.
* **3D Reconstruction**: Triangulate matched points into a sparse 3D point cloud with color.
* **Point Cloud Visualization**: Generate PNG renderings of 3D point clouds.
* **Post-Processing & Filtering**: Outlier removal, ground plane removal, and optional visualization.
* **Disparity Map Computation**: Generate disparity images using block matching.

## Prerequisites

* Python 3.8 or higher
* pip

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/SomaiaElbaradey/stereo-vision-BE.git
   cd stereo-vision-BE
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
├── calibrator.py                  # Camera calibration utilities
├── rectifier.py                  # Stereo image rectification
├── matcher.py                    # Feature detection & matching
├── stereo_geometry_estimation.py # Estimate essential matrix & pose
├── reconstruction.py             # Triangulation & visualization
├── filtering_refining.py         # Point cloud filtering & post-processing
├── main.py                       # FastAPI application & endpoints
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation (this file)
```

## Usage

### Running the Server

Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload
```

By default, the server will run at `http://127.0.0.1:8000`.

### API Endpoints

#### 1. `POST /upload/`

Upload multiple chessboard images for camera calibration.

* **Request**: `multipart/form-data` with files field (list of images).
* **Response**:

  ```json
  {
    "K": [[...], [...], [...]],        # Intrinsic matrix
    "dist": [...],                     # Distortion coefficients
    "visual_results": ["<base64>"...] # Side-by-side calibration visuals
  }
  ```

#### 2. `POST /rectify/`

Rectify a pair of stereo images using stored calibration.

* **Request**: `multipart/form-data` with `left` and `right` image files.
* **Response**:

  ```json
  {
    "left": "<base64>",  # Rectified left image
    "right": "<base64>"  # Rectified right image
  }
  ```

#### 3. `POST /match-features/`

Detect and match SIFT features between two images.

* **Request**: `multipart/form-data` with `left_image` and `right_image`.
* **Response**:

  ```json
  {
    "matched_image": "<base64>",          # Visualization of matches
    "keypoints1": [[x,y],...],
    "keypoints2": [[x,y],...],
    "good_matches": [{"queryIdx": i, "trainIdx": j},...]
  }
  ```

#### 4. `POST /estimate-geometry/`

Estimate stereo geometry (essential matrix, rotation, translation).

* **Request Body** (JSON):

  ```json
  {
    "keypoints1": [[x,y],...],
    "keypoints2": [[x,y],...],
    "matches": [{"queryIdx": i, "trainIdx": j},...],
    "K": [[...], [...], [...]]
  }
  ```
* **Response**:

  ```json
  {
    "E": [[...]],      # Essential matrix
    "R": [[...]],      # Rotation matrix
    "t": [...],        # Translation vector
    "inlier_pts1": [...],
    "inlier_pts2": [...]
  }
  ```

#### 5. `POST /reconstruct-3d/`

Triangulate inlier matches into a sparse 3D point cloud.

* **Request**: `multipart/form-data` with `left`, `right` images and form fields:

  * `K`, `R`, `t` (JSON strings)
  * `pts1`, `pts2` (JSON string arrays of point coordinates)
* **Response**:

  ```json
  { "sparse_point_cloud": "<base64>" }  # PNG of the 3D scatter plot
  ```

#### 6. `POST /visualize-point-cloud/`

Render an arbitrary 3D point cloud.

* **Request Body** (JSON):

  ```json
  {
    "pts3d": [[x,y,z],...],
    "colors": [[r,g,b],...]   # Optional
  }
  ```
* **Response**:

  ```json
  { "visualization": "<base64>" }
  ```

#### 7. `POST /post-process/`

Apply robust filtering and optional ground removal to a point cloud.

* **Request Body** (JSON):

  ```json
  {
    "pts3d": [[x,y,z],...],
    "colors": [[r,g,b],...],    # Optional
    "remove_ground": true|false
  }
  ```
* **Response**:

  ```json
  { "filtered_point_cloud": "<base64>" }
  ```

#### 8. `POST /compute-disparity/`

Compute a disparity map using block matching.

* **Request**: `multipart/form-data` with `left`, `right` grayscale images and optional form fields:

  * `num_disparities` (multiple of 16)
  * `block_size` (odd integer)
* **Response**:

  ```json
  { "disparity_map": "<base64>" }
  ```

## Modules Overview

* **calibrator.py**: Camera calibration via chessboard detection.
* **rectifier.py**: Stereo rectification using calibration and baseline.
* **matcher.py**: SIFT feature extraction and FLANN-based matching.
* **stereo\_geometry\_estimation.py**: Essential matrix estimation and pose recovery.
* **reconstruction.py**: Triangulation, color assignment, and visualization.
* **filtering\_refining.py**: Outlier removal, ground filtering, and robust pipeline.

## Requirements

See `requirements.txt` for exact versions. Key dependencies:

* `fastapi`, `uvicorn`: Web framework and ASGI server
* `opencv-python-headless`, `numpy`, `matplotlib`: Computer vision and plotting
* `open3d`: Point cloud processing




