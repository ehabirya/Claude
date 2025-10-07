"""
RunPod Serverless Handler for Digital Twin Generation
"""

from __future__ import annotations
import base64
import io
import json
import logging
import math
from typing import List, Dict, Any

import numpy as np
import cv2
from PIL import Image, ImageOps
import runpod
import trimesh
from scipy.spatial import Delaunay

# ---------- Tuning knobs ----------
MIN_PIXELS = 500_000
MIN_BRIGHTNESS = 80.0
MAX_BRIGHTNESS = 200.0
MIN_SHARPNESS = 100.0

MAX_IMAGE_DIM = 1280  # resize larger photos to this max side
MAX_CORNERS = 400     # hard cap to keep matching fast
PATCH_RADIUS = 2      # 5x5 descriptor
MATCH_Y_TOL = 0.10
MATCH_DIST_MAX = 50.0
MIN_MATCHES = 20
SMOOTH_ITERS = 3
# ----------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("twin-handler")


def _resize_if_needed(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    m = max(h, w)
    if m <= MAX_IMAGE_DIM:
        return img_bgr
    scale = MAX_IMAGE_DIM / float(m)
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_AREA)


class PhotoQualityAnalyzer:
    @staticmethod
    def analyze(image_array: np.ndarray) -> Dict[str, Any]:
        height, width = image_array.shape[:2]
        resolution = height * width

        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(gray.std())

        issues = []
        if resolution < MIN_PIXELS:
            issues.append(f"Resolution too low (minimum {MIN_PIXELS:,} pixels)")
        if brightness < MIN_BRIGHTNESS:
            issues.append("Image too dark - need better lighting")
        if brightness > MAX_BRIGHTNESS:
            issues.append("Image too bright - reduce exposure")
        if sharpness < MIN_SHARPNESS:
            issues.append("Image too blurry - hold camera steady")

        return {
            "is_good": not issues,
            "issues": issues,
            "metrics": {
                "resolution": int(resolution),
                "brightness": brightness,
                "sharpness": sharpness,
                "contrast": contrast,
                "width": int(width),
                "height": int(height),
            },
        }


class PersonFeatureDetector:
    @staticmethod
    def detect_skin_mask(image_array: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(image_array, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        return cv2.inRange(ycrcb, lower, upper)

    @staticmethod
    def extract_features(image_array: np.ndarray) -> List[Dict[str, Any]]:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        skin_mask = PersonFeatureDetector.detect_skin_mask(image_array)

        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=MAX_CORNERS,
            qualityLevel=0.01,
            minDistance=10,
        )

        features: List[Dict[str, Any]] = []
        if corners is None:
            return features

        h, w = gray.shape[:2]
        r = PATCH_RADIUS
        for c in corners:
            x, y = c.ravel()
            xi, yi = int(x), int(y)
            # ensure patch fits
            if xi - r < 0 or yi - r < 0 or xi + r >= w or yi + r >= h:
                continue

            is_person = skin_mask[yi, xi] > 0
            patch = gray[yi - r : yi + r + 1, xi - r : xi + r + 1].astype(np.float32)
            features.append(
                {
                    "x": float(x / w),
                    "y": float(y / h),
                    "type": "person" if is_person else "edge",
                    "descriptor": patch.flatten().tolist(),
                }
            )
        return features


class FeatureMatcher:
    @staticmethod
    def match(features1: List[Dict[str, Any]], features2: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        f1 = [f for f in features1 if f["type"] == "person"]
        f2 = [f for f in features2 if f["type"] == "person"]
        if not f1 or not f2:
            return []

        dlen = min(len(f1[0]["descriptor"]), len(f2[0]["descriptor"]))
        A = np.array([np.array(f["descriptor"][:dlen], dtype=np.float32) for f in f1])
        B = np.array([np.array(f["descriptor"][:dlen], dtype=np.float32) for f in f2])

        y1 = np.array([f["y"] for f in f1], dtype=np.float32)[:, None]
        y2 = np.array([f["y"] for f in f2], dtype=np.float32)[None, :]
        y_ok = np.abs(y1 - y2) < MATCH_Y_TOL

        matches: List[Dict[str, Any]] = []
        if not y_ok.any():
            return matches

        dists = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        dists[~y_ok] = np.inf

        best_idx2 = np.argmin(dists, axis=1)
        best_dist = dists[np.arange(dists.shape[0]), best_idx2]

        for i, j in enumerate(best_idx2):
            dist = float(best_dist[i])
            if not math.isfinite(dist) or dist >= MATCH_DIST_MAX:
                continue
            matches.append({"point1": f1[i], "point2": f2[j], "distance": dist})
        return matches


class BodyMeshGenerator:
    @staticmethod
    def triangulate_points(matches: List[Dict[str, Any]]) -> np.ndarray:
        pts = []
        for m in matches:
            x = (m["point1"]["x"] + m["point2"]["x"]) / 2.0
            y = (m["point1"]["y"] + m["point2"]["y"]) / 2.0
            disparity = abs(m["point1"]["x"] - m["point2"]["x"])
            z = disparity * 1.5
            pts.append([x, y, z])
        return np.array(pts, dtype=np.float32)

    @staticmethod
    def create_mesh(points_3d: np.ndarray, measurements: Dict[str, float]) -> trimesh.Trimesh:
        if points_3d.shape[0] < 4:
            raise ValueError("Not enough points for mesh generation")

        height_scale = float(measurements.get("height", 170)) / 170.0
        weight_factor = float(measurements.get("weight", 70)) / 70.0
        width_scale = float(np.sqrt(max(0.1, weight_factor)))

        pts = points_3d.copy()
        pts[:, 0] = (pts[:, 0] - 0.5) * 2.0 * width_scale
        pts[:, 1] = (1.0 - pts[:, 1]) * 2.0 * height_scale
        pts[:, 2] = pts[:, 2] * 0.5

        try:
            tri = Delaunay(pts[:, :2])
        except Exception as e:
            raise ValueError(f"Delaunay triangulation failed: {e}")

        mesh = trimesh.Trimesh(vertices=pts, faces=tri.simplices, process=True)
        if SMOOTH_ITERS > 0 and len(mesh.vertices) > 0:
            trimesh.smoothing.filter_laplacian(mesh, iterations=SMOOTH_ITERS)
        return mesh


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to BGR numpy array with EXIF orientation and size normalization."""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]
        img_data = base64.b64decode(base64_string, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {e}")

    try:
        pil_img = Image.open(io.BytesIO(img_data))
        pil_img = ImageOps.exif_transpose(pil_img)  # respect camera orientation
        img = np.array(pil_img)
    except Exception as e:
        raise ValueError(f"Unable to decode image: {e}")

    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return _resize_if_needed(bgr)


def _default_measurements() -> Dict[str, float]:
    return {
        "height": 170,
        "weight": 70,
        "shoulderWidth": 40,
        "chestCircumference": 90,
        "waistCircumference": 80,
        "hipCircumference": 95,
    }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        job_input = job.get("input", {}) or {}
        action = job_input.get("action", "generate_twin")
        logger.info("Action=%s", action)

        if action == "analyze_photo":
            photo_b64 = job_input.get("photo")
            if not photo_b64:
                return {"error": "No photo provided"}
            img = decode_base64_image(photo_b64)
            return PhotoQualityAnalyzer.analyze(img)

        if action == "generate_twin":
            photos_b64 = job_input.get("photos") or []
            measurements = job_input.get("measurements") or _default_measurements()
            if len(photos_b64) != 3:
                return {"error": "Exactly 3 photos required (front, side, back)"}

            logger.info("Decoding images…")
            images = [decode_base64_image(p) for p in photos_b64]

            logger.info("Analyzing photo quality…")
            quality = [PhotoQualityAnalyzer.analyze(img) for img in images]
            for i, q in enumerate(quality, 1):
                if not q["is_good"]:
                    return {
                        "success": False,
                        "error": f"Photo {i} quality issues: {', '.join(q['issues'])}",
                        "quality_results": quality,
                    }

            logger.info("Extracting features…")
            feats = [PersonFeatureDetector.extract_features(img) for img in images]

            logger.info("Matching features…")
            m01 = FeatureMatcher.match(feats[0], feats[1])
            m12 = FeatureMatcher.match(feats[1], feats[2])
            m02 = FeatureMatcher.match(feats[0], feats[2])
            all_matches = m01 + m12 + m02

            if len(all_matches) < MIN_MATCHES:
                return {
                    "success": False,
                    "error": f"Not enough matching points ({len(all_matches)}). Ensure photos show same person clearly.",
                    "matches_found": len(all_matches),
                }

            logger.info("Creating 3D mesh (%d matches)…", len(all_matches))
            pts3d = BodyMeshGenerator.triangulate_points(all_matches)
            mesh = BodyMeshGenerator.create_mesh(pts3d, measurements)

            obj_str = trimesh.exchange.obj.export_obj(mesh)
            obj_b64 = base64.b64encode(obj_str.encode("utf-8")).decode("utf-8")

            return {
                "success": True,
                "mesh_obj_base64": obj_b64,
                "mesh_obj": obj_str,
                "statistics": {
                    "num_vertices": int(len(mesh.vertices)),
                    "num_faces": int(len(mesh.faces)),
                    "num_matches": int(len(all_matches)),
                },
                "quality_results": quality,
            }

        return {"error": f"Unknown action: {action}"}

    except Exception as e:
        logger.exception("Error in handler")
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
