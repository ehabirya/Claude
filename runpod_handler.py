"""
RunPod Serverless Handler for Digital Twin Generation
This handler processes requests on RunPod serverless infrastructure
"""

import runpod
import numpy as np
import cv2
from PIL import Image
import io
import base64
import trimesh
from scipy.spatial import Delaunay
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhotoQualityAnalyzer:
    @staticmethod
    def analyze(image_array):
        height, width = image_array.shape[:2]
        resolution = height * width
        
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        contrast = gray.std()
        
        issues = []
        
        if resolution < 500000:
            issues.append("Resolution too low (minimum 500k pixels)")
        if brightness < 80:
            issues.append("Image too dark - need better lighting")
        if brightness > 200:
            issues.append("Image too bright - reduce exposure")
        if sharpness < 100:
            issues.append("Image too blurry - hold camera steady")
            
        return {
            "is_good": len(issues) == 0,
            "issues": issues,
            "metrics": {
                "resolution": resolution,
                "brightness": float(brightness),
                "sharpness": float(sharpness),
                "contrast": float(contrast)
            }
        }


class PersonFeatureDetector:
    @staticmethod
    def detect_skin_tone(image_array):
        ycrcb = cv2.cvtColor(image_array, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
        return mask
    
    @staticmethod
    def extract_features(image_array):
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        skin_mask = PersonFeatureDetector.detect_skin_tone(image_array)
        
        corners = cv2.goodFeaturesToTrack(
            gray, 
            maxCorners=500, 
            qualityLevel=0.01, 
            minDistance=10
        )
        
        features = []
        height, width = image_array.shape[:2]
        
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                x_int, y_int = int(x), int(y)
                
                is_person = skin_mask[y_int, x_int] > 0
                
                x1, y1 = max(0, x_int-2), max(0, y_int-2)
                x2, y2 = min(width, x_int+3), min(height, y_int+3)
                patch = gray[y1:y2, x1:x2].flatten()
                
                features.append({
                    "x": float(x / width),
                    "y": float(y / height),
                    "type": "person" if is_person else "edge",
                    "descriptor": patch.tolist()
                })
        
        return features


class FeatureMatcher:
    @staticmethod
    def match(features1, features2):
        person_features1 = [f for f in features1 if f["type"] == "person"]
        person_features2 = [f for f in features2 if f["type"] == "person"]
        
        matches = []
        
        for f1 in person_features1:
            best_match = None
            best_distance = float('inf')
            
            for f2 in person_features2:
                desc1 = np.array(f1["descriptor"])
                desc2 = np.array(f2["descriptor"])
                min_len = min(len(desc1), len(desc2))
                
                distance = np.linalg.norm(desc1[:min_len] - desc2[:min_len])
                y_diff = abs(f1["y"] - f2["y"])
                
                if y_diff < 0.1 and distance < best_distance:
                    best_distance = distance
                    best_match = f2
            
            if best_match and best_distance < 50:
                matches.append({
                    "point1": f1,
                    "point2": best_match,
                    "distance": float(best_distance)
                })
        
        return matches


class BodyMeshGenerator:
    @staticmethod
    def triangulate_points(matches):
        points_3d = []
        
        for match in matches:
            x = (match["point1"]["x"] + match["point2"]["x"]) / 2
            y = (match["point1"]["y"] + match["point2"]["y"]) / 2
            disparity = abs(match["point1"]["x"] - match["point2"]["x"])
            z = disparity * 1.5
            
            points_3d.append([x, y, z])
        
        return np.array(points_3d)
    
    @staticmethod
    def create_mesh(points_3d, measurements):
        height_scale = measurements.get("height", 170) / 170
        weight_factor = measurements.get("weight", 70) / 70
        width_scale = np.sqrt(weight_factor)
        
        scaled_points = points_3d.copy()
        scaled_points[:, 0] = (scaled_points[:, 0] - 0.5) * 2 * width_scale
        scaled_points[:, 1] = (1 - scaled_points[:, 1]) * 2 * height_scale
        scaled_points[:, 2] = scaled_points[:, 2] * 0.5
        
        if len(scaled_points) < 4:
            raise ValueError("Not enough points for mesh generation")
        
        points_2d = scaled_points[:, :2]
        tri = Delaunay(points_2d)
        
        mesh = trimesh.Trimesh(
            vertices=scaled_points,
            faces=tri.simplices,
            process=True
        )
        
        trimesh.smoothing.filter_laplacian(mesh, iterations=3)
        
        return mesh


def decode_base64_image(base64_string):
    """Decode base64 string to numpy array"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    img_array = np.array(img)
    
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    elif img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    return img_array


def handler(job):
    """
    RunPod serverless handler function
    
    Expected input format:
    {
        "input": {
            "action": "generate_twin" or "analyze_photo",
            "photos": [base64_photo1, base64_photo2, base64_photo3],
            "photo": base64_photo,
            "measurements": {
                "height": 170,
                "weight": 70,
                "shoulderWidth": 40,
                "chestCircumference": 90,
                "waistCircumference": 80,
                "hipCircumference": 95
            }
        }
    }
    """
    try:
        job_input = job.get("input", {})
        action = job_input.get("action", "generate_twin")
        
        logger.info(f"Processing action: {action}")
        
        if action == "analyze_photo":
            photo_b64 = job_input.get("photo")
            if not photo_b64:
                return {"error": "No photo provided"}
            
            img_array = decode_base64_image(photo_b64)
            result = PhotoQualityAnalyzer.analyze(img_array)
            
            return result
            
        elif action == "generate_twin":
            photos_b64 = job_input.get("photos", [])
            measurements = job_input.get("measurements", {
                "height": 170,
                "weight": 70,
                "shoulderWidth": 40,
                "chestCircumference": 90,
                "waistCircumference": 80,
                "hipCircumference": 95
            })
            
            if len(photos_b64) != 3:
                return {"error": "Exactly 3 photos required (front, side, back)"}
            
            logger.info("Decoding images...")
            images = [decode_base64_image(photo) for photo in photos_b64]
            
            logger.info("Analyzing photo quality...")
            quality_analyzer = PhotoQualityAnalyzer()
            quality_results = [quality_analyzer.analyze(img) for img in images]
            
            for i, result in enumerate(quality_results):
                if not result["is_good"]:
                    return {
                        "success": False,
                        "error": f"Photo {i+1} quality issues: {', '.join(result['issues'])}",
                        "quality_results": quality_results
                    }
            
            logger.info("Extracting features...")
            feature_detector = PersonFeatureDetector()
            features = [feature_detector.extract_features(img) for img in images]
            
            logger.info("Matching features...")
            matcher = FeatureMatcher()
            matches_01 = matcher.match(features[0], features[1])
            matches_12 = matcher.match(features[1], features[2])
            matches_02 = matcher.match(features[0], features[2])
            
            all_matches = matches_01 + matches_12 + matches_02
            
            if len(all_matches) < 20:
                return {
                    "success": False,
                    "error": f"Not enough matching points ({len(all_matches)}). Ensure photos show same person clearly.",
                    "matches_found": len(all_matches)
                }
            
            logger.info(f"Found {len(all_matches)} matches")
            
            logger.info("Creating 3D mesh...")
            mesh_generator = BodyMeshGenerator()
            points_3d = mesh_generator.triangulate_points(all_matches)
            mesh = mesh_generator.create_mesh(points_3d, measurements)
            
            obj_data = trimesh.exchange.obj.export_obj(mesh)
            obj_base64 = base64.b64encode(obj_data.encode('utf-8')).decode('utf-8')
            
            logger.info("Digital twin generation complete")
            
            return {
                "success": True,
                "mesh_obj_base64": obj_base64,
                "mesh_obj": obj_data,
                "statistics": {
                    "num_vertices": len(mesh.vertices),
                    "num_faces": len(mesh.faces),
                    "num_matches": len(all_matches)
                },
                "quality_results": quality_results
            }
        
        else:
            return {"error": f"Unknown action: {action}"}
    
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}", exc_info=True)
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
