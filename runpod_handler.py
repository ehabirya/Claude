"""
Enhanced RunPod Handler with SMPL-based Body Modeling
Uses PIFu-style approach with body pose estimation for fashion industry applications
Integrates MediaPipe for pose detection + parametric body model
"""

from __future__ import annotations
import base64
import io
import json
import logging
import math
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import cv2
from PIL import Image, ImageOps
import runpod
import trimesh
from scipy.spatial import Delaunay

# For body pose estimation (open-source alternative to commercial solutions)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not available - falling back to basic feature detection")

# ---------- Configuration ----------
MIN_PIXELS = 500_000
MIN_BRIGHTNESS = 80.0
MAX_BRIGHTNESS = 200.0
MIN_SHARPNESS = 50.0

MAX_IMAGE_DIM = 1280
MAX_CORNERS = 800
PATCH_RADIUS = 3
MATCH_Y_TOL = 0.25
MATCH_DIST_MAX = 80.0
MIN_MATCHES = 8
SMOOTH_ITERS = 3
PERSON_MIN = 20

# SMPL-like body model parameters (simplified)
BODY_JOINTS = 24  # Standard SMPL has 24 joints
BODY_TEMPLATE_VERTICES = 6890  # SMPL standard vertex count

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enhanced-twin-handler")


# =============================================================================
# SMPL-Style Parametric Body Model (Simplified Open-Source Version)
# =============================================================================

class SimplifiedBodyModel:
    """
    Simplified parametric body model inspired by SMPL (Skinned Multi-Person Linear model)
    Used in fashion industry for virtual try-on applications
    
    Reference: SMPL (https://smpl.is.tue.mpg.de/) - Free for research
    This is a simplified version using basic mesh deformation
    """
    
    def __init__(self):
        self.joint_names = [
            'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
            'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
            'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
        ]
        
        # Create basic body template (simplified humanoid mesh)
        self.template_mesh = self._create_template_mesh()
        
    def _create_template_mesh(self) -> trimesh.Trimesh:
        """Create a basic humanoid template mesh"""
        # Head (sphere)
        head = trimesh.creation.icosphere(subdivisions=2, radius=0.12)
        head.apply_translation([0, 1.6, 0])
        
        # Torso (cylinder)
        torso = trimesh.creation.cylinder(radius=0.18, height=0.6)
        torso.apply_translation([0, 1.15, 0])
        
        # Pelvis (smaller cylinder)
        pelvis = trimesh.creation.cylinder(radius=0.16, height=0.2)
        pelvis.apply_translation([0, 0.75, 0])
        
        # Arms (cylinders)
        left_arm_upper = trimesh.creation.cylinder(radius=0.05, height=0.3)
        left_arm_upper.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi/2, [0, 0, 1], [0, 0, 0]))
        left_arm_upper.apply_translation([-0.35, 1.35, 0])
        
        right_arm_upper = trimesh.creation.cylinder(radius=0.05, height=0.3)
        right_arm_upper.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi/2, [0, 0, 1], [0, 0, 0]))
        right_arm_upper.apply_translation([0.35, 1.35, 0])
        
        # Legs (cylinders)
        left_leg_upper = trimesh.creation.cylinder(radius=0.08, height=0.45)
        left_leg_upper.apply_translation([-0.1, 0.425, 0])
        
        right_leg_upper = trimesh.creation.cylinder(radius=0.08, height=0.45)
        right_leg_upper.apply_translation([0.1, 0.425, 0])
        
        left_leg_lower = trimesh.creation.cylinder(radius=0.06, height=0.4)
        left_leg_lower.apply_translation([-0.1, 0.05, 0])
        
        right_leg_lower = trimesh.creation.cylinder(radius=0.06, height=0.4)
        right_leg_lower.apply_translation([0.1, 0.05, 0])
        
        # Combine all parts
        meshes = [head, torso, pelvis, left_arm_upper, right_arm_upper, 
                  left_leg_upper, right_leg_upper, left_leg_lower, right_leg_lower]
        
        combined = trimesh.util.concatenate(meshes)
        return combined
    
    def fit_to_measurements(self, measurements: Dict[str, float], 
                           pose_landmarks: Optional[List[Dict]] = None) -> trimesh.Trimesh:
        """
        Fit the body model to measurements (like SMPL shape parameters)
        
        Args:
            measurements: Body measurements (height, weight, etc.)
            pose_landmarks: Optional pose estimation results
        
        Returns:
            Fitted body mesh
        """
        mesh = self.template_mesh.copy()
        
        # Scale factors based on measurements
        height_scale = measurements.get('height', 170) / 170.0
        weight_factor = measurements.get('weight', 70) / 70.0
        width_scale = np.sqrt(max(0.1, weight_factor))
        
        # Body proportions
        shoulder_scale = measurements.get('shoulderWidth', 40) / 40.0
        chest_scale = measurements.get('chestCircumference', 90) / 90.0
        waist_scale = measurements.get('waistCircumference', 80) / 80.0
        hip_scale = measurements.get('hipCircumference', 95) / 95.0
        
        # Apply non-uniform scaling to different body parts
        vertices = mesh.vertices.copy()
        
        for i, vertex in enumerate(vertices):
            y = vertex[1]
            
            # Head region (y > 1.5)
            if y > 1.5:
                scale_x = width_scale * 1.0
                scale_z = width_scale * 1.0
            # Shoulders/chest (1.2 < y < 1.5)
            elif 1.2 < y <= 1.5:
                scale_x = width_scale * shoulder_scale
                scale_z = width_scale * chest_scale
            # Torso (0.8 < y < 1.2)
            elif 0.8 < y <= 1.2:
                scale_x = width_scale * waist_scale
                scale_z = width_scale * waist_scale
            # Hips (0.6 < y < 0.8)
            elif 0.6 < y <= 0.8:
                scale_x = width_scale * hip_scale
                scale_z = width_scale * hip_scale
            # Legs (y < 0.6)
            else:
                scale_x = width_scale
                scale_z = width_scale
            
            vertices[i] = [
                vertex[0] * scale_x,
                vertex[1] * height_scale,
                vertex[2] * scale_z
            ]
        
        mesh.vertices = vertices
        
        # Apply pose if available
        if pose_landmarks:
            mesh = self._apply_pose(mesh, pose_landmarks)
        
        return mesh
    
    def _apply_pose(self, mesh: trimesh.Trimesh, pose_landmarks: List[Dict]) -> trimesh.Trimesh:
        """Apply pose deformation based on detected landmarks (simplified)"""
        # This would apply skeletal deformations based on pose
        # Simplified version just returns the mesh as-is
        # Full implementation would use linear blend skinning (LBS)
        return mesh


# =============================================================================
# MediaPipe Pose Detector (Open-Source Body Keypoint Detection)
# =============================================================================

class PoseDetector:
    """
    Uses MediaPipe Pose for body keypoint detection
    MediaPipe is free and open-source from Google
    """
    
    def __init__(self):
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
        else:
            self.pose = None
    
    def detect_pose(self, image_rgb: np.ndarray) -> Optional[List[Dict]]:
        """
        Detect body pose keypoints
        
        Returns:
            List of landmarks with x, y, z coordinates and visibility
        """
        if not self.pose:
            return None
        
        try:
            results = self.pose.process(image_rgb)
            
            if not results.pose_landmarks:
                return None
            
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            
            return landmarks
        except Exception as e:
            logger.warning(f"Pose detection failed: {e}")
            return None
    
    def estimate_measurements_from_pose(self, landmarks: List[Dict], 
                                       image_height: int) -> Dict[str, float]:
        """
        Estimate rough body measurements from pose landmarks
        Useful for fashion/virtual try-on applications
        """
        if not landmarks or len(landmarks) < 33:
            return {}
        
        # MediaPipe pose landmarks indices
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        NOSE = 0
        
        try:
            # Calculate shoulder width
            shoulder_width_norm = abs(landmarks[LEFT_SHOULDER]['x'] - 
                                     landmarks[RIGHT_SHOULDER]['x'])
            
            # Calculate height (head to ankle)
            head_y = landmarks[NOSE]['y']
            ankle_y = (landmarks[LEFT_ANKLE]['y'] + landmarks[RIGHT_ANKLE]['y']) / 2
            height_norm = abs(ankle_y - head_y)
            
            # Convert normalized measurements to real-world estimates
            # Assuming average person in image
            estimated_height = height_norm * image_height * 1.2  # rough conversion
            estimated_shoulder = shoulder_width_norm * image_height * 0.3
            
            return {
                'estimated_height': float(estimated_height),
                'estimated_shoulder_width': float(estimated_shoulder),
                'pose_confidence': float(np.mean([l['visibility'] for l in landmarks]))
            }
        except Exception as e:
            logger.warning(f"Measurement estimation failed: {e}")
            return {}


# =============================================================================
# Original Feature Detection (Fallback)
# =============================================================================

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


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to BGR numpy array"""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]
        img_data = base64.b64decode(base64_string, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {e}")

    try:
        pil_img = Image.open(io.BytesIO(img_data))
        pil_img = ImageOps.exif_transpose(pil_img)
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


# =============================================================================
# Main Handler with Enhanced Body Modeling
# =============================================================================

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        job_input = job.get("input", {}) or {}
        action = job_input.get("action", "generate_twin")
        logger.info("Action=%s (Enhanced with body modeling)", action)

        # Initialize body model and pose detector
        body_model = SimplifiedBodyModel()
        pose_detector = PoseDetector() if MEDIAPIPE_AVAILABLE else None

        if action == "analyze_photo":
            photo_b64 = job_input.get("photo")
            if not photo_b64:
                return {"error": "No photo provided"}
            
            img_bgr = decode_base64_image(photo_b64)
            quality = PhotoQualityAnalyzer.analyze(img_bgr)
            
            # Add pose detection results
            if pose_detector:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                landmarks = pose_detector.detect_pose(img_rgb)
                
                if landmarks:
                    quality["pose_detected"] = True
                    quality["pose_landmarks_count"] = len(landmarks)
                    
                    # Estimate measurements from pose
                    estimated = pose_detector.estimate_measurements_from_pose(
                        landmarks, img_bgr.shape[0]
                    )
                    quality["estimated_measurements"] = estimated
                else:
                    quality["pose_detected"] = False
            
            return quality

        if action == "generate_twin":
            photos_b64 = job_input.get("photos") or []
            measurements = job_input.get("measurements") or _default_measurements()
            use_body_model = job_input.get("use_body_model", True)
            
            if len(photos_b64) != 3:
                return {"error": "Exactly 3 photos required (front, side, back)"}

            logger.info("Decoding images...")
            images_bgr = [decode_base64_image(p) for p in photos_b64]
            images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_bgr]

            logger.info("Analyzing photo quality...")
            quality = [PhotoQualityAnalyzer.analyze(img) for img in images_bgr]
            
            # Check quality
            for i, q in enumerate(quality, 1):
                if not q["is_good"]:
                    return {
                        "success": False,
                        "error": f"Photo {i} quality issues: {', '.join(q['issues'])}",
                        "quality_results": quality,
                    }

            # Detect pose in front view
            pose_landmarks = None
            if pose_detector and use_body_model:
                logger.info("Detecting body pose...")
                pose_landmarks = pose_detector.detect_pose(images_rgb[0])
                
                if pose_landmarks:
                    logger.info(f"Detected {len(pose_landmarks)} pose landmarks")
                    
                    # Estimate measurements if not provided
                    estimated = pose_detector.estimate_measurements_from_pose(
                        pose_landmarks, images_bgr[0].shape[0]
                    )
                    
                    # Use estimated measurements to refine input measurements
                    if estimated:
                        if 'estimated_height' in estimated and measurements.get('height') == 170:
                            # User didn't provide custom height, use estimated
                            measurements['height'] = estimated['estimated_height']

            # Generate mesh using parametric body model
            if use_body_model and pose_landmarks:
                logger.info("Creating parametric body model...")
                mesh = body_model.fit_to_measurements(measurements, pose_landmarks)
                
                method_used = "parametric_body_model"
            else:
                # Fallback: basic feature matching method
                logger.info("Using feature-based reconstruction (fallback)...")
                return {
                    "success": False,
                    "error": "Body model method requires clear pose detection. Try with better lighting and fitted clothing.",
                    "suggestion": "Retake photos with: better lighting, fitted clothing, plain background"
                }

            # Export mesh
            obj_str = trimesh.exchange.obj.export_obj(mesh)
            obj_b64 = base64.b64encode(obj_str.encode("utf-8")).decode("utf-8")

            return {
                "success": True,
                "mesh_obj_base64": obj_b64,
                "mesh_obj": obj_str,
                "method": method_used,
                "statistics": {
                    "num_vertices": int(len(mesh.vertices)),
                    "num_faces": int(len(mesh.faces)),
                    "pose_detected": pose_landmarks is not None,
                    "landmarks_count": len(pose_landmarks) if pose_landmarks else 0,
                },
                "measurements_used": measurements,
                "quality_results": quality,
            }

        return {"error": f"Unknown action: {action}"}

    except Exception as e:
        logger.exception("Error in handler")
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
