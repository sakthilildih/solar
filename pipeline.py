
import os
import math
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultralytics import YOLO
from shapely.geometry import Polygon, Point
from shapely.affinity import rotate, translate
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import cv2
import time

ARCGIS_SERVICE_URL = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"

MAPBOX_TOKEN = "pk.eyJ1Ijoic2FrdGhpbGlsZGloIiwiYSI6ImNtaXBzMDU2eTBkOWQzZ3I0bW4wZ2IzYmwifQ.IPPxwMUhC_fSX9O3n7vo8g"
MAPBOX_ZOOM = 18

USE_DUAL_SOURCE = True
DUAL_SOURCE_STRATEGY = "OR"

MODEL_PATHS = ["best.pt", "best (2).pt"]
USE_ENSEMBLE = True
ENSEMBLE_STRATEGY = "max_confidence"

INPUT_XLSX = "input/sites.xlsx"
OUTPUT_DIR = "output"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
ARTIFACTS_DIR = os.path.join(OUTPUT_DIR, "artifacts")
JSON_DIR = os.path.join(OUTPUT_DIR, "json")

IMG_WIDTH = 512
IMG_HEIGHT = 512
KM_RADIUS = 0.04

BUFFER_ZONE_1_SQFT = 1200
BUFFER_ZONE_2_SQFT = 2400

GSD_CM_PER_PIXEL = 10.88
GSD_M_PER_PIXEL = GSD_CM_PER_PIXEL / 100

SQFT_TO_SQM_CONVERSION = 0.092903

WP_PER_SQM_ASSUMPTION = 175

CALCULATE_CAPACITY = True

CONFIDENCE_THRESHOLD = 0.3

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

def calculate_buffer_radii():
    """Calculate buffer zone radii in pixels from square feet areas."""
    buffer_zone_1_sqm = BUFFER_ZONE_1_SQFT * SQFT_TO_SQM_CONVERSION
    buffer_zone_2_sqm = BUFFER_ZONE_2_SQFT * SQFT_TO_SQM_CONVERSION
    
    # Area = pi * r^2 => r = sqrt(Area / pi)
    radius_1_m = math.sqrt(buffer_zone_1_sqm / math.pi)
    radius_2_m = math.sqrt(buffer_zone_2_sqm / math.pi)
    
    # Convert from meters to pixels
    radius_1_pixels = radius_1_m / GSD_M_PER_PIXEL
    radius_2_pixels = radius_2_m / GSD_M_PER_PIXEL
    
    return radius_1_pixels, radius_2_pixels


def load_models(model_paths):
    """
    Load multiple YOLO models for ensemble prediction.
    
    Args:
        model_paths: List of paths to model files
    
    Returns:
        List of tuples (model_name, model_object)
    """
    models = []
    for path in model_paths:
        try:
            model = YOLO(path)
            models.append((path, model))
            print(f"   ✅ Loaded model: {path}")
        except Exception as e:
            print(f"   ⚠️ Failed to load {path}: {e}")
    return models


def predict_with_model(model_info, image_path, confidence_threshold):
    """
    Run prediction with a single model (thread-safe).
    
    Args:
        model_info: Tuple of (model_name, model_object)
        image_path: Path to image
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        Dictionary with prediction results
    """
    model_name, model = model_info
    try:
        results = model.predict(
            source=image_path,
            save=False,
            conf=confidence_threshold,
            verbose=False
        )
        return {
            'model_name': model_name,
            'result': results[0],
            'success': True
        }
    except Exception as e:
        return {
            'model_name': model_name,
            'error': str(e),
            'success': False
        }


def ensemble_predictions(predictions, strategy="max_confidence"):
    """
    Combine predictions from multiple models.
    
    Args:
        predictions: List of prediction dictionaries from different models
        strategy: Ensemble strategy ("max_confidence", "average", "voting")
    
    Returns:
        Combined prediction result with ensemble metadata
    """

    successful_preds = [p for p in predictions if p['success']]
    
    if not successful_preds:
        return None, []
    

    if strategy == "max_confidence":
        best_pred = None
        best_score = -1
        contributing_models = []
        
        for pred in successful_preds:
            result = pred['result']
            if result.obb and len(result.obb.xywhr) > 0:
                max_conf = float(result.obb.conf.max())
                if max_conf > best_score:
                    best_score = max_conf
                    best_pred = pred
                    contributing_models = [pred['model_name']]
        
        if best_pred:
            return best_pred['result'], contributing_models
        else:
        
            return successful_preds[0]['result'], []
    
    return successful_preds[0]['result'], [successful_preds[0]['model_name']]


def download_image_arcgis(lat, lon, save_path):
    """
    Download satellite image from ArcGIS World Imagery API for given coordinates.
    Uses direct REST API calls - no authentication required.
    
    Args:
        lat: Latitude of the center point
        lon: Longitude of the center point
        save_path: Path where the image will be saved
    
    Returns:
        Tuple of (success: bool, message: str)
    """

    MIN_RADIUS = 0.15  # km
    km_radius = max(KM_RADIUS, MIN_RADIUS)
    

    delta = km_radius / 110.0  # ~1° ≈ 110 km
    
    params = {
        'bbox': f"{lon - delta},{lat - delta},{lon + delta},{lat + delta}",
        'bboxSR': '4326',  # WGS84
        'size': f'{IMG_WIDTH},{IMG_HEIGHT}',
        'imageSR': '4326',
        'format': 'png',
        'f': 'image'
    }
    
    try:
        response = requests.get(ARCGIS_SERVICE_URL, params=params, timeout=30)
        response.raise_for_status()
        
        # Save the image using PIL to ensure proper format
        img = Image.open(BytesIO(response.content))
        img.save(save_path)
        
        return True, "Success"
    except requests.exceptions.RequestException as e:
        return False, f"HTTP Error: {str(e)}"
    except Exception as e:
        return False, str(e)


def download_image_mapbox(lat, lon, save_path):
    """
    Download satellite image from Mapbox API for given coordinates.
    
    Args:
        lat: Latitude of the center point
        lon: Longitude of the center point
        save_path: Path where the image will be saved
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    img_size = f"{IMG_WIDTH}x{IMG_HEIGHT}"
    url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{MAPBOX_ZOOM}/{img_size}?access_token={MAPBOX_TOKEN}"
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        with open(save_path, "wb") as f:
            f.write(response.content)
        
        return True, "Success"
    except requests.exceptions.RequestException as e:
        return False, f"HTTP Error: {str(e)}"
    except Exception as e:
        return False, str(e)


def download_images_dual_source(lat, lon, arcgis_path, mapbox_path):
    """
    Download satellite images from both ArcGIS and Mapbox simultaneously.
    
    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        arcgis_path: Save path for ArcGIS image
        mapbox_path: Save path for Mapbox image
    
    Returns:
        Dictionary with download results for both sources
    """
    results = {
        "arcgis": {"success": False, "message": ""},
        "mapbox": {"success": False, "message": ""}
    }
    
    # Download from ArcGIS
    arcgis_success, arcgis_msg = download_image_arcgis(lat, lon, arcgis_path)
    results["arcgis"]["success"] = arcgis_success
    results["arcgis"]["message"] = arcgis_msg
    
    # Download from Mapbox
    mapbox_success, mapbox_msg = download_image_mapbox(lat, lon, mapbox_path)
    results["mapbox"]["success"] = mapbox_success
    results["mapbox"]["message"] = mapbox_msg
    
    return results


def calculate_overlap_area_sqm(panel_obb, buffer_center_x, buffer_center_y, buffer_radius_pixels, gsd_m_per_pixel):
    """
    Calculate the overlap area between a panel OBB and circular buffer zone.
    
    Args:
        panel_obb: Tuple (cx, cy, w, h, r) - center x, center y, width, height, rotation
        buffer_center_x: X coordinate of buffer zone center
        buffer_center_y: Y coordinate of buffer zone center
        buffer_radius_pixels: Radius of buffer zone in pixels
        gsd_m_per_pixel: Ground Sample Distance in meters per pixel
    
    Returns:
        Overlap area in square meters
    """
    cx, cy, w, h, r = panel_obb
    
    
    half_w = w / 2
    half_h = h / 2
    unrotated_polygon = Polygon([
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h)
    ])
    
 
    rotated_polygon = rotate(unrotated_polygon, r, origin='center', use_radians=False)
    
   
    panel_polygon = translate(rotated_polygon, xoff=cx, yoff=cy)
    
   
    buffer_center_point = Point(buffer_center_x, buffer_center_y)
    buffer_circle = buffer_center_point.buffer(buffer_radius_pixels, resolution=16)
    

    intersection = panel_polygon.intersection(buffer_circle)
    intersection_area_pixels = intersection.area
    

    overlap_area_sqm = intersection_area_pixels * (gsd_m_per_pixel ** 2)
    
    return overlap_area_sqm


def determine_qc_status(has_solar, confidence, image_path):
    """
    Determine QC status based on detection results and image quality.
    
    Returns:
        "VERIFIABLE" or "NOT_VERIFIABLE"
    """
    if not os.path.exists(image_path):
        return "NOT_VERIFIABLE"
    
    try:
        img = Image.open(image_path)
  
        if img.size[0] < 256 or img.size[1] < 256:
            return "NOT_VERIFIABLE"
    except:
        return "NOT_VERIFIABLE"
    
    if has_solar and confidence >= 0.2:
        return "VERIFIABLE"
    
    if not has_solar:
        return "VERIFIABLE"
   
    if has_solar and confidence < 0.2:
        return "NOT_VERIFIABLE"
    
    return "VERIFIABLE"


def create_audit_overlay(image_path, result, buffer_center_x, buffer_center_y, 
                         radius_1_pixels, radius_2_pixels, has_solar, 
                         relevant_buffer_radius_sqft, best_panel_confidence, 
                         total_panel_area_sqm, best_bbox_or_mask, output_path):
    """
    Create audit-friendly overlay image with detections and buffer zones.
    """
    try:
        
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        overlay = img.copy()
        

        cv2.circle(overlay, 
                  (int(buffer_center_x), int(buffer_center_y)), 
                  int(radius_1_pixels), 
                  (255, 0, 0), 2)
        cv2.circle(overlay, 
                  (int(buffer_center_x), int(buffer_center_y)), 
                  int(radius_2_pixels), 
                  (0, 255, 0), 2)
        
        # Draw detected panels (OBBs)
        if result.obb and len(result.obb.xywhr) > 0:
            for i, det_xywhr in enumerate(result.obb.xywhr):
                cx, cy, w, h, r = det_xywhr.cpu().numpy()
                confidence = result.obb.conf[i].item()
                
                # Calculate box corners
                angle_rad = math.radians(r)
                corners = []
                for dx, dy in [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]:
                    # Rotate
                    rx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
                    ry = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
                    # Translate
                    px = int(cx + rx)
                    py = int(cy + ry)
                    corners.append((px, py))
                
                # Draw OBB - Yellow for best match, Red for others
                color = (0, 255, 255) if best_bbox_or_mask and abs(cx - best_bbox_or_mask[0]) < 1 else (0, 0, 255)
                pts = np.array(corners, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts], True, color, 2)
                
                # Add confidence label
                label = f"{confidence:.2f}"
                cv2.putText(overlay, label, (int(cx), int(cy)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add text annotations
        y_offset = 30
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0)
        
        annotations = [
            f"Has Solar: {has_solar}",
            f"Buffer: {relevant_buffer_radius_sqft} sq.ft" if relevant_buffer_radius_sqft else "Buffer: N/A",
            f"Confidence: {best_panel_confidence:.2f}" if has_solar else "No Detection",
            f"Area: {total_panel_area_sqm:.2f} sqm" if has_solar else ""
        ]
        
        for text in annotations:
            if text:
                # Background rectangle
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(overlay, (10, y_offset - 20), (20 + text_width, y_offset + 5), bg_color, -1)
                # Text
                cv2.putText(overlay, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                y_offset += 30
        
        # Save overlay
        cv2.imwrite(output_path, overlay)
        return True
    except Exception as e:
        print(f"   ⚠️ Failed to create overlay: {e}")
        return False


def detect_solar_panels(image_path, models, radius_1_pixels, radius_2_pixels):
    """
    Run YOLO inference with ensemble of models and detect solar panels within buffer zones.
    Uses multi-threading for parallel prediction.
    
    Args:
        image_path: Path to the image
        models: List of (model_name, model_object) tuples
        radius_1_pixels: Inner buffer radius in pixels
        radius_2_pixels: Outer buffer radius in pixels
    
    Returns:
        Dictionary with detection results and ensemble metadata
    """
    start_time = time.time()
    
    # Run predictions in parallel using ThreadPoolExecutor
    predictions = []
    
    if USE_ENSEMBLE and len(models) > 1:
        print(f"   🔄 Running ensemble prediction with {len(models)} models...")
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            # Submit all prediction tasks
            future_to_model = {
                executor.submit(predict_with_model, model_info, image_path, CONFIDENCE_THRESHOLD): model_info[0]
                for model_info in models
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    pred_result = future.result()
                    predictions.append(pred_result)
                    if pred_result['success']:
                        print(f"      ✅ {model_name}: Complete")
                    else:
                        print(f"      ❌ {model_name}: {pred_result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"      ❌ {model_name}: Exception - {e}")
                    predictions.append({
                        'model_name': model_name,
                        'error': str(e),
                        'success': False
                    })
    else:
        # Single model mode
        print(f"   🔍 Running single model prediction...")
        pred_result = predict_with_model(models[0], image_path, CONFIDENCE_THRESHOLD)
        predictions.append(pred_result)
    
    prediction_time = time.time() - start_time
    print(f"   ⏱️  Prediction time: {prediction_time:.2f}s")
    
    # Combine predictions using ensemble strategy
    result, contributing_models = ensemble_predictions(predictions, ENSEMBLE_STRATEGY)
    
    if result is None:
        # All predictions failed, return empty result
        return {
            "result": None,
            "has_solar": False,
            "confidence": 0.0,
            "pv_area_sqm_est": 0.0,
            "buffer_radius_sqft": BUFFER_ZONE_2_SQFT,
            "bbox_or_mask": [],
            "buffer_center_x": 0,
            "buffer_center_y": 0,
            "ensemble_metadata": {
                "models_used": [m[0] for m in models],
                "strategy": ENSEMBLE_STRATEGY,
                "contributing_models": [],
                "prediction_time_s": prediction_time
            }
        }
    
    # Image dimensions
    img_width = result.orig_shape[1]
    img_height = result.orig_shape[0]
    image_center_x = img_width / 2
    image_center_y = img_height / 2
    
    # Initialize variables
    has_solar = False
    relevant_buffer_radius_sqft = None
    total_panel_area_sqm = 0.0
    highest_overlap_area_sqm = 0.0
    best_panel_confidence = 0.0
    best_bbox_or_mask = []
    capacity_kw = 0.0
    found_in_1200_buffer = False
    
    # Check 1200 sq.ft buffer zone first
    if result.obb and len(result.obb.xywhr) > 0:
        for i, det_xywhr in enumerate(result.obb.xywhr):
            cx, cy, w, h, r = det_xywhr.cpu().numpy()
            confidence = result.obb.conf[i].item()
            
            overlap_area_sqm = calculate_overlap_area_sqm(
                (cx, cy, w, h, r),
                image_center_x,
                image_center_y,
                radius_1_pixels,
                GSD_M_PER_PIXEL
            )
            
            if overlap_area_sqm > highest_overlap_area_sqm:
                highest_overlap_area_sqm = overlap_area_sqm
                total_panel_area_sqm = (w * h) * (GSD_M_PER_PIXEL ** 2)
                has_solar = True
                relevant_buffer_radius_sqft = BUFFER_ZONE_1_SQFT
                best_panel_confidence = confidence
                best_bbox_or_mask = [float(cx), float(cy), float(w), float(h), float(r)]
                found_in_1200_buffer = True
    
    # If no solar in 1200 sq.ft buffer, check 2400 sq.ft buffer
    if not found_in_1200_buffer:
        highest_overlap_area_sqm = 0.0
        total_panel_area_sqm = 0.0
        best_panel_confidence = 0.0
        best_bbox_or_mask = []
        has_solar = False
        
        if result.obb and len(result.obb.xywhr) > 0:
            for i, det_xywhr in enumerate(result.obb.xywhr):
                cx, cy, w, h, r = det_xywhr.cpu().numpy()
                confidence = result.obb.conf[i].item()
                
                overlap_area_sqm = calculate_overlap_area_sqm(
                    (cx, cy, w, h, r),
                    image_center_x,
                    image_center_y,
                    radius_2_pixels,
                    GSD_M_PER_PIXEL
                )
                
                if overlap_area_sqm > highest_overlap_area_sqm:
                    highest_overlap_area_sqm = overlap_area_sqm
                    total_panel_area_sqm = (w * h) * (GSD_M_PER_PIXEL ** 2)
                    has_solar = True
                    relevant_buffer_radius_sqft = BUFFER_ZONE_2_SQFT
                    best_panel_confidence = confidence
                    best_bbox_or_mask = [float(cx), float(cy), float(w), float(h), float(r)]
    
    # Default buffer if no solar found
    if not has_solar:
        relevant_buffer_radius_sqft = BUFFER_ZONE_2_SQFT
    
    # Calculate capacity in kW if solar panels detected
    if has_solar and CALCULATE_CAPACITY:
        capacity_kw = (total_panel_area_sqm * WP_PER_SQM_ASSUMPTION) / 1000
    
    return {
        "result": result,
        "has_solar": has_solar,
        "confidence": best_panel_confidence,
        "pv_area_sqm_est": total_panel_area_sqm,
        "capacity_kw": capacity_kw,
        "buffer_radius_sqft": relevant_buffer_radius_sqft,
        "bbox_or_mask": best_bbox_or_mask,
        "buffer_center_x": image_center_x,
        "buffer_center_y": image_center_y,
        "ensemble_metadata": {
            "models_used": [m[0] for m in models],
            "strategy": ENSEMBLE_STRATEGY,
            "contributing_models": contributing_models,
            "prediction_time_s": prediction_time
        }
    }






def process_single_site(sample_id, lat, lon, models, radius_1_pixels, radius_2_pixels):
    """
    Process a single site: download images from dual sources, detect panels, merge results.
    
    Returns:
        JSON record for the site
    """
    print(f"\n{'='*60}")
    print(f"Processing Sample ID: {sample_id}")
    print(f"Coordinates: ({lat}, {lon})")
    
    if USE_DUAL_SOURCE:
        # Dual-source mode: Download from both ArcGIS and Mapbox
        print(f"⬇️  Downloading satellite images from dual sources...")
        
        arcgis_filename = f"{sample_id}_arcgis.png"
        mapbox_filename = f"{sample_id}_mapbox.png"
        arcgis_path = os.path.join(IMAGE_DIR, arcgis_filename)
        mapbox_path = os.path.join(IMAGE_DIR, mapbox_filename)
        
        download_results = download_images_dual_source(lat, lon, arcgis_path, mapbox_path)
        
        arcgis_success = download_results["arcgis"]["success"]
        mapbox_success = download_results["mapbox"]["success"]
        
        print(f"   {'✅' if arcgis_success else '❌'} ArcGIS: {download_results['arcgis']['message']}")
        print(f"   {'✅' if mapbox_success else '❌'} Mapbox: {download_results['mapbox']['message']}")
        
        # Track which sources succeeded
        successful_sources = []
        detection_results = []
        
        # Run detection on ArcGIS image if available
        if arcgis_success:
            print(f"🔍 Running detection on ArcGIS image...")
            arcgis_detection = detect_solar_panels(arcgis_path, models, radius_1_pixels, radius_2_pixels)
            arcgis_detection["source"] = "ArcGIS World Imagery"
            detection_results.append(arcgis_detection)
            successful_sources.append("ArcGIS")
            print(f"   {'✅' if arcgis_detection['has_solar'] else '❌'} ArcGIS Detection: {arcgis_detection['has_solar']}")
        
        # Run detection on Mapbox image if available
        if mapbox_success:
            print(f"🔍 Running detection on Mapbox image...")
            mapbox_detection = detect_solar_panels(mapbox_path, models, radius_1_pixels, radius_2_pixels)
            mapbox_detection["source"] = "Mapbox Satellite"
            detection_results.append(mapbox_detection)
            successful_sources.append("Mapbox")
            print(f"   {'✅' if mapbox_detection['has_solar'] else '❌'} Mapbox Detection: {mapbox_detection['has_solar']}")
        
        # Merge results based on strategy
        if not detection_results:
            # Both downloads failed
            print(f"   ❌ Both image sources failed")
            return {
                "sample_id": int(sample_id),
                "lat": float(lat),
                "lon": float(lon),
                "has_solar": False,
                "confidence": 0.0,
                "pv_area_sqm_est": 0.0,
                "capacity_kw": 0.0,
                "buffer_radius_sqft": BUFFER_ZONE_2_SQFT,
                "qc_status": "NOT_VERIFIABLE",
                "bbox_or_mask": "",
                "image_metadata": {
                    "sources_attempted": ["ArcGIS", "Mapbox"],
                   "sources_succeeded": [],
                    "capture_date": datetime.now().strftime("%Y-%m-%d"),
                    "error": "All image sources failed"
                }
            }
        
        # Merge detections based on strategy
        if DUAL_SOURCE_STRATEGY == "OR":
            # Positive if EITHER source detects solar
            has_solar_any = any(d["has_solar"] for d in detection_results)
            
            if has_solar_any:
                # Choose result with highest confidence
                final_detection = max(detection_results, key=lambda x: x["confidence"])
                has_solar = True
            else:
                # No detection from either source
                final_detection = detection_results[0]
                has_solar = False
        else:  # "AND" strategy
            # Positive only if BOTH sources detect solar
            has_solar = all(d["has_solar"] for d in detection_results) if len(detection_results) == 2 else False
            final_detection = max(detection_results, key=lambda x: x["confidence"]) if has_solar else detection_results[0]
        
        # Extract merged results
        confidence = final_detection["confidence"]
        pv_area_sqm = final_detection["pv_area_sqm_est"]
        capacity_kw = final_detection.get("capacity_kw", 0.0)
        buffer_radius = final_detection["buffer_radius_sqft"]
        bbox_or_mask = final_detection["bbox_or_mask"]
        ensemble_metadata = final_detection.get("ensemble_metadata", {})
        detection_source = final_detection.get("source", "Unknown")
        
        # Use the image from the successful detection for overlay
        if arcgis_success and (not detection_results or final_detection == detection_results[0]):
            primary_image_path = arcgis_path
        elif mapbox_success:
            primary_image_path = mapbox_path
        else:
            primary_image_path = arcgis_path
        
        print(f"   🔄 Merged Result ({DUAL_SOURCE_STRATEGY} strategy):")
        print(f"   {'✅' if has_solar else '❌'} Final Detection: {has_solar}")
        if has_solar:
            print(f"   📊 Best Source: {detection_source}")
            print(f"   📊 Confidence: {confidence:.2f}")
            print(f"   📏 Panel Area: {pv_area_sqm:.2f} sqm")
            print(f"   ⚡ Estimated Capacity: {capacity_kw:.2f} kW")
            print(f"   🎯 Buffer Zone: {buffer_radius} sq.ft")
        
    else:
        # Single-source mode (ArcGIS only)
        print(f"⬇️  Downloading satellite image...")
        image_filename =  f"{sample_id}_satellite.png"
        image_path = os.path.join(IMAGE_DIR, image_filename)
        
        success, message = download_image_arcgis(lat, lon, image_path)
        
        if not success:
            print(f"   ❌ Failed to download image: {message}")
            return {
                "sample_id": int(sample_id),
                "lat": float(lat),
                "lon": float(lon),
                "has_solar": False,
                "confidence": 0.0,
                "pv_area_sqm_est": 0.0,
                "capacity_kw": 0.0,
                "buffer_radius_sqft": BUFFER_ZONE_2_SQFT,
                "qc_status": "NOT_VERIFIABLE",
                "bbox_or_mask": "",
                "image_metadata": {
                    "source": "ArcGIS World Imagery",
                    "capture_date": datetime.now().strftime("%Y-%m-%d"),
                    "error": message
                }
            }
        
        print(f"   ✅ Image downloaded successfully")
        
        # Run detection
        print(f"🔍 Running solar panel detection...")
        detection_result = detect_solar_panels(image_path, models, radius_1_pixels, radius_2_pixels)
        
        has_solar = detection_result["has_solar"]
        confidence = detection_result["confidence"]
        pv_area_sqm = detection_result["pv_area_sqm_est"]
        capacity_kw = detection_result.get("capacity_kw", 0.0)
        buffer_radius = detection_result["buffer_radius_sqft"]
        bbox_or_mask = detection_result["bbox_or_mask"]
        ensemble_metadata = detection_result.get("ensemble_metadata", {})
        detection_source = "ArcGIS World Imagery"
        primary_image_path = image_path
        successful_sources = ["ArcGIS"]
        final_detection = detection_result
        
        print(f"   {'✅' if has_solar else '❌'} Solar Panels Detected: {has_solar}")
        if has_solar:
            print(f"   📊 Confidence: {confidence:.2f}")
            print(f"   📏 Panel Area: {pv_area_sqm:.2f} sqm")
            print(f"   ⚡ Estimated Capacity: {capacity_kw:.2f} kW")
            print(f"   🎯 Buffer Zone: {buffer_radius} sq.ft")
    
    # Determine QC status
    qc_status = determine_qc_status(has_solar, confidence, primary_image_path)
    print(f"   🔍 QC Status: {qc_status}")
    
    # Create audit overlay
    print(f"🎨 Creating audit overlay...")
    artifact_filename = f"{sample_id}_audit.png"
    artifact_path = os.path.join(ARTIFACTS_DIR, artifact_filename)
    
    overlay_success = create_audit_overlay(
        primary_image_path,
        final_detection["result"],
        final_detection["buffer_center_x"],
        final_detection["buffer_center_y"],
        radius_1_pixels,
        radius_2_pixels,
        has_solar,
        buffer_radius,
        confidence,
        pv_area_sqm,
        bbox_or_mask,
        artifact_path
    )
    
    if overlay_success:
        print(f"   ✅ Audit overlay created: {artifact_filename}")
    
    # Construct JSON record
    json_record = {
        "sample_id": int(sample_id),
        "lat": float(lat),
        "lon": float(lon),
        "has_solar": bool(has_solar),
        "confidence": float(confidence),
        "pv_area_sqm_est": float(pv_area_sqm),
        "capacity_kw": float(capacity_kw),
        "buffer_radius_sqft": int(buffer_radius),
        "qc_status": qc_status,
        "bbox_or_mask": str(bbox_or_mask) if bbox_or_mask else "",
        "image_metadata": {
            "sources": successful_sources if USE_DUAL_SOURCE else [detection_source],
            "best_source": detection_source,
            "dual_source_mode": USE_DUAL_SOURCE,
            "dual_source_strategy": DUAL_SOURCE_STRATEGY if USE_DUAL_SOURCE else None,
            "capture_date": datetime.now().strftime("%Y-%m-%d"),
            "coverage_km": KM_RADIUS * 2 if not USE_DUAL_SOURCE else None,
            "gsd_cm_per_pixel": GSD_CM_PER_PIXEL
        },
        "ensemble_metadata": ensemble_metadata
    }
    
    # Save individual JSON file
    json_filename = f"{sample_id}_result.json"
    json_path = os.path.join(JSON_DIR, json_filename)
    
    with open(json_path, "w") as f:
        json.dump(json_record, f, indent=2)
    
    print(f"   ✅ JSON output saved: {json_filename}")
    
    return json_record



# =========================
# MAIN PIPELINE
# =========================

def main():
    """Main pipeline execution."""
    print("\n" + "="*60)
    print("ROOFTOP SOLAR PV DETECTION PIPELINE (ENSEMBLE)")
    print("EcoInnovators Ideathon 2026 Challenge")
    print("="*60)
    
    # Load YOLO models
    print(f"\n📦 Loading YOLO models...")
    if USE_ENSEMBLE:
        print(f"   Mode: Ensemble ({len(MODEL_PATHS)} models)")
        print(f"   Strategy: {ENSEMBLE_STRATEGY}")
    else:
        print(f"   Mode: Single model")
    
    try:
        models = load_models(MODEL_PATHS if USE_ENSEMBLE else [MODEL_PATHS[0]])
        if not models:
            print(f"   ❌ No models loaded successfully")
            return
        print(f"   ✅ {len(models)} model(s) loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load models: {e}")
        return
    
    # Calculate buffer radii
    radius_1_pixels, radius_2_pixels = calculate_buffer_radii()
    print(f"\n📐 Buffer Zone Parameters:")
    print(f"   1200 sq.ft buffer radius: {radius_1_pixels:.2f} pixels")
    print(f"   2400 sq.ft buffer radius: {radius_2_pixels:.2f} pixels")
    
    # Read input Excel file
    print(f"\n📄 Reading input file: {INPUT_XLSX}")
    try:
        df = pd.read_excel(INPUT_XLSX)
        print(f"   ✅ Loaded {len(df)} sites")
        print(f"   Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"   ❌ Failed to read Excel file: {e}")
        return
    
    # Process each site
    all_results = []
    
    for idx, row in df.iterrows():
        sample_id = int(row["sample_id"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        
        result = process_single_site(sample_id, lat, lon, models, radius_1_pixels, radius_2_pixels)
        all_results.append(result)
    
    # Save combined results
    combined_json_path = os.path.join(OUTPUT_DIR, "all_results.json")
    with open(combined_json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"   Total sites processed: {len(all_results)}")
    print(f"   Sites with solar: {sum(1 for r in all_results if r['has_solar'])}")
    print(f"   Sites without solar: {sum(1 for r in all_results if not r['has_solar'])}")
    print(f"   Verifiable results: {sum(1 for r in all_results if r['qc_status'] == 'VERIFIABLE')}")
    print(f"\n📁 Outputs saved to:")
    print(f"   JSON files: {JSON_DIR}")
    print(f"   Audit overlays: {ARTIFACTS_DIR}")
    print(f"   Satellite images: {IMAGE_DIR}")
    print(f"   Combined results: {combined_json_path}")
    print("\n🎯 Pipeline completed successfully!\n")


if __name__ == "__main__":
    main()