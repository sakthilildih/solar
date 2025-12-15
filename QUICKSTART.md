# 🚀 Quick Start Guide - For Judges

**EcoInnovators Ideathon 2026 - Solar PV Detection System**

---

## ⚡ Quick Execution Options

### Option 1: Docker (Recommended) ⭐

**Prerequisites**: Docker installed

```bash
# 1. Navigate to project directory
cd solar-detection

# 2. Run with Docker Compose (one command!)
docker-compose up

# That's it! Results will be in ./output/
```

**What happens**:
- Automatically downloads satellite images for coordinates in `input/sites.xlsx`
- Runs dual YOLO-OBB model ensemble for detection
- Generates JSON results + visual audit overlays
- Outputs saved to `output/` folder

---

### Option 2: Local Python Execution

**Prerequisites**: Python 3.10+

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run pipeline
python pipeline.py

# Results in ./output/
```

---

## 📊 About Our Models

### Architecture
- **Type**: YOLOv8-OBB (Oriented Bounding Box Detection)
- **Models**: 2-model ensemble (`best.pt` + `best (2).pt`)
- **Size**: ~23 MB each
- **Input**: 512×512 satellite images
- **Output**: Rotated bounding boxes with confidence scores

### Ensemble Strategy
- **Parallel Inference**: Both models run simultaneously using threading
- **Selection Method**: Max-confidence strategy picks best detection
- **Performance**: ~0.07-0.17s per site (2x faster than sequential)

### Training Data
- **Sources**: Roboflow public datasets (Alfred Weber Institute, LSGI547, Piscinas Y Tenistable)
- **Images**: ~2,000-3,000 annotated rooftop solar panels
- **Format**: YOLO OBB rotated bounding boxes
- **Coverage**: Diverse geographic regions, roof types, imaging conditions

### Model Capabilities
✅ Detects solar panels at various orientations (rotated boxes)  
✅ Works across flat and sloped roofs  
✅ Handles buffer zone detection (1200 & 2400 sq.ft)  
✅ Estimates panel area in square meters  
✅ Provides confidence scores + quality control status  

---

## 📁 Input Format

**File**: `input/sites.xlsx`

| sample_id | latitude | longitude |
|-----------|----------|-----------|
| 1001      | 12.9716  | 77.5946   |
| 1002      | 28.6139  | 77.2090   |

---

## 📤 Output Format

### 1. JSON Results (`output/all_results.json`)
```json
{
  "sample_id": 2002,
  "lat": 12.9788,
  "lon": 77.6001,
  "has_solar": true,
  "confidence": 0.6048,
  "pv_area_sqm_est": 2.45,
  "capacity_kw": 0.43,
  "buffer_radius_sqft": 1200,
  "qc_status": "VERIFIABLE",
  "ensemble_metadata": {
    "models_used": ["best.pt", "best (2).pt"],
    "strategy": "max_confidence",
    "prediction_time_s": 0.117
  }
}
```

### 2. Visual Artifacts (`output/artifacts/`)
- Satellite image with overlay
- Blue circle: 1200 sq.ft buffer
- Green circle: 2400 sq.ft buffer
- Yellow boxes: Detected solar panels
- Confidence scores displayed

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Automated Image Fetch** | ArcGIS World Imagery API (no auth required) |
| **Dual Buffer Zones** | 1200 sq.ft (inner) + 2400 sq.ft (outer) |
| **Area Quantification** | Precise measurement using geometric overlap |
| **Capacity Estimation** | Kilowatt capacity (175 Wp/m² assumption) |
| **Quality Control** | VERIFIABLE / NOT_VERIFIABLE status |
| **Ensemble Learning** | 2 models for improved accuracy & robustness |
| **Audit-Ready** | JSON + visual overlays for transparency |

---

## 📈 Performance Metrics

- **Inference Speed**: 0.07-0.17s per site
- **Memory Usage**: 2-4 GB
- **Expected F1 Score**: ~0.80-0.85
- **Scaling**: Processes unlimited sites sequentially

---

## 🔍 Verification Workflow

```
Input (lat, lon) 
    ↓
Fetch Satellite Image (ArcGIS)
    ↓
Ensemble Model Inference (2x YOLO-OBB)
    ↓
Buffer Zone Detection (1200 → 2400 sq.ft)
    ↓
Area Quantification (intersection overlap)
    ↓
Output: JSON + Audit Overlay
```

---

## 📞 Questions?

See detailed documentation:
- **README.md** - Full project documentation
- **MODEL_CARD.md** - Model details, limitations, biases
- **TRAINING_LOGS.md** - Training metrics and reproducibility
- **README_DOCKER.md** - Advanced Docker usage

---

**Project by EcoInnovators | EcoInnovators Ideathon 2026**
