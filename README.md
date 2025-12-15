# 🌞 AI-Powered Rooftop Solar PV Detection System

> **EcoInnovators Ideathon 2026 Challenge Submission**  
> A governance-ready, auditable, and low-cost remote verification digital pipeline for detecting rooftop solar installations across India.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)

> **🚀 For Judges**: See [QUICKSTART.md](QUICKSTART.md) for quick execution steps and model overview!

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Challenge Context](#-challenge-context)
- [Technical Approach](#-technical-approach)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Input/Output Format](#-inputoutput-format)
- [Model Details](#-model-details)
- [Evaluation Criteria](#-evaluation-criteria)
- [Docker Deployment](#-docker-deployment)
- [License](#-license)
- [Citations](#-citations)

---

## 🎯 Overview

This project implements an automated rooftop solar PV detection system designed for the **PM Surya Ghar: Muft Bijli Yojana** scheme. The system verifies rooftop solar installations remotely using satellite imagery and AI, ensuring subsidies reach genuine beneficiaries while maintaining public trust.

### Key Capabilities
- ✅ **Binary Classification**: Determines if rooftop solar panels are present at given coordinates
- 📏 **Area Quantification**: Estimates total solar panel area in square meters
- 🎯 **Buffer Zone Detection**: Checks 1200 sq.ft and 2400 sq.ft radius zones
- 🔍 **Quality Control**: Provides verifiability status for each detection
- 📊 **Audit Artifacts**: Generates visual overlays with detections and buffer zones
- 🚀 **Ensemble Learning**: Uses multiple YOLO-OBB models for improved accuracy

---

## 🏛️ Challenge Context

### PM Surya Ghar: Muft Bijli Yojana
Launched by Prime Minister Narendra Modi on **February 15, 2024**, this government scheme aims to:
- Provide **free electricity** to households in India
- Light up **1 crore (10 million) households**
- Provide up to **300 units of free electricity** every month
- Investment of over **₹75,000 crores**

### Governance Challenge
The scheme requires verification of rooftop solar installations to ensure:
- Subsidies reach genuine beneficiaries
- Public trust remains high
- Auditable and transparent verification process
- Low-cost alternative to slow field inspections

### Our Solution
This system answers a simple question at any given coordinate (latitude, longitude):

> **"Has a rooftop solar system actually been installed here?"**

It works reliably across India's diverse roof types (sloped, flat) and imaging conditions, with emphasis on **accuracy, auditability, and generalization** across states.

---

## 🔬 Technical Approach

### Architecture Overview

```
┌─────────────────┐
│ Input: Excel    │
│ (lat, lon)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Satellite Image │
│ Fetcher         │
│ (ArcGIS API)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Ensemble Model  │
│ Inference       │
│ (2x YOLO-OBB)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Buffer Zone     │
│ Detection       │
│ (1200/2400 sqft)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Area            │
│ Quantification  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Output:         │
│ - JSON Results  │
│ - Audit Overlay │
└─────────────────┘
```

### Technology Stack
- **Deep Learning**: YOLOv8-OBB (Oriented Bounding Box) for rotated object detection
- **Ensemble Strategy**: Multi-model ensemble with max-confidence selection
- **Satellite Imagery**: ArcGIS World Imagery API (no authentication required)
- **Geometric Processing**: Shapely for buffer zone intersection calculations
- **Parallel Processing**: ThreadPoolExecutor for concurrent model inference
- **Containerization**: Docker for reproducible deployments

---

## ✨ Features

### 1. Automated Image Acquisition
- Fetches high-resolution satellite imagery from ArcGIS World Imagery
- Configurable coverage radius (default: 0.04 km)
- Image size: 512×512 pixels
- Ground Sample Distance (GSD): ~10.88 cm/pixel

### 2. Ensemble Model Inference
- **2 YOLO-OBB models** running in parallel
- Thread-safe concurrent prediction
- Ensemble strategies: max_confidence, average, voting
- Inference time: ~0.07-0.17 seconds per site

### 3. Buffer Zone Detection
- **Inner Buffer**: 1200 sq.ft circular zone
- **Outer Buffer**: 2400 sq.ft circular zone
- Geometric overlap calculation using Shapely
- Returns the buffer zone with the largest panel overlap

### 4. Area Quantification
- Precise measurement using oriented bounding boxes (OBB)
- Accounts for panel rotation angles
- Calculates intersection area with buffer zones
- Output in square meters (sqm)
- **Capacity Estimation**: Calculates kW capacity using 175 Wp/m² assumption

### 5. Quality Control Status
- **VERIFIABLE**: Clear evidence (present/not present)
- **NOT_VERIFIABLE**: Insufficient evidence (low resolution, occlusion, stale imagery)

### 6. Audit Artifacts
- Visual overlays with:
  - Blue circle: 1200 sq.ft buffer zone
  - Green circle: 2400 sq.ft buffer zone
  - Yellow boxes: Best matching solar panel
  - Red boxes: Other detected panels
  - Confidence scores and metadata

---

## 🛠️ Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda package manager
- (Optional) Docker for containerized deployment

### Option 1: Local Installation

#### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd solar-detection
```

#### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n solar-detection python=3.10
conda activate solar-detection
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Option 2: Docker Installation
See [Docker Deployment](#-docker-deployment) section below.

---

## 🚀 Usage

### Command Line Execution

#### Basic Usage
```bash
python pipeline.py
```

The script expects:
- **Input**: Excel file at `input/sites.xlsx` with columns:
  - `sample_id`: Unique identifier
  - `latitude`: Latitude coordinate (WGS84)
  - `longitude`: Longitude coordinate (WGS84)

#### Input File Format
```
sample_id | latitude | longitude
----------|----------|----------
1001      | 12.9716  | 77.5946
1002      | 28.6139  | 77.2090
1003      | 19.0760  | 72.8777
```

### Configuration

Edit `pipeline.py` to customize:

```python
# Model Configuration
MODEL_PATHS = ["best.pt", "best (2).pt"]  # Model files
USE_ENSEMBLE = True                        # Enable/disable ensemble
ENSEMBLE_STRATEGY = "max_confidence"       # Ensemble strategy

# Buffer Zone Configuration
BUFFER_ZONE_1_SQFT = 1200  # Inner buffer (sq.ft)
BUFFER_ZONE_2_SQFT = 2400  # Outer buffer (sq.ft)

# Image Parameters
IMG_WIDTH = 512             # Image width (pixels)
IMG_HEIGHT = 512            # Image height (pixels)
KM_RADIUS = 0.04           # Coverage radius (km)
GSD_CM_PER_PIXEL = 10.88   # Ground sample distance

# Detection Threshold
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for detections
```

### Output Structure

After execution, outputs are saved to `output/`:

```
output/
├── all_results.json          # Combined results for all sites
├── images/
│   ├── 1001_satellite.png    # Raw satellite images
│   ├── 1002_satellite.png
│   └── ...
├── artifacts/
│   ├── 1001_audit.png        # Audit overlays with detections
│   ├── 1002_audit.png
│   └── ...
└── json/
    ├── 1001_result.json      # Individual JSON results
    ├── 1002_result.json
    └── ...
```

---

## 📁 Project Structure

```
solar-detection/
│
├── pipeline.py                 # Main detection pipeline
├── requirements.txt            # Python dependencies
├── best.pt                     # YOLO-OBB model #1 (23.39 MB)
├── best (2).pt                 # YOLO-OBB model #2 (23.40 MB)
│
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Docker Compose configuration
├── .dockerignore               # Docker build exclusions
├── .gitignore                  # Git exclusions
│
├── README.md                   # This file
├── README_DOCKER.md            # Docker deployment guide
│
├── input/                      # Input data directory
│   ├── sites.xlsx              # Production input file
│   └── solar_test_sites.xlsx   # Test input file
│
└── output/                     # Output directory (generated)
    ├── all_results.json        # Combined JSON results
    ├── images/                 # Satellite images
    ├── artifacts/              # Audit overlays
    └── json/                   # Individual JSON files
```

---

## 📊 Input/Output Format

### Input Format (Excel)
The input Excel file must contain the following columns:

| Column Name | Type    | Description                          |
|-------------|---------|--------------------------------------|
| sample_id   | Integer | Unique identifier for the site       |
| latitude    | Float   | Latitude in WGS84 decimal degrees    |
| longitude   | Float   | Longitude in WGS84 decimal degrees   |

### Output Format (JSON)

Each site generates a JSON record with the following structure:

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
  "bbox_or_mask": "[316.41, 270.51, 23.53, 8.79, 0.33]",
  "image_metadata": {
    "source": "ArcGIS World Imagery",
    "capture_date": "2025-12-15",
    "coverage_km": 0.08,
    "gsd_cm_per_pixel": 10.88
  },
  "ensemble_metadata": {
    "models_used": ["best.pt", "best (2).pt"],
    "strategy": "max_confidence",
    "contributing_models": ["best.pt"],
    "prediction_time_s": 0.117
  }
}
```

### Field Descriptions

| Field                | Type    | Description                                                  |
|----------------------|---------|--------------------------------------------------------------|
| sample_id            | int     | Unique site identifier                                       |
| lat                  | float   | Latitude coordinate                                          |
| lon                  | float   | Longitude coordinate                                         |
| has_solar            | bool    | Whether solar panels detected in buffer zones                |
| confidence           | float   | Confidence score (0.0-1.0) of best detection                 |
| pv_area_sqm_est      | float   | Estimated solar panel area in square meters                  |
| capacity_kw          | float   | Estimated capacity in kilowatts (175 Wp/m² assumption)       |
| buffer_radius_sqft   | int     | Buffer zone where panel detected (1200 or 2400)              |
| qc_status            | string  | Quality control status (VERIFIABLE/NOT_VERIFIABLE)           |
| bbox_or_mask         | string  | Bounding box: [cx, cy, width, height, rotation_deg]          |
| image_metadata       | object  | Image source, date, coverage, GSD information                |
| ensemble_metadata    | object  | Models used, strategy, contributing models, prediction time  |

---

## 🤖 Model Details

### Architecture
- **Framework**: Ultralytics YOLOv8-OBB
- **Model Type**: Oriented Bounding Box (OBB) detection
- **Input Size**: 512×512 pixels (configurable)
- **Output**: Rotated bounding boxes with confidence scores

### Training Data Sources

The models were trained using datasets from the following sources:

1. **Alfred Weber Institute of Economics** (Roboflow)
   - Solar panel detection dataset
   - Diverse geographical coverage

2. **LSGI547 Project** (Roboflow)
   - Rooftop solar panel annotations
   - High-quality labeled data

3. **Piscinas Y Tenistable** (Roboflow)
   - Additional training data
   - Various roof types and conditions

> **Note**: All datasets are publicly available under permissive licenses with proper attribution.

### Model Performance

#### Ensemble Configuration
- **Number of Models**: 2
- **Ensemble Strategy**: Max-confidence selection
- **Parallel Inference**: Thread-based concurrency
- **Average Prediction Time**: 0.07-0.17 seconds per site

#### Detection Parameters
- **Confidence Threshold**: 0.3 (configurable)
- **IoU Threshold**: Default YOLOv8 settings
- **NMS**: Non-Maximum Suppression enabled

### Model Files
- `best.pt`: Primary YOLO-OBB model (23.39 MB)
- `best (2).pt`: Secondary YOLO-OBB model (23.40 MB)

### Known Limitations & Mitigation

| Limitation                        | Mitigation Strategy                              |
|-----------------------------------|--------------------------------------------------|
| Urban vs rural performance gap    | Ensemble models with diverse training data       |
| Occlusion by trees/tanks          | QC status flagging (NOT_VERIFIABLE)              |
| Varying image quality             | Minimum resolution checks                        |
| Geocoding jitter                  | Large buffer zones (1200/2400 sq.ft)             |
| Seasonal variations               | Latest available imagery from ArcGIS             |

### Retraining Guidance

To retrain or fine-tune the models:

1. **Prepare Dataset**: Annotate rooftop solar panels using tools like Roboflow
2. **Export Format**: YOLO OBB format (text files with rotated boxes)
3. **Training Script**: Use Ultralytics YOLO training pipeline
   ```bash
   yolo obb train data=solar.yaml model=yolov8n-obb.pt epochs=100 imgsz=512
   ```
4. **Validation**: Test on held-out regions/states
5. **Deployment**: Replace `best.pt` or add as additional model to ensemble

---

## 📈 Evaluation Criteria

The system is evaluated across four key dimensions as per hackathon guidelines:

### 1. Detection Accuracy (40%)
- **Metric**: F1 Score on `has_solar` classification
- **Target**: High precision and recall across diverse scenarios
- **Validation**: Tested on multiple Indian cities and roof types

### 2. Quantification Quality (20%)
- **Metric**: RMSE (Root Mean Square Error) for PV area estimation
- **Unit**: Square meters (sqm)
- **Approach**: Geometric overlap calculation with buffer zones

### 3. Generalization & Robustness (20%)
- Performance across diverse states and roof types
- Handling of look-alikes (water tanks, skylights, air conditioners)
- Resilience to poor imagery conditions
- Multi-model ensemble for improved robustness

### 4. Code Quality, Documentation, Usability (20%)
- ✅ Clean, well-documented code with type hints
- ✅ Comprehensive README with setup instructions
- ✅ Docker support for reproducible deployment
- ✅ Detailed model card with limitations and bias documentation
- ✅ Audit-friendly outputs (JSON + visual artifacts)

---

## 🐳 Docker Deployment

### Quick Start with Docker Compose

```bash
# Build and run
docker-compose up

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop container
docker-compose down
```

### Using Docker CLI

```bash
# Build the image
docker build -t solar-detection:latest .

# Run the container
docker run --rm \
  -v "$(pwd)/input:/app/input:ro" \
  -v "$(pwd)/output:/app/output" \
  solar-detection:latest
```

### Docker Configuration

**Resource Limits** (adjustable in `docker-compose.yml`):
- **CPU**: 4 cores (limit), 2 cores (reservation)
- **Memory**: 8 GB (limit), 4 GB (reservation)

**Volume Mounts**:
- `./input:/app/input:ro` (read-only input data)
- `./output:/app/output` (read-write output data)

For detailed Docker instructions, see [README_DOCKER.md](README_DOCKER.md).

---

## 📜 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 EcoInnovators

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📚 Citations

### Datasets

1. **Alfred Weber Institute of Economics**  
   *Solar Panel Detection Dataset*  
   Available at: Roboflow Universe  
   License: Public Domain / CC BY 4.0

2. **LSGI547 Project**  
   *Rooftop Solar Panel Dataset*  
   Available at: Roboflow Universe  
   License: Public Domain / CC BY 4.0

3. **Piscinas Y Tenistable**  
   *Solar Panel Detection Dataset*  
   Available at: Roboflow Universe  
   License: Public Domain / CC BY 4.0

### Software & APIs

- **Ultralytics YOLOv8**: Glenn Jocher et al. (2023)  
  [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

- **ArcGIS World Imagery**: Esri  
  [https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer](https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer)

- **Shapely**: Sean Gillies et al.  
  [https://github.com/shapely/shapely](https://github.com/shapely/shapely)

---

## 🤝 Contributing

This project was developed for the **EcoInnovators Ideathon 2026 Challenge**. For questions or contributions, please refer to the hackathon guidelines.

### Development Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd solar-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python pipeline.py
```

---

## 📞 Support

For issues, questions, or feedback related to this project:
- Open an issue on the GitHub repository
- Contact the development team via hackathon communication channels

---

## 🙏 Acknowledgments

- **PM Surya Ghar: Muft Bijli Yojana** for the challenge and governance initiative
- **EcoInnovators Ideathon 2026** organizing committee
- **Open-source community** for datasets, models, and tools
- **Roboflow** for dataset hosting and annotation tools

---

<div align="center">

**Built with ❤️ for a sustainable future**

*Empowering governance through AI-powered verification*

</div>
