# Docker Deployment Guide

## Quick Start

### Using Docker Compose (Recommended)

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

## Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose installed (included with Docker Desktop)
- Input Excel file in `./input/solar_test_sites.xlsx`

## Directory Structure

```
ECO_INNOV/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pipeline.py
├── best.pt
├── best (2).pt
├── input/
│   └── solar_test_sites.xlsx
└── output/              # Created automatically
    ├── images/
    ├── artifacts/
    └── json/
```

## Configuration

### Environment Variables

Set in `docker-compose.yml` or pass with `-e` flag:

```yaml
environment:
  - PYTHONUNBUFFERED=1
  - OPENCV_VIDEOIO_DEBUG=0
```

### Resource Limits

Adjust in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
```

## Volume Mounts

### Input Volume (Read-Only)
- **Host**: `./input`
- **Container**: `/app/input`
- **Contains**: Excel file with site coordinates

### Output Volume (Read-Write)
- **Host**: `./output`
- **Container**: `/app/output`
- **Contains**: Generated images, JSON results, audit overlays

## Troubleshooting

### Build Issues

**Problem**: Build fails with dependency errors
```bash
# Clear Docker cache and rebuild
docker-compose build --no-cache
```

**Problem**: Model files not found
```bash
# Ensure model files exist
ls -lh "best.pt" "best (2).pt"
```

### Runtime Issues

**Problem**: Permission denied on output directory
```bash
# Fix permissions
chmod 777 output/
```

**Problem**: Input file not found
```bash
# Check file exists
ls -lh input/solar_test_sites.xlsx
```

### View Container Logs

```bash
# Docker Compose
docker-compose logs -f

# Docker CLI
docker logs <container-id>
```

## Advanced Usage

### Interactive Shell

```bash
# Enter container shell
docker run -it --rm \
  -v "$(pwd)/input:/app/input:ro" \
  -v "$(pwd)/output:/app/output" \
  solar-detection:latest /bin/bash
```

### Custom Input File

```bash
# Use different input file
docker run --rm \
  -v "$(pwd)/custom_input:/app/input:ro" \
  -v "$(pwd)/output:/app/output" \
  solar-detection:latest
```

### Single Model Mode

Modify `pipeline.py` before building:
```python
USE_ENSEMBLE = False  # Use only first model
```

## Image Information

- **Base Image**: python:3.10-slim
- **Size**: ~2-3 GB
- **Models Included**: best.pt, best (2).pt
- **Python Version**: 3.10

## Performance

- **Ensemble Mode**: ~0.07-0.09s per site (2 models in parallel)
- **Memory Usage**: ~2-4 GB
- **CPU Usage**: Scales with available cores

## Security Notes

- Input volume mounted as read-only (`:ro`)
- No sensitive data in image
- Runs as default user (can be changed to non-root)

## Cleanup

```bash
# Remove container and volumes
docker-compose down -v

# Remove image
docker rmi solar-detection:latest

# Clean up all Docker resources
docker system prune -a
```
