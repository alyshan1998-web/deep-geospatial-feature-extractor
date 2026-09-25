# Deep Geospatial Feature Extractor: Multispectral Segmentation to GIS Polygons

Deep learning pipeline converting multispectral satellite imagery directly into GIS-compatible vector layers (GeoPackage, GeoJSON) using PyTorch semantic segmentation and topological vectorization.

## Architecture Highlights
- **Multi-Band U-Net:** Handles raw 4-band or multi-spectral sensor inputs directly without down-sampling to RGB.
- **Topological Polygonization:** Runs raster-to-shape vectorization and applies Douglas-Peucker simplification to eliminate pixel staircase artifacts.

## Quickstart
```bash
pip install -r requirements.txt
python -m src.inference
