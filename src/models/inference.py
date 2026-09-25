import rasterio
import torch
import numpy as np
from src.models.unet_multiband import MultispectralUNet
from src.processing.polygonizer import mask_to_clean_polygons

def predict_and_vectorize(input_raster: str, output_geojson: str, in_channels: int = 4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultispectralUNet(in_channels=in_channels, num_classes=2).to(device)
    model.eval()

    with rasterio.open(input_raster) as src:
        data = src.read(list(range(1, in_channels + 1))).astype(np.float32)
        transform = src.transform
        crs = src.crs

    tensor = torch.from_numpy(data / 10000.0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        binary_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    gdf = mask_to_clean_polygons(binary_mask, transform, crs, min_area_m2=25.0, tolerance=0.7)
    gdf.to_file(output_geojson, driver="GeoJSON")
    print(f"Extracted {len(gdf)} vector features -> {output_geojson}")

if __name__ == "__main__":
    print("Ready for image vectorization execution.")
