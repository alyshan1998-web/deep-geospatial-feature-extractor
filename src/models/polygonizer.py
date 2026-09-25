import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

def mask_to_clean_polygons(
    mask: np.ndarray,
    transform: rasterio.transform.Affine,
    crs: str,
    min_area_m2: float = 30.0,
    tolerance: float = 0.5
) -> gpd.GeoDataFrame:
    """Converts raster segmentation binary arrays into clean, topologically valid vector geometries."""
    geometries = []
    for geom_dict, value in shapes(mask.astype(np.int16), mask=(mask == 1), transform=transform):
        poly = shape(geom_dict)
        if poly.is_valid and poly.area >= min_area_m2:
            simplified = poly.simplify(tolerance=tolerance, preserve_topology=True)
            geometries.append(simplified)
            
    return gpd.GeoDataFrame(geometry=geometries, crs=crs)
