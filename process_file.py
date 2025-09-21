import rasterio
from rasterio.shutil import copy
from rasterio.enums import Resampling
from rasterio.windows import Window
from tqdm import tqdm

input_tif = "data/World_Atlas_2015.tif"
output_cog = "data/lp_cog.tif"

tile_size = 4096

# Open the source TIFF
with rasterio.open(input_tif) as src:
    profile = src.profile.copy()
    
    # Update profile for a Cloud-Optimized GeoTIFF
    profile.update(
        driver='GTiff',
        compress='ZSTD',       # other options: 'DEFLATE', 'LZW'
        tiled=True,
        blockxsize=256,        # tile width in pixels
        blockysize=256,        # tile height in pixels
        BIGTIFF='IF_SAFER'     # auto-handle very large files
    )
    
    # Loop over the raster in windows
    for row_off in tqdm(range(0, src.height, tile_size)):
        for col_off in range(0, src.width, tile_size):
            window = Window(col_off, row_off,
                            min(tile_size, src.width - col_off),
                            min(tile_size, src.height - row_off))
            
            # Update profile for this tile
            tile_profile = profile.copy()
            tile_profile.update({
                'height': window.height,
                'width': window.width,
                'transform': rasterio.windows.transform(window, src.transform)
            })

            tile_filename = f"data/lp_{row_off}_{col_off}.tif"
            with rasterio.open(tile_filename, 'w', **tile_profile) as dst:
                dst.write(src.read(1, window=window), 1)

            print(f"Wrote {tile_filename}")
