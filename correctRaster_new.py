import os
from pathlib import Path

import numpy as np
import geopandas as gpd
from dotenv import load_dotenv
from tqdm import tqdm
from rasterio import open as rio_open
from shapely.geometry import box
from datetime import datetime

def correctTiles(folder):
    '''
    Create a new shapefile that only contains tiles overlapping the orthophoto.
    '''
    folder = Path(folder)
    load_dotenv()
    outputfolder = Path(os.environ['workdirectory'])
    tilefile = outputfolder / 'Tiles.shp'

    print(f"📄 Reading tilefile: {tilefile}")
    df = gpd.read_file(tilefile)
    print(f"📦 Loaded {len(df)} tiles from shapefile.")

    df['bEmpty'] = False

    tif_path = folder / str(os.environ['name_ortho_stitch'])
    print(f"🛰️ Reading orthophoto: {tif_path}")

    with rio_open(tif_path) as src:
        raster_bounds = src.bounds
        raster_crs = src.crs
        raster_polygon = box(raster_bounds.left, raster_bounds.bottom, raster_bounds.right, raster_bounds.top)

    # Reproject tiles if needed
    if df.crs != raster_crs:
        print("🔄 Reprojecting tile geometries to match raster CRS.")
        df = df.to_crs(raster_crs)

    print(f"🔍 Checking {len(df)} tiles for overlap with orthophoto bounds...")

    for idx, tiledf in tqdm(df.iterrows(), desc='Checking tile overlap', unit='tile', total=len(df)):
        poly = tiledf.geometry

        if not poly.intersects(raster_polygon):
            df.loc[idx, 'bEmpty'] = True
            print(f"❌ Tile {idx} does NOT overlap — marking as empty.")

    nonEmptyDf = df[df['bEmpty'] == False]
    n_empty = (df['bEmpty'] == True).sum()
    n_non_empty = (df['bEmpty'] == False).sum()

    corrected_file = outputfolder / 'Tiles_corrected.shp'
    corrected_file.parent.mkdir(exist_ok=True, parents=True)
    nonEmptyDf.to_file(corrected_file)

    print(f"✅ Found {n_non_empty} non-empty tiles (kept).")
    print(f"🗑️ Found {n_empty} empty tiles (removed).")
    print(f"✅ Corrected shapefile saved to: {corrected_file}")

    backup_file = outputfolder / 'backup' / ('Tiles_backup_' + datetime.now().strftime('%d%m%Y_%H%M') + '.shp')
    backup_file.parent.mkdir(exist_ok=True, parents=True)
    df.to_file(backup_file)
    print(f"📦 Original Tiles backed up to: {backup_file}")

if __name__ == '__main__':
    load_dotenv()
    correctTiles(os.environ['orthofolder'])
