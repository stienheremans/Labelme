import os
import uuid
from pathlib import Path
import numpy as np
import random
import rasterio as rio
import geopandas as gpd
from dotenv import load_dotenv
from rasterio.merge import merge
from rasterio.mask import mask
from skimage import io, exposure
from datetime import datetime
from PIL import Image
from tqdm import tqdm

import utils

name_tile_file = 'Tiles_sel.shp'

def normalize_tile(tile):
    print('--- Normalizing tile ---')
    print('Original tile shape:', tile.shape)
    print('Original tile min:', tile.min())
    print('Original tile max:', tile.max())
    print('Original tile dtype:', tile.dtype)

    nodata_val = 0
    valid_pixels = tile != nodata_val
    if not np.any(valid_pixels):
        raise ValueError("Tile contains only nodata")

    tile[tile == nodata_val] = 0

    tile = tile.swapaxes(0, 1).swapaxes(1, 2)
    tile = np.clip(tile, 0, None)

    if tile.shape[2] >= 3:
        tile = tile[..., [2, 1, 0]]
        try:
            p2, p98 = np.percentile(tile, (2, 98), axis=(0, 1))
        except Exception as e:
            print(f"Percentile calculation failed: {e}")
            p2, p98 = tile.min(axis=(0, 1)), tile.max(axis=(0, 1))
        p2 = np.where(p2 == p98, 0, p2)
        p98 = np.where(p98 == p2, 1, p98)
        tile = (255 * (tile - p2) / (p98 - p2)).clip(0, 255).astype(np.uint8)
        return tile

    elif tile.shape[2] == 1:
        tile = tile.squeeze(axis=2)
        p2, p98 = np.percentile(tile, (2, 98))
        if p2 == p98:
            p2, p98 = 0, 1
        tile = (255 * (tile - p2) / (p98 - p2)).clip(0, 255).astype(np.uint8)
        return tile
    else:
        raise ValueError(f"Unexpected tile shape: {tile.shape}")

def markTiles(folder, fLabels=0.1, nTiles=None, nLabelers=10):
    folder = Path(folder)
    load_dotenv()
    outputfolder = Path(os.environ['workdirectory'])
    tilefile = outputfolder / name_tile_file
    df = gpd.read_file(tilefile)

    df['Labeler'] = 0
    df['TileID'] = ""

    nlTiledf = df[df['Labeler'] == 0].copy()
    l = len(nlTiledf)
    if nTiles is None:
        nTiles = int(fLabels * l)
        nTiles = (nTiles // nLabelers) * nLabelers
    else:
        nTiles = nTiles * nLabelers

    if nTiles > l:
        print(f"⚠️ Not enough tiles available ({l}) for requested ({nTiles}). Using all available tiles instead.")
        nTiles = l

    ids = random.sample(range(l), nTiles)
    ids = np.array([nlTiledf.index[id] for id in ids])
    labelIDs = np.array_split(ids, nLabelers)

    LabelerID = 1
    for lIDs in labelIDs:
        for id in lIDs:
            df.loc[id, 'Labeler'] = LabelerID
            df.loc[id, 'TileID'] = str(uuid.uuid4())
        LabelerID += 1

    df.to_file(tilefile)
    backup_file = outputfolder / 'backup' / name_tile_file.replace('.shp', datetime.now().strftime('__%d%m%Y_%H%M.shp'))
    backup_file.parent.mkdir(exist_ok=True, parents=True)
    df.to_file(backup_file)

def extractTileFiles(folder, ext='jpg'):
    import shutil
    folder = Path(folder)
    load_dotenv()
    outputfolder = Path(os.environ['workdirectory'])
    tifs = [folder / str(os.environ['name_ortho_stitch'])]
    sources = [rio.open(f) for f in tifs]

    tilefile = outputfolder / name_tile_file
    print('Reading tile polygons from:', tilefile.resolve())

    df = gpd.read_file(tilefile)
    print('Number of tiles in shapefile:', len(df))
    lTiledf = df[df['Labeler'] >= 1].copy()
    print('Number of labeler-assigned tiles:', len(lTiledf))
    lTiledf.reset_index(inplace=True)

    for idx, tiledf in tqdm(lTiledf.iterrows(), desc='Extracting tiles', total=len(lTiledf), unit='tile'):
        poly = tiledf.geometry
        tile, out_trans = mask(sources[0], [poly], crop=True, nodata=0)

        nodata_val = 0
        valid_pixels = tile != nodata_val

        if not np.any(valid_pixels):
            print(f"Tile {tiledf.get('TileID', idx)} contains only nodata — skipping")
            continue

        tile[tile == nodata_val] = 0

        out_meta = sources[0].meta.copy()
        out_meta.update({"driver": "GTiff", "height": tile.shape[1], "width": tile.shape[2], "transform": out_trans, "dtype": "float32"})

        if out_meta['count'] > 3:
            out_meta.update({"count": 3})

        imgfile = outputfolder / 'Images' / f"Labeler{tiledf['Labeler']}" / f"{tiledf['TileID']}.{ext}"
        imgfile.parent.mkdir(exist_ok=True, parents=True)

        if ext in ['jpg', 'png']:
            try:
                tile = tile.swapaxes(0, 1).swapaxes(1, 2)  # (bands, rows, cols) -> (rows, cols, bands)
                if tile.ndim == 3 and tile.shape[2] >= 3:
                    tile = tile.astype(np.uint8)
                elif tile.ndim == 2:
                    tile = tile.astype(np.uint8)
                Image.fromarray(tile).save(imgfile)
            except Exception as e:
                print(f"Tile saving error for {imgfile}: {e}")
        else:
            with rio.open(imgfile, "w", **out_meta) as dest:
                dest.write(tile[:3].astype(np.float32))

if __name__ == '__main__':
    load_dotenv()
    folder = os.environ['orthofolder']

    # Empty existing Labeler folders before writing new tiles
    import shutil
    outputfolder = Path(os.environ['workdirectory'])
    labeler_folder = outputfolder / 'Images'
    if labeler_folder.exists():
        for subfolder in labeler_folder.glob('Labeler*'):
            if subfolder.is_dir():
                print(f"🧹 Clearing {subfolder}...")
                shutil.rmtree(subfolder)

    bExtractLabelTiles = True
    if bExtractLabelTiles:
        markTiles(folder, nTiles=int(os.environ['number_of_tiles_per_labeler']), nLabelers=int(os.environ['number_of_labelers']))
        extractTileFiles(folder, ext='jpg')
        extractTileFiles(folder, ext='tif')

    bExtractAllTiles = False
    if bExtractAllTiles:
        extractTileFiles(folder, ext='jpg')
        extractTileFiles(folder, ext='tif')
