
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
import skimage
from skimage import io, exposure
from datetime import datetime
from PIL import Image
import matplotlib as plt
from tqdm import tqdm

import utils


def extractTileFiles(folder, ext='jpg'):
    '''
    Extract the tiles that are marked for a labeler to RGB images for LabelMe
    :param folder: The parent folder where individual labeler folders will be made to store the images
    :param ext: option to save the image tiles as 'jpg', 'png' or 'tif' format (only 1 or 3 channels are allowed).
    :return: None
    '''

    folder = Path(folder)
    load_dotenv()
    outputfolder = Path(os.environ['workdirectory'])
    # get all tif files & open them
    #tifs = utils.getTifFiles(folder)
    tifs = [
        folder / str(os.environ['name_ortho_stitch'])
    ]
    sources = [rio.open(f) for f in tifs]

    # get tile file
    tilefile = outputfolder / name_tile_file
    df = gpd.read_file(tilefile)
    lTiledf = df[df['Labeler'] >= 1].copy()  # non-labelled
    lTiledf.reset_index(inplace=True)

    for idx, tiledf in tqdm(lTiledf.iterrows(), desc='Extracting tiles', total=len(lTiledf), unit='tile'):
        poly = tiledf.geometry
        # extract RGB tile
        tile, out_trans = merge(sources, bounds=poly.bounds, nodata=-9999, precision=50)
        # print(tile.shape)
        # save & reload file to convert the ndarray to rasterio dataset
        out_meta = sources[0].meta.copy()

        out_meta.update({"driver": "GTiff",
                         "height": tile.shape[1],
                         "width": tile.shape[2],
                         "transform": out_trans,
                         "dtype": "float32" # Ensure 32-bit dtype
                         })
        num_chan = out_meta['count']
        if num_chan > 3:
            out_meta.update({"count": 3})

        # 1) make image file & path if necessary
        imgfile = '{}.{}'.format(tiledf['TileID'], ext)
        imgfile = outputfolder / 'Images' / 'Labeler{}'.format(tiledf['Labeler']) / imgfile
        imgfile.parent.mkdir(exist_ok=True, parents=True)

        # 2) write tile to imgfile
        if ext == 'jpg':
            tile = tile.swapaxes(0, 1).swapaxes(1, 2)
            if np.max(tile) <= 1:
                tile = tile.clip(0,1) # For single band. Remove non-data values. For NDVI, use .clip(-1,1)
                image_min = tile.min(axis=(0, 1), keepdims=True)
                image_max = tile.max(axis=(0, 1), keepdims=True)
                tile = (255 * (tile - image_min) / (image_max - image_min)).astype(np.uint8)


            if tile.shape[2] >= 3:
                # tile = tile[..., :3] # RGB
                tile = tile[..., [2, 1, 0]]  # BGR to RGB
                #tile = tile[..., [4,1,0]] # false color image

                # Enhance contrast using Contrast Stretching
                #tile = exposure.rescale_intensity(tile, in_range="image")
                p2, p98 = np.percentile(tile, (2, 98))
                tile = exposure.rescale_intensity(tile, in_range=(p2, p98)) #in_range="image"
                io.imsave(imgfile, tile)
            elif tile.shape[2] == 1:
                tile = tile.squeeze(axis=2)  # Remove single-channel axis
                tile = Image.fromarray(tile, mode='L')  # Convert to 8-bit grayscale
                tile = np.array(tile, dtype=np.uint8)  # Ensure it's 8-bit
                #viridis = plt.cm.get_cmap('viridis', 12)
                #colormap = viridis(tile / 255.0)  # Normalize for colormap
                #colormap = (colormap[:,:,:3]*255.0).astype(np.uint8) # Remove alpha & scale
                io.imsave(imgfile, tile)
            else:
                print("Your tile has 2 channels.")

        elif ext == 'png':
            tile = tile.swapaxes(0, 1).swapaxes(1, 2)
            if np.max(tile) <= 1:
                tile = tile.clip(0, 1)  # For single band. Remove non-data values. For NDVI, use .clip(-1,1)
                image_min = tile.min(axis=(0, 1), keepdims=True)
                image_max = tile.max(axis=(0, 1), keepdims=True)
                tile = (255 * (tile - image_min) / (image_max - image_min)).astype(np.uint8)


            if tile.shape[2] >= 3:
                # tile = tile[..., :3] # RGB
                tile = tile[..., [2, 1, 0]]  # BGR to RGB
                #tile = tile[..., [4,1,0]] # false color image

                # Enhance contrast using Contrast Stretching
                #tile = exposure.rescale_intensity(tile, in_range="image")
                p2, p98 = np.percentile(tile, (2, 98))
                tile = exposure.rescale_intensity(tile, in_range=(p2, p98)) #in_range="image"
                io.imsave(imgfile, tile)
            elif tile.shape[2] == 1:
                tile = tile.squeeze(axis=2)  # Remove single-channel axis
                tile = Image.fromarray(tile, mode='L')  # Convert to 8-bit grayscale
                tile = np.array(tile, dtype=np.uint8)  # Ensure it's 8-bit
                #viridis = plt.cm.get_cmap('viridis', 12)
                #colormap = viridis(tile / 255.0)  # Normalize for colormap
                #colormap = (colormap[:, :, :3] * 255.0).astype(np.uint8)  # Remove alpha & scale
                io.imsave(imgfile, tile)
            else:
                print("Your tile has 2 channels.")
        else:
            with rio.open(imgfile, "w", **out_meta) as dest:
                num_channels = tile.shape[0]  # Get the number of channels in the tile

                if num_channels == 1:
                    tile = tile.astype(np.float32)  # Convert to 32-bit
                    dest.write(tile)  # Write single-channel tile

                elif num_channels >= 3:
                    selected_bands = [0, 1, 2]  # Choose any three bands

                    if max(selected_bands) < num_channels:  # Ensure selected bands exist
                        new_tile = tile[selected_bands].astype(np.float32)  # Convert to 32-bit
                        dest.write(new_tile)  # Write the selected bands
                    else:
                        print(f"Invalid band selection. Available bands: {num_channels}")

                else:
                    print("Unexpected number of channels in the tile")


def extractAllTiles(folder, ext='.jpg'):
    '''
    Extract all tiles to RGB images for LabelMe
    :param folder: the folder were all the images will be stored
    :param ext: option to save the image tiles as 'jpg', 'png' or 'tif' format (only 1 or 3 channels are allowed).
    :return: None
    '''
    folder = Path(folder)
    load_dotenv()
    outputfolder = Path(os.environ['workdirectory'])

    # get all tif files & open them
    tifs = [
        folder / str(os.environ['name_ortho_stitch'])
    ]

    sources = [rio.open(f) for f in tifs]

    # get tile file
    tilefile = os.path.join(outputfolder, name_tile_file)
    allTiledf = gpd.read_file(tilefile)

    for idx, tiledf in tqdm(allTiledf.iterrows(), desc='Extracting tiles', total=len(allTiledf), unit='tile'):
        poly = tiledf.geometry
        # extract RGB tile
        tile, out_trans = merge(sources, bounds=poly.bounds, nodata=-9999, precision=50)
        #print(tile.shape)

        # save & reload file to convert the ndarray to rasterio dataset
        out_meta = sources[0].meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": tile.shape[1],
                         "width": tile.shape[2],
                         "transform": out_trans,
                         "dtype": "float32" # Ensure 32-bit dtype
                         })
        num_chan = out_meta['count']
        if num_chan > 3:
            out_meta.update({"count": 3})

        # when no unique ID is present, generate one
        if tiledf['TileID'] is None:
            uid = str(uuid.uuid4())
            allTiledf.loc[idx, 'TileID'] = uid
        else:
            uid = tiledf['TileID']

        # 1) make image file & path if necessary
        imgfile = '{}.{}'.format(uid, ext)
        imgfile = outputfolder / 'Images' / 'All' / imgfile
        imgfile.parent.mkdir(exist_ok=True, parents=True)

        # 2) write tile to imgfile
        if ext == 'jpg':
            tile = tile.swapaxes(0, 1).swapaxes(1, 2)
            if np.max(tile) <= 1:
                tile = tile.clip(0, 1)  # For single band. Remove non-data values. For NDVI, use .clip(-1,1)
                image_min = tile.min(axis=(0, 1), keepdims=True)
                image_max = tile.max(axis=(0, 1), keepdims=True)
                tile = (255 * (tile - image_min) / (image_max - image_min)).astype(np.uint8)



            if tile.shape[2] >= 3:
                # tile = tile[..., :3] # RGB
                tile = tile[..., [2, 1, 0]]  # BGR to RGB
                #tile = tile[..., [4,1,0]] # false color image

                # Enhance contrast using Contrast Stretching
                #tile = exposure.rescale_intensity(tile, in_range="image")
                p2, p98 = np.percentile(tile, (2, 98))
                tile = exposure.rescale_intensity(tile, in_range=(p2, p98)) #in_range="image"
                io.imsave(imgfile, tile)
            elif tile.shape[2] == 1:
                tile = tile.squeeze(axis=2)  # Remove single-channel axis
                tile = Image.fromarray(tile, mode='L')  # Convert to 8-bit grayscale
                tile = np.array(tile, dtype=np.uint8)  # Ensure it's 8-bit

                #colormap = plt.cm.viridis(tile/255.0) # Normalize for colormap
                #colormap = (colormap[:,:,:3]*255.0).astype(np.uint8) # Remove alpha & scale
                io.imsave(imgfile, tile)
            else:
                print("Your tile has 2 channels.")

        elif ext == 'png':
            tile = tile.swapaxes(0, 1).swapaxes(1, 2)
            if np.max(tile) <= 1:
                tile = tile.clip(0, 1)  # Remove non-data values. For NDVI, use .clip(-1,1)
                image_min = tile.min(axis=(0, 1), keepdims=True)
                image_max = tile.max(axis=(0, 1), keepdims=True)
                tile = (255 * (tile - image_min) / (image_max - image_min)).astype(np.uint8)

            if tile.shape[2] >= 3:
                # tile = tile[..., :3] # RGB
                tile = tile[..., [2, 1, 0]]  # BGR to RGB
                #tile = tile[..., [4,1,0]] # false color image

                # Enhance contrast using Contrast Stretching
                #tile = exposure.rescale_intensity(tile, in_range="image")
                p2, p98 = np.percentile(tile, (2, 98))
                tile = exposure.rescale_intensity(tile, in_range=(p2, p98)) #in_range="image"
                io.imsave(imgfile, tile)
            elif tile.shape[2] == 1:
                tile = tile.squeeze(axis=2)  # Remove single-channel axis
                tile = Image.fromarray(tile, mode='L')  # Convert to 8-bit grayscale
                tile = np.array(tile, dtype=np.uint8)  # Ensure it's 8-bit
                #colormap = plt.cm.viridis(tile / 255.0)  # Normalize for colormap
                #colormap = (colormap[:, :, :3] * 255.0).astype(np.uint8)  # Remove alpha & scale
                io.imsave(imgfile, tile)
            else:
                print("Your tile has 2 channels.")
        else:
            with rio.open(imgfile, "w", **out_meta) as dest:
                num_channels = tile.shape[0]  # Get the number of channels in the tile

                if num_channels == 1:
                    tile = tile.astype(np.float32)  # Convert to 32-bit
                    dest.write(tile)  # Write single-channel tile

                elif num_channels >= 3:
                    selected_bands = [2, 1, 0]  # Choose any three bands

                    if max(selected_bands) < num_channels:  # Ensure selected bands exist
                        new_tile = tile[selected_bands].astype(np.float32)  # Convert to 32-bit
                        dest.write(new_tile)

                    else:
                        print(f"Invalid band selection. Available bands: {num_channels}")

                else:
                    print("Unexpected number of channels in the tile")

    # last step, save changes to UUID of the tiles
    allTiledf.to_file(tilefile)

    # and we make a backup
    tilefile = os.path.join(outputfolder, 'backup', name_tile_file)
    tilefile = tilefile.replace('.shp', datetime.now().strftime('__%d%m%Y_%H%M.shp'))
    allTiledf.to_file(tilefile)

def markTiles(folder, fLabels=0.1, nTiles=None, nLabelers=10):
    '''
    Make new column marked as 'Labeler' -> who it will label
    :param folder:
    :param fLabels: the fraction of the available tiles that will be kept for labelling
    :param nLabelers: number of labelers to join the project
    :return: None, the function updates the file name_tile_file with a TileID and a Labeler number
    '''
    folder = Path(folder)
    load_dotenv()
    outputfolder = Path(os.environ['workdirectory'])
    # output file
    outfile = outputfolder / name_tile_file
    bboxgdf = gpd.read_file(outfile)

    # get tile file
    tilefile = outputfolder / name_tile_file
    df = gpd.read_file(tilefile)
    if 'TileID' not in df.columns:
        df['TileID'] = ""
    if 'Labeler' not in df.columns:
        df['Labeler'] = 0
        uniqueLabelerIDs = [0]
    else:
        uniqueLabelerIDs = list(df['Labeler'].unique())

    nlTiledf = df[df['bTileUsed'] == False].copy()  # non-labelled
    #nlTiledf.reset_index(inplace=True)

    l = len(nlTiledf)
    # calculate how many tiles we need
    if nTiles is None:
        nTiles = int(fLabels * l)
        nTiles = (nTiles // nLabelers) * nLabelers
    else:
        nTiles = nTiles*nLabelers
    # randomly subsample
    ids = []
    while True:
        ri = random.randint(0, l)
        if ri not in ids:
            ids.append(ri)
        if len(ids) == nTiles:
            break

    # get the IDs from the original array,
    # which can then be used on the dataframe df
    ids = np.array([nlTiledf.index[id] for id in ids]) # ids = np.array([nlTiledf.index[id-1] for id in ids])

    # And split them over the xx labelers
    labelIDs = np.split(ids, nLabelers)
    for l in labelIDs:
        utils.log('Labeler tiles:', array=l)

    LabelerID = 1
    # tag the dataframe
    for lIDs in labelIDs:
        while LabelerID in uniqueLabelerIDs:
            LabelerID += 1

        uniqueLabelerIDs.append(LabelerID)

        for id in lIDs:
            df.loc[id, 'Labeler'] = LabelerID
            df.loc[id, 'TileID'] = str(uuid.uuid4())

    outfile = os.path.join(outputfolder, name_tile_file)
    df.to_file(outfile)

    # and we make a backup
    outfile = os.path.join(outputfolder, 'backup', name_tile_file)
    outfile = outfile.replace('.shp', datetime.now().strftime('__%d%m%Y_%H%M.shp'))
    df.to_file(outfile)

if __name__ == '__main__':
    load_dotenv()
    folder = os.environ['orthofolder']
    name_tile_file = 'Tiles_sel.shp'


    bExtractLabelTiles = True
    if bExtractLabelTiles:
        markTiles(folder, nTiles=int(os.environ['number_of_tiles_per_labeler']), nLabelers=int(os.environ['number_of_labelers']))
        extractTileFiles(folder, ext='jpg')

    bExtractAllTiles = False
    if bExtractAllTiles:
        extractAllTiles(folder, ext='tif')