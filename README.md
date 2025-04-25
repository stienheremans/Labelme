# Labeling orthomosaics semi-automatically in labelme

### Requirements
- Pycharm (community edition)
- Python 3.12
- QGis 3.40.4

### Install python modules
1. Open terminal in Pycharm. Make sure your (pip) environment is activated and that your are in the directory where your requirements.txt is located. 
2. Type following command in the terminal
~~~shell
pip install -r requirements.txt
~~~

:rocket: Ready to start :rocket:

### Processing steps
0. Modify the hyperparameters in the .env file. 
1. drawRaster.py: makes raster of tiles for DL model, select buffer to be large enough to cover full orthomosaic.
2. correctRaster.py: remove empty boxes (with no pixels).
3. In QGIS: Make a copy of your tile file and select the tiles that have a dead tree within the tile (data of Thomas).
4. extractTiles.py: crop the orthomosaic to the extent of every box.
5. Label the images using labelme. Type following command in the terminal:
~~~shell
labelme 
~~~
or reference directly to your folder and add your pre-defined labels:
~~~shell
labelme C:/Users/stien_heremans/VSCProjects/Labelme/Images/Labeler1 --labels labels.txt
~~~
or reference directly to your folder and add your pre-defined flags (for annotation or classification):
~~~shell
labelme ./Images/FC --flags flags.txt --nodata
~~~
6. Convert_json_to_polygons.py: make shape file with labels.
7. Merge the alive trees and dead trees, dissolve boundaries. Make sure the dead trees are on top of the alive trees (shape_labels_merging.model3 script in QGIS). 
