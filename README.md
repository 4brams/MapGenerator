# Map Generator

A project that trying to generator map. Use DeepLabV3+, OpenEarthMap Dataset to to train and generate segmentation mask. Then, use Contour Polygonization, Polygon Approximation, Douglas–Peucker, Contour Smoothing, Skeletonization, Morphological Opening, Gaussian Blur and Morphological Closing to process segmentation mask and output the map.

<hr/>

## Quick Start

#### Step1

Run `pip install -r requirements.txt`.

#### Step2

Put your images in `input` folder.

#### Step3

Run `python main.py`.

#### Step4

Enter width and height of each of your images(meter).

#### Step5

Go to `outpot` folder and wait for result.
