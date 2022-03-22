# Coco Images Conversion for Retraining Preparation

The coco-dataset is used to retrain the DeepLabV3 model. The per person images
and corresponding masks are pre-processed with the code in `gepetto` repository:
`third_party/SemanticSegmentation/utils/CoCo_DensePose` on
[gitlab](https://gitlab.g6labs.com/3dselfies/gepetto/-/tree/master/third_party/SemanticSegmentation/utils/CoCo_DensePose).

The images after the pre-processing are placed on `192.168.1.74` at location:
`/groups1/3DSelfie/jie/datasets/coco-densepose/`, where `train-bg-output` is used for model
retraining and `minival-bg-output` is used for testing.

Note, one can mount the `/groups1` to any server by:
```
sudo mount -t nfs 192.168.1.74:/groups1 $HOME/3dselfie
ls -trld $HOME/3dselfie/3DSelfie/jie/datasets/coco-densepose/*
```

## Resize Input Images
To retrain the DeepLabV3 model, all the images need to be resized to be the same
size. In this retraining, for example, all images are resized to Height x Width
as 640 x 480 pixels.

To prepare the images for retraining, one can do:
```
mkdir -p dataset
python image_conversion.py \
  -i $HOME/3dselfie/3DSelfie/jie/datasets/coco-densepose/train-bg-output/crop \
  -o dataset/Images \
  -t Image
```

## Resize and Process Corresponding Masks
The masks need to be converted into black (background) and white (human) images.
And then they should be converted to the same size as the images above.

This can be done by:
```
python image_conversion.py \
  -i $HOME/3dselfie/3DSelfie/jie/datasets/coco-densepose/train-bg-output/mask_bg \
  -o dataset/Masks \
  -t Mask
```
