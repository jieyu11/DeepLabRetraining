# Inference with DeepLabV3
## Retrained DeepLabV3
If the retraining of the model is intended but not yet done, please find the
code in `ModelRetraining/.` folder for details.

## Inference with Retrained DeepLabV3
Running the docker image built with name: `bgsegmentation_deeplab_retrain`.

```
#!/bin/bash

dockertag="3af2dc_jie"
dockerimage=bgsegmentation_deeplab_retrain:${dockertag}
imgext="png"
inputdir=/path/to/the/images/folder
outputdir=/path/to/the/outputs/folder
mkdir -p $outputdir
INPUT_DIR="/work/data/inputs"
OUTPUT_DIR="/work/data/outputs"

nvidia-docker run --rm \
    -v $inputdir:/work/data/inputs \
    -v $outputdir:/work/data/outputs \
    $dockerimage \
    /bin/bash -c "echo running docker; \
    python inference.py \
		-m weights_deeplabv3_retrain_coco_width480_height640.pt \
        -i ${INPUT_DIR} \
        -o ${OUTPUT_DIR} \
    "
```

The above shell script reads an input director for inference and
save the images to the output directory.

## Inference with Original DeepLabV3

Similar to above, just change the inference code inside the docker image
to generate the masks.

```
#!/bin/bash

dockertag="3af2dc_jie"
dockerimage=bgsegmentation_deeplab_retrain:${dockertag}
imgext="png"
inputdir=/path/to/the/images/folder
outputdir=/path/to/the/outputs/folder
mkdir -p $outputdir
INPUT_DIR="/work/data/inputs"
OUTPUT_DIR="/work/data/outputs"

nvidia-docker run --rm \
    -v $inputdir:/work/data/inputs \
    -v $outputdir:/work/data/outputs \
    $dockerimage \
    /bin/bash -c "echo running docker; \
    python inference_rawmodel.py \
        -i ${INPUT_DIR} \
        -o ${OUTPUT_DIR} \
    "
```
