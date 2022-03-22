# PyTorch DeepLab V3 Retraining

## Prerequisite
Before doing the retraining, a dataset needs to be prepared, where `background`
pixels has color black (`[0, 0, 0]`) and `human` white (`[255, 255, 255]`). To use
coco dataset, the images are prepared using code in `../CocoMaskConversion/`.

## Retrieve the Retraing Package
To retrain the DeepLabV3 model, there is already a retraining code available (with some modification).
To retrieve the code, do the following:
```
git clone https://github.com/msminhas93/DeepLabv3FineTuning.git
cd DeepLabv3FineTuning
git checkout bcdc3dfc79a5b75bc30c52b32315661c0a4da17e
cd ..
```

## Install dependencies
There is an `out-of-memory` error while running the code by building a docker image. Therefore,
in the following practice, we're creating a `conda` enviroment and running under that.

### Create Conda Environment
Do the following only once to create the conda environment.
```
export CONDA_ENV_NAME=deeplabretrain
echo $CONDA_ENV_NAME

conda create -n $CONDA_ENV_NAME python=3.6.9

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME
pip install -r DeepLabv3FineTuning/requirements.txt
```

### Activate Conda Environment
To activate the conda environment, which is needed for every new login, do the following:
```
conda activate deeplabretrain
```

## Usage of the Module

### General Usage
```
Usage: main.py [OPTIONS]

Options:
  --data-directory TEXT  Specify the data directory.  [required]
  --exp_directory TEXT   Specify the experiment directory.  [required]
  --epochs INTEGER       Specify the number of epochs you want to run the
                         experiment for. Default is 25.

  --batch-size INTEGER   Specify the batch size for the dataloader. Default is 4.
  --help                 Show this message and exit.
```

### DeepLabV3 Retraining
To run the code with the `train` part of the coco dataset, which are prepared in
`../CocoMaskConversion/dataset/Images` and `../CocoMaskConversion/dataset/Masks`.

The file `datahandler.py` under the coloned repository has some issue running on
our server, therefore, a fix is make and the updated file needs to be copied to 
the repository:
```
cp datahandler.py DeepLabv3FineTuning/.
```

Now go into the repository and do the following to retrain the model:
```
cd DeepLabv3FineTuning
mkdir -p models
python main.py --data-directory ../CocoMaskConversion/dataset --exp_directory models
```

The retrained model file `weights.py` should appear under `models/`.