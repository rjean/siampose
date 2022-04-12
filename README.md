# SiamPose
Self-Supervised Learning of Pose-Informed Latents

## Installation
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

# Data
## Objectron Data
Two dataloaders are provided for Objectron, one which is "file-based" and is used for prototyping on machines with fast SSD, while the other one is "hdf5" based for the formal experiments.

### HDF5 preprocessing
For the HDF5 dataloader expects a pre-processed HDF5 file, which contains the full frame images and annotations in the same package.

TODO: Add HDF5 generation notebook.

### File-based preprocessing
For the file-based dataloader, bounding box crops are taken for each object instance in the Objectron raw video sequences. The "raw" objectron annotations are used directly, without further processing. This dataloader allow reducing the orignal 2Tb dataset to a more tractable 2Gb dataset for quick experiment cycles. 

## UCF-101 Data
TODO: Add UCF-101 notebooks.

# Training and evaluation.

## Pre-training a model.
Assuming you have the prepared HDF5 data in the /home/raphael/datasets/objectron folder, you can start pre-training with the following command:
``` 
python siampose/main.py --data=~/datasets/objectron --output=output --config=configsconfig-pretrain-8gb.yaml
```
During pre-training, the accuracy on category prediction is used as a proxy for the model quality.

## Evaluation on pose estimation.
To evaluate on the zero-shot pose estimation task, you must first generate the embeddings using the main program.
```
python siampose/main.py    --data=~/datasets/objectron \
                                --output=output/pretrain_224 \
                                --config=examples/local/config-pretrain-8gb.yaml
                                --embeddings
                                --embeddings-ckpt=output/pretrain_224/last_model.ckpt
```
This will generate embeddings for all images in the training and validation set. Care must be taken to use the same split as in training, or else you will get leaky results.

Once the embeddings are generated, the evaluation script can be launched.
```
python siampose/zero_shot_pose.py output/pretrain_224 --subset_size=5000 --cpu
```
3D IoU @ 50% precision will be reported for each individual category in objectron.

## Qualitative evaluation notebooks.
The qualitative evaluation notebooks can be found in the "notebooks" folder.

