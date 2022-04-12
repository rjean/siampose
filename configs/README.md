# Example configurations
This folder contains configurations that were used for experiments on a single 2080 Ti GPU. 

## Pre-train configurations
All pre-training configurations for 224x224 HDF5 Objecton in the folder have the "config-pretrain-8gb" prefix. They can run on a single 1070 Ti 8Gb GPU, but most of the experiments were performed on a 2080 Ti card (11gb). As most experiments rely on heavy data augmentations, a 8 core / 16 thread CPU such as a Intel 9900k or better is recommended. With such a configuration, it is possible to get to 45 minutes / epoch, and do a full 25 epoch training in less than 24 hours.

## Keypoint Regression
Direct keypoint regression is still experimental at this stage. Configurations are provided as-is for reference.

## Objection detection configurations.
Some experimental configurations are also available for object detection with Detectron 2. They might be used for further experiments on transfer learning.
