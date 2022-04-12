# Jupyter Notebooks
This folder contains Jupyter Notebooks used for visualisation and for data pre-processing.

## Development notebooks.
The ["dev" subfolder](dev) contains notebooks that were used during development, and might a good starting point to design automated unit test. Most of the development code was moved to python modules once validate in the notebooks.

## Pose estimation.
The [Zero Shot Pose Estimation Notebook](zero_shot_pose_vis.ipynb) can be used to do pose estimation. It was last ran to generate failure cases in the supplementary material of the paper.

## Qualitative evaluation notebooks.
There are two qualitative evaluation notebooks. The [Objectron Result Analysis Notebook](objectron_result_analysis.ipynb) one is to be used with the file-based dataloader and is less up to date and was use for prototyping, while the [Objectron Result Analysis Notebook - HDF5](objectron_result_analysis_hdf5.ipynb) is more up to date, but requires the larger "HDF5" Objectron dataset.
