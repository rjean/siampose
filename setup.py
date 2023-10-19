from setuptools import setup, find_packages


setup(
    name="siampose",
    version="0.1.1",
    packages=find_packages(include=["siampose", "siampose.*"] + ["lightning-bolts"]),
    python_requires=">=3.6",
    install_requires=["pytorch-lightning", "traitlets", "orion", "torchmetrics", "opencv-python","h5py", "thelper","albumentations", "frozendict", "tensorflow","natsort","deco"],
    #packages=["traitlets"],
    entry_points={
        "console_scripts": ["main=siampose.main:main"],
    },
)
