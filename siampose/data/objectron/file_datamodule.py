import torch
import glob
import os
from PIL import Image, ImageOps
import random
import re
from natsort import natsorted
from deco import concurrent, synchronized
import numpy as np
import pytorch_lightning
import typing
from torch.utils.data.dataloader import DataLoader

imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
import torchvision.transforms as T
from torchvision.transforms.transforms import RandomResizedCrop, Resize


@concurrent
def natsorted_p(data):
    return natsorted(data)


@synchronized
def natsorted_dict(dictionnary):
    for key in dictionnary:
        dictionnary[key] = natsorted_p(dictionnary[key])
    return dictionnary


# https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/
def expand2square(pil_img, background_color=0):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class ObjectronDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root="datasets/objectron_96x96",
        split="train",
        memory=False,
        single=False,
        transform=None,
        debug_subset_size=None,
        return_indices=False,
        objectron_pair="uniform",
        objectron_exclude=[],
        enable_cache=False,
        horizontal_flip=True,
    ):
        self.root = root
        self.memory = memory
        self.pairing = objectron_pair  # Pairing strategy: uniform, next
        print(f"Pairing mode: {objectron_pair}. Memory dataloader: {self.memory}")
        self.split = split
        self.transform = transform
        self.size = None
        self.single = single
        self.return_indices = return_indices
        self.enable_cache = enable_cache

        if "OBJECTRON_CACHE" in os.environ:
            self.enable_cache = True
        self.horizontal_flip = horizontal_flip
        # splits = glob.glob(f"{root}/*/")
        # if len(splits)==0:
        #    raise ValueError(f"Could not find splits in {root}")
        # splits = [x.split("/")[-2] for x in splits]
        # split = splits[0]
        self.categories = glob.glob(f"{root}/{split}/*/")
        self.categories = [x.split("/")[-2] for x in self.categories]
        self.categories.sort()  # To have the same order as in the ImageFolder dataset.
        for exluded in objectron_exclude:
            self.categories.remove(exluded)
            print(f"Excluding {exluded} from dataset.")
        self.classes = self.categories

        # self.categories_list = []
        # for category in self.categories:
        #    self.categories_list.append(category)

        self.number_of_pictures = 0
        #
        # self.
        self.sequences_by_categories = {}
        self.seq_subset = []
        for category in self.categories:
            sequences = self._get_sequences(category, self.split)
            self.sequences_by_categories[category] = sequences
            self.seq_subset.append(sequences)

        self._load_samples(self.split)

        if debug_subset_size is not None:
            self.samples = random.sample(self.samples, debug_subset_size)
        # categories

    def _get_basenames(self, category, split="train"):
        files = glob.glob(f"{self.root}/{split}/{category}/*.jpg")
        basenames = [os.path.basename(x) for x in files]
        return basenames

    def _get_sequences(self, category, split="train"):
        basenames = self._get_basenames(category, split)
        sequences = {}
        for basename in basenames:
            sequence_id = basename.split(".")[
                0
            ]  # "_".join([basename.split("_")[-3],basename.split("_")[-2],basename.split("_")[-1]])
            if sequence_id in sequences:
                sequences[sequence_id].append(basename)
            else:
                sequences[sequence_id] = [basename]

        sequences = natsorted_dict(sequences)
        # sequences = natsorted_dict(sequences)
        # for sequence_id in sequences: #Sort sequences
        #    sequences[sequence_id] = natsorted(sequences[sequence_id])
        return sequences

    def _load_samples(self, split="train", debug=True):
        samples = []
        for category in self.categories:
            self.number_of_pictures = 0
            for sequence in self.sequences_by_categories[category]:
                if len(self.sequences_by_categories[category][sequence]) > 5:
                    self.number_of_pictures += len(
                        self.sequences_by_categories[category][sequence]
                    )
                    for basename in self.sequences_by_categories[category][sequence]:
                        # frame_id = basename.split(".")[-2].split("_")[-1]
                        m = re.search("batch-(\d+)_(\d+)_(\d+).(\d+)\.jpg", basename)
                        batch_number = int(m[1])
                        sequence_number = int(m[2])
                        object_id = int(m[3])
                        frame_id = int(m[4])
                        sample = {
                            "category": category,
                            "sequence": sequence,
                            "basename": basename,
                            "split": split,
                            "frame_id": frame_id,
                        }
                        samples.append(sample)
                else:
                    print(f"Skipping {category}/{sequence} : Not enough samples!")
            print(
                f"Category {category} has {len(self.sequences_by_categories[category])} sequences, for a total of {self.number_of_pictures} pictures"
            )

        print(f"Total of {len(samples)} samples")
        self.samples = samples

    def get_pair_of_filenames(self, sample, root):
        split = sample["split"]
        sequence = sample["sequence"]
        basename = sample["basename"]
        category = sample["category"]
        image_path1 = f"{root}/{split}/{category}/{basename}"
        for i in range(0, 30):
            if self.pairing == "uniform":
                other_basename = random.sample(
                    self.sequences_by_categories[category][sequence], 1
                )[0]
            elif self.pairing == "next":
                current_index = self.sequences_by_categories[category][sequence].index(
                    basename
                )
                other_basename = self.get_next_basename(
                    current_index, category, sequence
                )
            elif self.pairing == "previous":
                current_index = self.sequences_by_categories[category][sequence].index(
                    basename
                )
                other_basename = self.get_previous_basename(
                    current_index, category, sequence
                )
            elif self.pairing == "next_and_previous":
                current_index = self.sequences_by_categories[category][sequence].index(
                    basename
                )
                if random.choice([True, False]):
                    other_basename = self.get_next_basename(
                        current_index, category, sequence
                    )
                else:
                    other_basename = self.get_previous_basename(
                        current_index, category, sequence
                    )
            elif self.pairing == "same":
                other_basename = basename  # Basic SimSiam setup
            else:
                raise ValueError(f"Unsupported pairing scheme: {self.pairing}")
                # x=x+1
                # other_basename = self.sequences_by_categories[category][sequence]
            if other_basename != basename or self.pairing == "same":
                image_path2 = f"{root}/{split}/{category}/{other_basename}"
                return image_path1, image_path2, category

        raise ValueError(
            f"Unable to find another different image for this batch. Please check if there is more than one sample in the sequence! {image_path1}"
        )

    def get_next_basename(self, current_index, category, sequence):
        if (current_index + 1) < len(self.sequences_by_categories[category][sequence]):
            next_index = current_index + 1
        else:
            next_index = current_index - 1
        next_basename = self.sequences_by_categories[category][sequence][next_index]
        return next_basename

    def get_previous_basename(self, current_index, category, sequence):
        if current_index == 0:
            previous_index = 1  # For the first picture, give the next one instead of the previous one.
        else:
            previous_index = current_index - 1
        next_basename = self.sequences_by_categories[category][sequence][previous_index]
        return next_basename

    def get_sequence_uid(self, idx):
        return self.samples[idx][""]

    def __getitem__(self, idx):
        success = False
        filename1 = None
        filename2 = None
        for i in range(0, 5):
            # Some images are not having the right dimensions. We will simply skip them, and try the next one.
            filename1, filename2, category = self.get_pair_of_filenames(
                self.samples[idx + i], self.root
            )
            if filename1 == filename2 and self.pairing != "same":
                continue  # Sometimes, randomly sampling will give back the same file twice.

            image1 = Image.open(filename1)
            image2 = Image.open(filename2)
            image1 = expand2square(image1)
            image2 = expand2square(image2)

            if (
                self.horizontal_flip
                and random.choice([True, False])
                and not self.memory
            ):
                image1 = ImageOps.mirror(image1)
                image2 = ImageOps.mirror(image2)

            if image1.size != image2.size:
                print(
                    f"Images not of the same size: {filename1}, {filename2}, skipping!"
                )
                continue
            else:
                success = True
                if self.size == None:
                    self.size = image1.size
                if image1.size != self.size or image2.size != self.size:
                    print(
                        f"Images not of the same size as previous images: {filename1}, {filename2}, skipping!"
                    )
                    continue
                #                    raise ValueError(f"Images size not the same as previous ones: {filename1}, {filename2}!")
                break
        if not success:
            raise ValueError(
                f"Multiple images not having the right dimensions! {filename1}, {filename2}"
            )

        sample = {}
        # if self.transform:
        image1, image2 = self.transform(image1), self.transform(image2)
        sample["OBJ_CROPS"] = (image1, image2)
        uid = (
            self.samples[idx]["category"]
            + "-"
            + self.samples[idx]["sequence"]
            + "-"
            + str(self.samples[idx]["frame_id"])
        )
        sample["UID"] = uid
        sample["CAT_ID"] = self.categories.index(category)
        return sample

    def __len__(self):
        return len(self.samples)


OBJECTRON_PATH = "datasets/objectron/96x96/"


class ObjectronFileDataModule(pytorch_lightning.LightningDataModule):
    def __init__(
        self,
        data_dir: str = OBJECTRON_PATH,
        batch_size=512,
        image_size=96,
        num_workers=6,
        pairing="next",
        dryrun=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.issetup = False
        self.num_workers = num_workers
        self.pairing = pairing
        self.dryrun = dryrun

    def setup(self, stage=None):
        if not self.issetup:
            # train_transform = self.get_objectron_transform(self.image_size)
            # eval_transform = self.get_objectron_transform(self.image_size, evaluation=True)
            self.train_transform = self.get_objectron_transform(self.image_size)
            self.eval_transform = self.get_objectron_transform(
                self.image_size, evaluation=True
            )
            self.train_dataset = ObjectronDataset(
                OBJECTRON_PATH,
                split="train",
                transform=self.train_transform,
                objectron_pair=self.pairing,
            )
            self.val_dataset = ObjectronDataset(
                OBJECTRON_PATH,
                split="valid",
                transform=self.eval_transform,
                objectron_pair=self.pairing,
            )
            self.test_dataset = ObjectronDataset(
                OBJECTRON_PATH,
                split="test",
                transform=self.eval_transform,
                objectron_pair=self.pairing,
            )
            # self.train_dataset = ObjectronDataset(OBJECTRON_PATH,split="train", transform=self.train_transform)
            # self.train_eval_dataset = ObjectronDataset(OBJECTRON_PATH,split="train", transform=eval_transform)
            # self.val_dataset = ObjectronDataset(OBJECTRON_PATH, split="valid", transform=self.eval_transform)
            # self.test_dataset = ObjectronDataset(OBJECTRON_PATH, split="test", transform=self.eval_transform)
            self.train_sample_count = len(self.train_dataset)
            self.valid_sample_count = len(self.val_dataset)
            self.issetup = True
            # self.seq_subset = self.sequence_list

    def get_objectron_transform(
        self, image_size, mean_std=imagenet_mean_std, evaluation=False
    ):
        if not evaluation:
            # hflip = T.RandomHorizontalFlip()
            p_blur = 0.5 if image_size > 32 else 0
            transform_list = [
                T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                # hflip,
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply(
                    [
                        T.GaussianBlur(
                            kernel_size=image_size // 20 * 2 + 1, sigma=(0.1, 2.0)
                        )
                    ],
                    p=p_blur,
                ),
                T.ToTensor(),
                T.Normalize(*mean_std),
            ]
            return T.Compose(transform_list)
        else:
            return T.Compose([T.ToTensor(), T.Normalize(*mean_std)])

    def train_dataloader(self, evaluation=False) -> DataLoader:
        self.train_dataset.transform = self.train_transform
        if evaluation:
            self.train_dataset.transform = self.eval_transform
            self.train_dataset.memory = True  # Avoid horizontal flip for evaluation.
        if (
            self.dryrun
        ):  # Just to quickly test the training loop. Trains on "test set", validation on valid set.
            print("WARNING: DRY RUN. Not performing real training.")
            return self.test_dataloader()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self, evaluation=False) -> DataLoader:
        self.val_dataset.memory = False
        if evaluation:
            self.val_dataset.memory = True  # Avoid horizontal flip for evaluation.
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


from tqdm import tqdm

if __name__ == "__main__":
    print("Data Module for Objectron 'File-Based'")
    dm = ObjectronFileDataModule(num_workers=8)
    dm.setup()
    for batch in tqdm(
        dm.train_dataloader(), total=int(len(dm.train_dataset) / dm.batch_size)
    ):
        images, y = batch
        images1, images2, meta = images
        assert images1.shape == images2.shape
        assert images2.shape[1:4] == torch.Size([3, 96, 96])
        # print("Train loader ok")
