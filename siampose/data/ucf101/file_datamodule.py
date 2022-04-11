import torch
import glob
import tqdm
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
imagenet_mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]
import torchvision.transforms as T
from torchvision.transforms.transforms import RandomResizedCrop, Scale

@concurrent
def natsorted_p(data):
    return natsorted(data)

@synchronized
def natsorted_dict(dictionnary):
    for key in dictionnary:
        dictionnary[key] = natsorted_p(dictionnary[key])
    return dictionnary


class UCF101Dataset(torch.utils.data.Dataset):

    nbClasses = 101

    classMapping = {'ApplyEyeMakeup': 1, 'ApplyLipstick': 2, 'Archery': 3, 'BabyCrawling': 4, 'BalanceBeam': 5, 'BandMarching': 6, 'BaseballPitch': 7, 'Basketball': 8, 'BasketballDunk': 9, 'BenchPress': 10, 'Biking': 11, 'Billiards': 12, 'BlowDryHair': 13, 'BlowingCandles': 14, 'BodyWeightSquats': 15, 'Bowling': 16, 'BoxingPunchingBag': 17, 'BoxingSpeedBag': 18, 'BreastStroke': 19, 'BrushingTeeth': 20, 'CleanAndJerk': 21, 'CliffDiving': 22, 'CricketBowling': 23, 'CricketShot': 24, 'CuttingInKitchen': 25, 'Diving': 26, 'Drumming': 27, 'Fencing': 28, 'FieldHockeyPenalty': 29, 'FloorGymnastics': 30, 'FrisbeeCatch': 31, 'FrontCrawl': 32, 'GolfSwing': 33, 'Haircut': 34, 'Hammering': 35, 'HammerThrow': 36, 'HandstandPushups': 37, 'HandstandWalking': 38, 'HeadMassage': 39, 'HighJump': 40, 'HorseRace': 41, 'HorseRiding': 42, 'HulaHoop': 43, 'IceDancing': 44, 'JavelinThrow': 45, 'JugglingBalls': 46, 'JumpingJack': 47, 'JumpRope': 48, 'Kayaking': 49, 'Knitting': 50, 'LongJump': 51, 'Lunges': 52, 'MilitaryParade': 53, 'Mixing': 54, 'MoppingFloor': 55, 'Nunchucks': 56, 'ParallelBars': 57, 'PizzaTossing': 58, 'PlayingCello': 59, 'PlayingDaf': 60, 'PlayingDhol': 61, 'PlayingFlute': 62, 'PlayingGuitar': 63, 'PlayingPiano': 64, 'PlayingSitar': 65, 'PlayingTabla': 66, 'PlayingViolin': 67, 'PoleVault': 68, 'PommelHorse': 69, 'PullUps': 70, 'Punch': 71, 'PushUps': 72, 'Rafting': 73, 'RockClimbingIndoor': 74, 'RopeClimbing': 75, 'Rowing': 76, 'SalsaSpin': 77, 'ShavingBeard': 78, 'Shotput': 79, 'SkateBoarding': 80, 'Skiing': 81, 'Skijet': 82, 'SkyDiving': 83, 'SoccerJuggling': 84, 'SoccerPenalty': 85, 'StillRings': 86, 'SumoWrestling': 87, 'Surfing': 88, 'Swing': 89, 'TableTennisShot': 90, 'TaiChi': 91, 'TennisSwing': 92, 'ThrowDiscus': 93, 'TrampolineJumping': 94, 'Typing': 95, 'UnevenBars': 96, 'VolleyballSpiking': 97, 'WalkingWithDog': 98, 'WallPushups': 99, 'WritingOnBoard': 100, 'YoYo': 101, 'HandStandPushups': 37, 'HandStandWalking': 38}

    def __init__(self, root="datasets/ucf101_112x112", split="train", memory=False, single=False, transform=None, 
                 debug_subset_size=None, return_indices = False, pair="uniform", exclude=[],
                 horizontal_flip=True):
        self.root=root
        self.memory = memory
        self.pairing = pair #Pairing strategy: uniform, next
        print(f"Pairing mode: {pair}. Memory dataloader: {self.memory}")
        self.split = split
        self.transform = transform
        self.size = None
        self.single = single
        self.return_indices = return_indices
        
        self.horizontal_flip = horizontal_flip

        self.classIndicesByNames = UCF101Dataset.classMapping
        self.classNamesByIndices = {idx: name for name, idx in self.classIndicesByNames.items()}
        self.categories = sorted(self.classIndicesByNames.keys())

        for exluded in exclude:
            self.categories.remove(exluded)
            print(f"Excluding {exluded} from dataset.")
        self.classes = self.categories

        self.number_of_pictures = 0

        self.sequences_by_categories = {}
        self.seq_subset = []
        for category in self.categories:
            sequences = self._get_sequences(category, self.split)
            self.sequences_by_categories[category] = sequences
            self.seq_subset+=sequences

        self._load_samples(self.split)

        if debug_subset_size is not None:
            self.samples = random.sample(self.samples, debug_subset_size)     

    def _get_sequences(self, category, split="train", fold=1):
    
        if split == 'train':
            partition = 'train'
            splitIndices = [fold]
        elif split == 'valid':
            # NOTE: use the third split as validation set
            partition = 'test'
            splitIndices = [fold]
        #elif split == 'test':
        #    partition = 'test'
        #    splitIndices = [1, 2, 3]
        else:
            raise ValueError(f"Unsupported split: {split}")

        classId = self.classIndicesByNames[category]

        sequences = {}
        for splitId in splitIndices:
            seqDirectories = glob.glob(f"{self.root}/split-{splitId}/{partition}/class-{classId}/group-*/seq-*")
            for seqDirectory in seqDirectories:
                for file in glob.glob(f"{seqDirectory}/frame_*.jpg"):
                    sequence_id = os.path.dirname(file).replace(self.root, '')
                    if sequence_id in sequences:
                        sequences[sequence_id].append(file)
                    else:
                        sequences[sequence_id] = [file]

        sequences = natsorted_dict(sequences)
        return sequences

    def _load_samples(self, split="train"):
        samples = []
        for category in self.categories:
            self.number_of_pictures = 0
            for sequence in self.sequences_by_categories[category]:
                nbSamples = len(self.sequences_by_categories[category][sequence])
                if nbSamples > 5:
                    self.number_of_pictures += nbSamples
                    for file in self.sequences_by_categories[category][sequence]:
                        sequence_id = os.path.dirname(file).replace(self.root, '')
                        _, splitId, _, classId, groupId, seqId = sequence_id.split('/')
                        splitId = int(splitId.split('-')[1])
                        classId = int(classId.split('-')[1])
                        groupId = int(groupId.split('-')[1])
                        seqId = int(seqId.split('-')[1])
                        frameId = int(os.path.splitext(os.path.basename(file))[0].split('_')[1])

                        sample = {"category": self.classNamesByIndices[classId], "sequence": sequence, 
                                  "path": file, "split": split, "frame_id": frameId,
                                  "group_id": groupId, "seq_id": seqId, "split_id": splitId}
                        samples.append(sample)
                else:
                    print(f"Skipping {category}/{sequence} : Not enough samples (%d)!" % (nbSamples))
            print(f"Category {category} has {len(self.sequences_by_categories[category])} sequences, for a total of {self.number_of_pictures} pictures")
    
        print(f"Total of {len(samples)} samples")
        self.samples = samples

    def get_pair_of_filenames(self, sample, root):
        split = sample["split"]
        sequence = sample["sequence"]
        path = sample["path"]
        category = sample["category"]

        image_path1 = path
        for i in range(0,30):
            if self.pairing=="uniform":
                other_path = random.sample(self.sequences_by_categories[category][sequence], 1)[0]
            elif self.pairing=="next":
                current_index = self.sequences_by_categories[category][sequence].index(path)
                other_path = self.get_next_filepath(current_index, category, sequence)
            elif self.pairing=="previous":
                current_index = self.sequences_by_categories[category][sequence].index(path)
                other_path = self.get_previous_filepath(current_index, category, sequence)
            elif self.pairing=="next_and_previous":
                current_index = self.sequences_by_categories[category][sequence].index(path)
                if random.choice([True,False]):
                    other_path = self.get_next_filepath(current_index, category, sequence)
                else:
                    other_path = self.get_previous_filepath(current_index, category, sequence)
            elif self.pairing=="same":
                other_path=image_path1 #Basic SimSiam setup
            else:
                raise ValueError(f"Unsupported pairing scheme: {self.pairing}")

            image_path2 = other_path
            if image_path1 != image_path2 or self.pairing=="same":
                return image_path1, image_path2, category
        
        raise ValueError(f"Unable to find another different image for this batch. Please check if there is more than one sample in the sequence! {image_path1}")

    def get_next_filepath(self, current_index, category, sequence):
        if (current_index+1) < len(self.sequences_by_categories[category][sequence]):
            next_index=current_index+1
        else:
            next_index=current_index-1 
        return self.sequences_by_categories[category][sequence][next_index]
    
    def get_previous_filepath(self, current_index, category, sequence):
        if current_index==0:
            previous_index=1 #For the first picture, give the next one instead of the previous one.
        else:
            previous_index=current_index-1 
        return self.sequences_by_categories[category][sequence][previous_index]

    def __getitem__(self, idx):
        success = False
        filename1 = None
        filename2 = None
        
        while True:
            #Some images are not having the right dimensions. We will simply skip them, and try the next one.
            filename1, filename2, category = self.get_pair_of_filenames(self.samples[idx], self.root)
            
            if filename1==filename2 and self.pairing!="same":
                continue #Sometimes, randomly sampling will give back the same file twice.
            else:
                break
            
        image1 = Image.open(filename1)
        image2 = Image.open(filename2)

        if self.horizontal_flip and random.choice([True,False]) and not self.memory:
            image1 = ImageOps.mirror(image1)
            image2 = ImageOps.mirror(image2)
            
        frameId1 = int(os.path.splitext(os.path.basename(filename1))[0].split('_')[1])
        frameId2 = int(os.path.splitext(os.path.basename(filename2))[0].split('_')[1])
        
        sample = {}
        image1, image2 = self.transform(image1), self.transform(image2)
        sample["OBJ_CROPS"] = (image1, image2)
        sample["UID"] = self.samples[idx]["category"] + self.samples[idx]["sequence"] + '/frames-' + str(frameId1) + '-' + str(frameId2)
        sample["CAT_ID"] = self.categories.index(category)
        return sample
        

    def __len__(self):
        return len(self.samples)


UCF101_PATH = "datasets/ucf101/112x112/"

class UCF101FileDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, data_dir: str = UCF101_PATH, batch_size=512, image_size=112, num_workers=16, pairing="next", dryrun=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.issetup=False
        self.num_workers = num_workers
        self.pairing = pairing
        self.dryrun = dryrun
    
    def setup(self, stage=None):
        if not self.issetup:
            self.train_transform = self.get_ucf101_transform(self.image_size)
            self.eval_transform = self.get_ucf101_transform(self.image_size, evaluation=True)
            self.train_dataset = UCF101Dataset(self.data_dir, split="train", transform=self.train_transform, pair=self.pairing)
            self.val_dataset = UCF101Dataset(self.data_dir, split="valid", transform=self.eval_transform, pair=self.pairing)
            #self.test_dataset = UCF101Dataset(self.data_dir, split="test", transform=self.eval_transform, pair=self.pairing)
            self.train_sample_count = len(self.train_dataset)
            self.valid_sample_count = len(self.val_dataset)
            self.issetup=True
    
    def get_ucf101_transform(self, image_size, mean_std=imagenet_mean_std, evaluation=False):
        if not evaluation:
            p_blur = 0.5 if image_size > 32 else 0 
            transform_list = [
                T.RandomResizedCrop(image_size, scale=(0.6, 1.4)),
                T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
                T.ToTensor(),
                T.Normalize(*mean_std)]
            return T.Compose(transform_list)
        else:
            return  T.Compose([
                T.ToTensor(),
                T.Normalize(*mean_std)
            ])

    def train_dataloader(self, evaluation=False) -> DataLoader:
        self.train_dataset.transform=self.train_transform
        if evaluation:
            self.train_dataset.transform = self.eval_transform
            self.train_dataset.memory = True # Avoid horizontal flip for evaluation.
        if self.dryrun: # Just to quickly test the training loop. Trains on "test set", validation on valid set. 
            print("WARNING: DRY RUN. Not performing real training.")
            return self.test_dataloader()
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self, evaluation=False) -> DataLoader:
        self.val_dataset.memory = False
        if evaluation:
            self.val_dataset.memory = True # Avoid horizontal flip for evaluation.
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    

if __name__ == "__main__":
    print("Data Module for UCF101 'File-Based'")
    
    dm = UCF101FileDataModule(data_dir='/data/sbrodeur/UCF-101/frames_112x112' ,num_workers=1)
    dm.setup()
    for batch in tqdm.tqdm(dm.train_dataloader(), total=int(len(dm.train_dataset)/dm.batch_size)):
        pass

