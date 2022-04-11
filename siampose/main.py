#!/usr/bin/env python

import argparse
import logging
import os
import shutil
import sys
from torch.nn.modules.linear import Identity
from traitlets.traitlets import default
import yaml

import mlflow
from yaml import load
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger

from siampose.train import train, load_mlflow, STAT_FILE_NAME
from siampose.utils.hp_utils import check_and_log_hp
from siampose.models.model_loader import load_model
from siampose.utils.file_utils import rsync_folder
from siampose.utils.logging_utils import LoggerWriter, log_exp_details
from siampose.utils.reproducibility_utils import set_seed

from siampose.models.simsiam import save_mosaic #TODO: Place in some "utils" file.

import siampose.data.objectron.hdf5_parser
import siampose.data.objectron.file_datamodule


logger = logging.getLogger(__name__)


def main():
    """Main entry point of the program.

    Note:
        This main.py file is meant to be called using the cli,
        see the `examples/local/run.sh` file to see how to use it.

    """
    parser = argparse.ArgumentParser()
    # __TODO__ check you need all the following CLI parameters
    parser.add_argument('--config',
                        help='config file with generic hyper-parameters,  such as optimizer, '
                             'batch_size, ... -  in yaml format')
    parser.add_argument('--data', help='path to data', required=True)
    parser.add_argument('--data-module', default="hdf5", help="Data module to use. file or hdf5")
    parser.add_argument('--tmp-folder',
                        help='will use this folder as working folder - it will copy the input data '
                             'here, generate results here, and then copy them back to the output '
                             'folder')
    parser.add_argument('--output', help='path to outputs - will store files here', required=True)
    parser.add_argument('--disable-progressbar', action='store_true',
                        help='will disable the progressbar while going over the mini-batch')
    parser.add_argument('--start-from-scratch', action='store_true',
                        help='will not load any existing saved model - even if present')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--embeddings-device",type=str,default="cuda",help="Which device to use for embeddings generation.")
    parser.add_argument('--embeddings', action='store_true',help="Skip training and generate embeddings for evaluation.")
    parser.add_argument('--embeddings-test', action='store_true',help="Skip training and generate test embeddings for evaluation.")
    parser.add_argument('--embeddings-ckpt', type=str, default=None, help="Checkpoint to load when generating embeddings.")
    #parser.add_argument("--embeddings")
    parser.add_argument("--dryrun", action="store_true", help="Dry-run by training on the validtion set. Use only to test loop code.")

    mlflow_save_dir = "./mlruns"  # make into arg?
    tbx_save_dir = "./tensorboard"  # make into arg?

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.tmp_folder is not None:
        data_folder_name = os.path.basename(os.path.normpath(args.data))
        rsync_folder(args.data, args.tmp_folder)
        data_dir = os.path.join(args.tmp_folder, data_folder_name)
        output_dir = os.path.join(args.tmp_folder, 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        data_dir = args.data
        output_dir = args.output

    # to intercept any print statement:
    sys.stdout = LoggerWriter(logger.info)
    sys.stderr = LoggerWriter(logger.warning)

    assert args.config is not None
    with open(args.config, 'r') as stream:
        hyper_params = load(stream, Loader=yaml.FullLoader)
    exp_name = hyper_params["exp_name"]
    output_dir = os.path.join(output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(output_dir, "config.backup"))
    assert "output_dir" not in hyper_params
    hyper_params["output_dir"] = output_dir
    os.makedirs(mlflow_save_dir, exist_ok=True)
    mlf_logger = MLFlowLogger(
        experiment_name=exp_name,
        save_dir=mlflow_save_dir,
    )
    if os.path.exists(os.path.join(output_dir, STAT_FILE_NAME)):
        mlf_logger._run_id = load_mlflow(output_dir)
        logger.warning(f"WILL CONTINUE LOGGING IN MLFLOW RUN ID: {mlf_logger._run_id}")
    os.makedirs(tbx_save_dir, exist_ok=True)
    tbx_logger = TensorBoardLogger(
        save_dir=tbx_save_dir,
        name=exp_name,
        default_hp_metric=False,
    )

    log_path = os.path.join(output_dir, "console.log")
    handler = logging.handlers.WatchedFileHandler(log_path)
    formatter = logging.Formatter(logging.BASIC_FORMAT)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    mlflow.set_experiment(exp_name)
    mlflow.start_run(run_id=mlf_logger.run_id)
    run(args, data_dir, output_dir, hyper_params, mlf_logger, tbx_logger)
    mlflow.end_run()
    if args.tmp_folder is not None:
        rsync_folder(output_dir + os.path.sep, args.output)


def save_list_to_file(filename, string_list):
    with open(filename,"w") as f:
        for string in string_list:
            f.write(f"{string}\n")


def run(args, data_dir, output_dir, hyper_params, mlf_logger, tbx_logger):
    """Setup and run the dataloaders, training loops, etc.

    Args:
        args: arguments passed from the cli
        data_dir (str): path to input folder
        output_dir (str): path to output folder
        hyper_params (dict): hyper parameters from the config file
        mlf_logger (obj): MLFlow logger callback.
        tbx_logger (obj): TensorBoard logger callback.
    """
    # __TODO__ change the hparam that are used from the training algorithm
    # (and NOT the model - these will be specified in the model itself)
    logger.info('List of hyper-parameters:')
    check_and_log_hp(
        ['architecture', 'batch_size', 'exp_name', 'max_epoch', 'optimizer', 'patience', 'seed'],
        hyper_params)

    if hyper_params["seed"] is not None:
        set_seed(hyper_params["seed"])

    if "precision" not in hyper_params:
        hyper_params["precision"] = 16

    if "pairing_strategy" not in hyper_params:
        hyper_params["pairing_strategy"]="next"

    log_exp_details(os.path.realpath(__file__), args)

    
    if args.data_module=="hdf5":
        valid_split_ratio = 0.1
        if not data_dir.endswith(".hdf5"):
            if args.embeddings_test: #Test set for evaluation.
                data_dir = os.path.join(data_dir, "extract_s5_raw_test.hdf5")
                valid_split_ratio=1 #All the test is is used for validation!
            else: #Train/validation set.
                data_dir = os.path.join(data_dir, "extract_s5_raw.hdf5")
        dm = selfsupmotion.data.objectron.hdf5_parser.ObjectronFramePairDataModule(
            hdf5_path=data_dir,
            tuple_length=hyper_params.get("tuple_length"),
            frame_offset=hyper_params.get("frame_offset"),
            tuple_offset=hyper_params.get("tuple_offset"),
            keep_only_frames_with_valid_kpts=hyper_params.get("keep_only_frames_with_valid_kpts"),
            input_height=hyper_params.get("input_height"),
            gaussian_blur=hyper_params.get("gaussian_blur"),
            jitter_strength=hyper_params.get("jitter_strength"),
            batch_size=hyper_params.get("batch_size"),
            num_workers=hyper_params.get("num_workers"),
            use_hflip_augment=hyper_params.get("use_hflip_augment"),
            shared_transform=hyper_params.get("shared_transform"),
            crop_height=hyper_params.get("crop_height"),
            crop_scale=(hyper_params.get("crop_scale_min"), hyper_params.get("crop_scale_max")),
            crop_ratio=(hyper_params.get("crop_ratio_min"), hyper_params.get("crop_ratio_max")),
            crop_strategy=hyper_params.get("crop_strategy"),
            sync_hflip=hyper_params.get("sync_hflip"),
            resort_keypoints=hyper_params.get("resort_keypoints"),
            pairing_strategy=hyper_params.get("pairing_strategy"),
            valid_split_ratio=valid_split_ratio
        )
        dm.setup()

    elif args.data_module=="file":
        dm = selfsupmotion.data.objectron.file_datamodule.ObjectronFileDataModule(
            num_workers=hyper_params["num_workers"],
            batch_size=hyper_params["batch_size"],
            pairing=hyper_params["pairing"],
            dryrun=args.dryrun)
        dm.setup() #In order to have the sample count.

   #se
    
    elif args.data_module=="ucf101":
        from selfsupmotion.data.test.test_file import nop
        #from selfsupmotion.data.ucf101.test_file import nop
        #selfsupmotion.data.objectron
        from selfsupmotion.data.ucf101.file_datamodule import UCF101FileDataModule
        dm = UCF101FileDataModule(
            data_dir=args.data,
            num_workers=hyper_params["num_workers"],
            batch_size=hyper_params["batch_size"],
            pairing=hyper_params["pairing"],
            dryrun=args.dryrun)
        dm.setup() #In order to have the sample count.
    else:
        raise ValueError(f"Invalid datamodule specified on CLI : {args.data_module}")

    if "num_samples" not in hyper_params:
        hyper_params["num_samples"] = None
        if hasattr(dm, "train_dataset"):
            if dm.train_dataset is not None:
                hyper_params["num_samples"] = len(dm.train_dataset)
        #else:
            

    if "num_samples_valid" not in hyper_params:
        if hasattr(dm, "val_dataset"):
            hyper_params["num_samples_valid"] = len(dm.val_dataset)
        else:
            hyper_params["num_samples_valid"] = None

    if "early_stop_metric" not in hyper_params:
        hyper_params["early_stop_metric"] = "val_loss"
    
    if args.embeddings or args.embeddings_test:
        if args.embeddings_ckpt is None:
            raise ValueError("Please manually provide the checkpoints using the --embeddings-ckpt argument")
        model = load_model(hyper_params, checkpoint=args.embeddings_ckpt)
        special = False
        if "SPECIAL:" not in args.embeddings_ckpt:
            ckpt = torch.load(args.embeddings_ckpt)
            print(f"Loading from weights from {args.embeddings_ckpt}")
            model.load_state_dict(ckpt["state_dict"])
        elif args.embeddings_ckpt=="SPECIAL:IMAGENET":
            model.online_network.encoder = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
            model.online_network.encoder.fc = Identity()
            special = True
        elif args.embeddings_ckpt=="SPECIAL:RANDOM":
            model.online_network.encoder = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False)
            model.online_network.encoder.fc = Identity()
            special=True
        else:
            raise ValueError(f"Unsuported special checkpoint: {args.embeddings_ckpt}")

        #model.load_from_checkpoint(args.embeddings_ckpt)
        if args.embeddings:
            generate_embeddings(args, model, datamodule=dm,train=True, image_size=hyper_params["image_size"], special=special, prefix="train_")
            generate_embeddings(args, model, datamodule=dm,train=False, image_size=hyper_params["image_size"], special=special, prefix="")
        elif args.embeddings_test:
            generate_embeddings(args, model, datamodule=dm,train=False, image_size=hyper_params["image_size"], special=special, prefix="test_")
    else:
        save_list_to_file(f"{output_dir}/train_sequences.txt", dm.train_dataset.seq_subset)
        save_list_to_file(f"{output_dir}/val_sequences.txt", dm.val_dataset.seq_subset)
        model = load_model(hyper_params)
        setattr(model, "_tbx_logger", tbx_logger)
        setattr(model, "_mlf_logger", mlf_logger)
        train(
            model=model,
            datamodule=dm,
            output=output_dir,
            use_progress_bar=not args.disable_progressbar,
            start_from_scratch=args.start_from_scratch,
            mlf_logger=mlf_logger,
            tbx_logger=tbx_logger,
            hyper_params=hyper_params,
        )

import torch
from tqdm import tqdm
import torch.nn.functional as F 
import numpy as np

def generate_embeddings(args, model, datamodule, train=True, image_size=224, special=False, prefix=None):
    assert prefix is not None
    if train:
        dataloader = datamodule.train_dataloader(evaluation=True) #Do not use data augmentation for evaluation.
        dataset  = datamodule.train_dataset
        #prefix="train_"
    else:
        dataloader = datamodule.val_dataloader(evaluation=True)
        dataset = datamodule.val_dataset
        #prefix=""

    encoder, projector = model.online_network.encoder, model.online_network.projector
    local_progress=tqdm(dataloader)
    
    model.online_network=model.online_network.to(args.embeddings_device)
    #max_batch = int(args.subset_size*len(dataset)/args.batch_size)
    all_features = torch.zeros((model.feature_dim, len(dataset))).half().cuda()
        #train_features = torch.zeros((encoder.fc.in_features, max_batch*args.batch_size))
        
    #train_labels = torch.zeros(max_batch*args.batch_size, dtype=torch.int64).cuda()
    all_targets = []
    sequence_uids = []
    with torch.no_grad():
        batch_num = 0
        for batch_idx, batch in enumerate(local_progress):
            images1 =  batch["OBJ_CROPS"][0]
            if batch_idx==0:
                save_mosaic("embeddings.png", images1)
            meta = batch["UID"]
            targets = batch["CAT_ID"]
            #images1, _, meta= data
            #images1, _ = images
            images1 = images1.to(args.embeddings_device, non_blocking=True)
            #meta = labels[1:]
            #targets = labels[0]
            base = batch_num*dataloader.batch_size

            #with autocast():
            model.online_network.zero_grad()
            #projector.zero_grad()
            #images1 = resize(images1,image_size)
            #_, z1, h1 = model.online_network(images1)
            #features= projector(encoder(images1)[0])
            #features = z1
            if not special:
                features, _, _ = model.online_network(images1)
            else:
                #Were doing resnet-50 evaluation here!
                features = model.online_network.encoder(images1)
                assert features.shape[1]==2048

                #features=encoder(images)[0]
            features = F.normalize(features, dim=1)
            all_features[:,base:base+len(images1)]=features.t().cpu()
            #train_labels[base:base+len(images)]=labels
            all_targets += targets.cpu().numpy().tolist()
            sequence_uids += meta
            batch_num+=1
            #if batch_num >= max_batch:
            #    break
    
    np.save(f"{args.output}/{prefix}embeddings.npy",all_features.cpu().numpy())
    train_info = np.vstack([np.array(all_targets),np.array(sequence_uids)])
    np.save(f"{args.output}/{prefix}info.npy", train_info)


if __name__ == '__main__':
    main()
