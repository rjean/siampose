import glob
import logging
import os
import yaml

import mlflow
import torch
import pl_bolts
import pytorch_lightning as pl
import orion.client
from siampose.checkpointing import ModelCheckpointLastOnly
from pytorch_lightning import Callback

logger = logging.getLogger(__name__)

STAT_FILE_NAME = "stats.yaml"


def train(**kwargs):  # pragma: no cover
    """Training loop wrapper. Used to catch exception if Orion is being used."""
    try:
        best_dev_metric = train_impl(**kwargs)
    except RuntimeError as err:
        if orion.client.cli.IS_ORION_ON and "CUDA out of memory" in str(err):
            logger.error(err)
            logger.error(
                "model was out of memory - assigning a bad score to tell Orion to avoid"
                "too big model"
            )
            best_dev_metric = -999
        else:
            raise err

    orion.client.report_results(
        [
            dict(
                name="dev_metric",
                type="objective",
                # note the minus - cause orion is always trying to minimize (cit. from the guide)
                value=-float(best_dev_metric),
            )
        ]
    )


def load_mlflow(output):
    """Load the mlflow run id.
    Args:
        output (str): Output directory
    """
    with open(os.path.join(output, STAT_FILE_NAME), "r") as stream:
        stats = yaml.load(stream, Loader=yaml.FullLoader)
    return stats["mlflow_run_id"]


def write_mlflow(output):
    """Write the mlflow info to resume the training plotting..
    Args:
        output (str): Output directory
    """
    mlflow_run = mlflow.active_run()
    mlflow_run_id = mlflow_run.info.run_id if mlflow_run is not None else "NO_MLFLOW"
    to_store = {"mlflow_run_id": mlflow_run_id}
    with open(os.path.join(output, STAT_FILE_NAME), "w") as stream:
        yaml.dump(to_store, stream)


def train_impl(
    model,
    datamodule,
    output,
    use_progress_bar,
    start_from_scratch,
    mlf_logger,
    tbx_logger,
    hyper_params,
):  # pragma: no cover

    write_mlflow(output)

    last_models = glob.glob(os.path.join(output, "last-*"))
    if start_from_scratch:
        logger.info("will not load any pre-existent checkpoint.")
        resume_from_checkpoint = None
    elif len(last_models) > 1:
        raise ValueError("more than one last model found to resume - provide only one")
    elif len(last_models) == 1:
        logger.info("resuming training from {}".format(last_models[0]))
        resume_from_checkpoint = last_models[0]
    else:
        logger.info("no model found - starting training from scratch")
        resume_from_checkpoint = None

    early_stopping, callbacks = None, []
    # from selfsupmotion.checkpointing import ModelCheckpointLastOnly
    if hyper_params["early_stop_metric"] not in ["None", "none", "", None]:
        early_stopping = pl.callbacks.EarlyStopping(
            monitor=hyper_params["early_stop_metric"],
            patience=hyper_params["patience"],
            verbose=use_progress_bar,
            mode="auto",
        )
        best_checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=output,
            filename="best-{epoch}-{step}",
            monitor=hyper_params["early_stop_metric"],
            verbose=use_progress_bar,
            mode="auto",
        )
        callbacks.extend([early_stopping, best_checkpoint_callback])

    callbacks.extend(
        [
            pl_bolts.callbacks.PrintTableMetricsCallback(),
            ModelCheckpointLastOnly(
                dirpath=output,
                filename="last-{epoch}-{step}",
                # verbose=use_progress_bar,
                # monitor=hyper_params["early_stop_metric"],
                # mode="max", #We have to hack arround to save the last checkpoint apparently!
                verbose=True,
                # save_top_k=3, #Just make sure that we save the last checkpoint.
                # save_last=True,
            ),
        ]
    )

    trainer = pl.Trainer(
        # @@@@@@@@@@@ TODO check if we can add an online evaluator w/ callback
        callbacks=callbacks,
        #checkpoint_callback=True,
        logger=[mlf_logger, tbx_logger],
        max_epochs=hyper_params["max_epoch"],
        resume_from_checkpoint=resume_from_checkpoint,
        gpus=torch.cuda.device_count(),
        auto_select_gpus=True,
        precision=hyper_params["precision"],
        #amp_level="O1",
        accelerator=None,
        accumulate_grad_batches=hyper_params.get("accumulate_grad_batches", 1),
    )

    trainer.fit(model, datamodule=datamodule)
    if early_stopping is not None:
        best_dev_result = float(early_stopping.best_score.cpu().numpy())
    else:
        return -999
    return best_dev_result
