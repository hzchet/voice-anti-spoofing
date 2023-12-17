import argparse
import collections
import warnings

import numpy as np
import torch
import hydra

import src.loss as module_loss
import src.metric as module_metric
import src.model as module_arch
from src.trainer import Trainer
from src.utils import prepare_device
from src.utils.data_loading import get_dataloaders
from src.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


@hydra.main(config_path="/workspace/configs", config_name="config.yaml")
def main(config):
    config = ConfigParser(config)
    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    loss_module = config.init_obj(config["loss"], module_loss).to(device)
    train_metrics = [
        config.init_obj(metric_dict, module_metric)
        for metric_dict in config["train_metrics"]
    ]
    valid_metrics = [
        config.init_obj(metric_dict, module_metric)
        for metric_dict in config["valid_metrics"]
    ]
    
    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj(config["optimizer"], torch.optim, trainable_params)
    lr_scheduler = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model,
        loss_module,
        train_metrics,
        valid_metrics,
        optimizer,
        config=config,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()


if __name__ == "__main__":
    main()
