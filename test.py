import os
from pathlib import Path

import hydra
import torch
from tqdm import tqdm
import torch.nn.functional as F

import src.model as module_model
from src.trainer import Trainer
from src.utils import ROOT_PATH
from src.utils.data_loading import get_dataloaders
from src.utils.parse_config import ConfigParser


DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


@hydra.main(config_path="/workspace/configs", config_name="config.yaml")
def main(config):
    config = ConfigParser(config)
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config['resume']))
    checkpoint = torch.load(config['resume'], map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        key = 'test'
        logger.info(f'{key}:')
        tensors_for_gpu = ['wav', 'label']
        result = []
        for batch in tqdm(dataloaders[key]):
            batch = Trainer.move_batch_to_device(batch, tensors_for_gpu, device)
            logits = model(**batch)['logits'].detach().cpu()
            probs = F.softmax(logits, dim=1)
            for i, audio_name in enumerate(batch['attack_type']):
                spoof_score = probs[i, 1]
                result.append({
                    "audio_name": audio_name,
                    "spoof_score": spoof_score
                })
             
        for item in result:
            audio_name = item['audio_name']
            spoof_score = item['spoof_score']
            print(f'Spoof score for the "{audio_name}": {spoof_score}')


if __name__ == "__main__":
    main()
