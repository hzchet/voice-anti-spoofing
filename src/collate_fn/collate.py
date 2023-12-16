import logging
from typing import List

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.T for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1).unsqueeze(1)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    label = torch.tensor([item['label'] for item in dataset_items])
    attack_type = [item['attack_type'] for item in dataset_items]
    speaker_id = [item['speaker_id'] for item in dataset_items]
    
    wav = [item['wav'] for item in dataset_items]
    spectrograms = [item['spectrogram'].squeeze(0) for item in dataset_items]
    spectrograms_batch = pad_sequence(spectrograms)

    if spectrograms_batch.shape[-1] >= 750:
        spectrograms_batch = spectrograms_batch[:, :, :, :750]
    else:
        spectrograms_batch = F.pad(spectrograms_batch, (0, 750 - spectrograms_batch.shape[-1]), mode='constant', value=0)
    
    return {
        'spectrogram': spectrograms_batch,
        'wav': wav,
        'label': label.long(),
        'attack_type': attack_type,
        'speaker_id': speaker_id
    }
