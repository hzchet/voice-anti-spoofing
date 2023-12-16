import os
from typing import List

import torchaudio

from src.base.base_dataset import BaseDataset


class InferenceDataset(BaseDataset):
    def __init__(self, path_to_audios: str, *args, **kwargs):
        assert os.path.exists(path_to_audios)
        index = self._get_index(path_to_audios)
        super().__init__(index, *args, **kwargs)
        
    def _get_index(self, path_to_audios):
        index = []
        audios = os.listdir(path_to_audios)
        for audio_name in audios:
            audio_path = os.path.join(path_to_audios, audio_name)
            
            index.append({
                "audio_path": audio_path,
                "attack_type": audio_name,
                "speaker_id": "-",
                "label": -1
            })
            
        return index
