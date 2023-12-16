import random

import torchaudio

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self, 
        index,
        limit=None,
        frontend: str = 'stft',
        *args,
        **kwargs
    ):
        super().__init__()
        self._index = self._filter_records_from_dataset(index, limit)
        if frontend == 'stft':
            self.wav2spec = torchaudio.transforms.Spectrogram(n_fft=512, win_length=320)
        else:
            raise ValueError()
        
    def __getitem__(self, ind):
        item = self._index[ind]
        wav, sr = torchaudio.load(item['audio_path'])
        if sr != 16000:
            wav= torchaudio.functional.resample(wav, sr, 16000)
        
        spec = self.wav2spec(wav)
        
        return {
            "spectrogram": spec,
            "label": item["label"],
            "wav": wav,
            "attack_type": item["attack_type"],
            "speaker_id": item["speaker_id"]
        }
    
    def __len__(self):
        return len(self._index)

    @staticmethod
    def _filter_records_from_dataset(
            index: list, limit
    ) -> list:
        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index
