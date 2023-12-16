import json
from pathlib import Path

from src.base.base_dataset import BaseDataset
from src.utils import ROOT_PATH


class AVSDataset(BaseDataset):
    def __init__(self, split: str, data_dir=None, *args, **kwargs):
        assert split in ('train', 'dev', 'eval')
        
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "LA"
        elif isinstance(data_dir, str):
            data_dir = Path(data_dir)
            
        assert data_dir.exists()
        
        self._data_dir = data_dir
        index = self._get_or_create_index(split)
            
        super().__init__(index, *args, **kwargs)
        
    def _get_or_create_index(self, split):
        index_path = self._data_dir / f"{split}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(split)
            with index_path.open('w') as f:
                json.dump(index, f, indent=2)
        
        return index
    
    def _create_index(self, split):
        ext = 'trn' if split == 'train' else 'trl'
        protocol_path = self._data_dir / "ASVspoof2019_LA_cm_protocols" / \
            f"ASVspoof2019.LA.cm.{split}.{ext}.txt"
        
        audios_path = self._data_dir / f"ASVspoof2019_LA_{split}/flac"
        assert audios_path.exists()
        audios_path = str(audios_path)
        
        index = []
        with protocol_path.open('r') as f:
            for line in f:
                items = line.strip().split(' ')
                index.append({
                    "speaker_id": items[0],
                    "audio_path": f"{audios_path}/{items[1]}.flac",
                    "attack_type": items[-2],
                    "label": 1 if items[-1] == 'spoof' else 0
                })
        
        return index
