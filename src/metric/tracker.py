import pandas as pd
import numpy as np

from src.metric.eer import compute_eer


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self._logits = []
        self._labels = []
        self._eer = None
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0
        
        self._logits.clear()
        self._labels.clear()
        self._eer = None

    def update(self, key, value, n=1):
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        if key != 'eer':
            self._data.total[key] += value * n
            self._data.counts[key] += n
            self._data.average[key] = self._data.total[key] / self._data.counts[key]
        else:
            logits, labels = value
            self._logits.append(logits)
            self._labels.append(labels)
        
    def avg(self, key):
        if key != 'eer':
            return self._data.average[key]
    
        logits = np.concatenate(self._logits, axis=0)
        labels = np.concatenate(self._labels, axis=0)
        
        bonafide_scores = logits[labels == 0]
        other_scores = logits[labels == 1]
        
        self._eer = compute_eer(bonafide_scores, other_scores)[0]
        return self._eer

    def result(self):
        res = dict(self._data.average)
        res.update({
            "eer": self.avg("eer") if self._eer is None else self._eer
        })
        return res

    def keys(self):
        keys = list(self._data.total.keys())
        keys.append("eer")
        return keys
