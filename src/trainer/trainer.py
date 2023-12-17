import logging
import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.base import BaseTrainer
from src.logger.utils import plot_spectrogram_to_buf
from src.metric.tracker import MetricTracker


logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            train_metrics,
            valid_metrics,
            optimizer,
            config,
            device,
            dataloaders,
            log_step=None,
            lr_scheduler=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, train_metrics, valid_metrics, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        self.len_epoch = len(self.train_dataloader)
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = log_step
        if self.log_step is None:
            self.log_step = len(self.train_dataloader)

        self.train_metrics_tracker = MetricTracker(
            "loss", "grad norm", *[m.name for m in self.train_metrics], writer=self.writer
        )
        self.evaluation_metrics_tracker = MetricTracker(
            "loss", *[m.name for m in self.valid_metrics], writer=self.writer
        )
        
        self.step_every_n_epochs = self.config["trainer"].get("step_every_n_epochs", 10)
        
        if self.train_dataloader.dataset.frontend in ('s1', 's2', 's3'):
            self.gpu_keys = ['wav', 'label']
        else:
            self.gpu_keys = ['spectrogram', 'label']
    
    @staticmethod
    def move_batch_to_device(batch, keys, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in keys:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics_tracker.reset()
        self.writer.mode = 'train'
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics_tracker,
                    batch_idx=batch_idx
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics_tracker.update("grad norm", self.get_grad_norm())
            if (batch_idx + 1) % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_predictions(**batch)
                self._log_scalars(self.train_metrics_tracker)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics_tracker.result()
                self.train_metrics_tracker.reset()
            if batch_idx + 1 == self.len_epoch:
                break

        log = last_train_metrics
        if (epoch + 1) % self.step_every_n_epochs == 0:
            self.lr_scheduler.step()
        
        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            if val_log is not None:
                log.update(**{f"{part}_{name}": value for name, value in val_log.items()})
        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker, batch_idx: int, part=None):
        if part == 'test':
            return self.inference(batch)

        batch = self.move_batch_to_device(batch, self.gpu_keys, self.device)
        if is_train:
            self.optimizer.zero_grad()
        outputs = self.model(**batch)
        if type(outputs) is dict:
            batch.update(outputs)
        else:
            batch["logits"] = outputs

        batch["loss"] = self.criterion(**batch)
        if is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
                
        metrics.update("loss", batch["loss"].item())
        if is_train:
            for met in self.train_metrics:
                metrics.update(met.name, met(**batch))
        else:
            for met in self.valid_metrics:
                metrics.update(met.name, met(**batch))
        return batch

    def inference(self, batch):
        batch = self.move_batch_to_device(batch, self.gpu_keys, self.device)
        outputs = self.model(**batch, is_inference=True)
        if type(outputs) is dict:
            batch.update(outputs)
        else:
            batch["logits"] = outputs
        
        return batch
    
    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.writer.mode = part
        self.evaluation_metrics_tracker.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics_tracker,
                    batch_idx=batch_idx,
                    part=part
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            if part != 'test':
                self._log_scalars(self.evaluation_metrics_tracker)
            self._log_predictions(**batch)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        if part != 'test':
            return self.evaluation_metrics_tracker.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            label,
            logits,
            wav,
            attack_type,
            speaker_id,
            examples_to_log: int = 10,
            *args,
            **kwargs,
    ):
        batch_size = len(wav)
        random_indices = torch.randperm(batch_size)[:examples_to_log]
        
        rows = []
        for i in random_indices:
            i_label = int(label[i])
            i_bonafide_score = F.softmax(logits[i, :])[0]
            i_speaker_id = speaker_id[i]
            i_attack = attack_type[i]
            i_wav = wav[i]
            
            if i_label == 1:
                i_label = 'spoofed'
            elif i_label == 0:
                i_label = 'bonafide'
            else:
                i_label = 'unkown'

            rows.append({
                "speaker_id": i_speaker_id,
                "attack_type": i_attack,
                "label": i_label,
                "audio": self.writer.create_audio_entry(i_wav),
                "bonafide_score": float(i_bonafide_score)
            })
        
        df = pd.DataFrame(rows)
        self.writer.add_table('predictions', df)
    
    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
