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

from hw_asr.base import BaseTrainer
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.logger.utils import plot_spectrogram_to_buf
from hw_asr.metric.utils import calc_cer, calc_wer
from hw_asr.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "loss", "grad norm", *[m.name for m in self.metrics], writer=self.writer
        )

        eval_metrics_names = []
        for m in self.metrics:
            if m.name != 'speaker acc':
                eval_metrics_names.append(m.name)
        self.evaluation_metrics = MetricTracker(
            # "loss",
            *eval_metrics_names, writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        batch['audios']['mix'] = batch['audios']['mix'].to(device)
        batch['audios']['refs'] = batch['audios']['refs'].to(device)
        batch['audios']['targets'] = batch['audios']['targets'].to(device)
        batch['speaker_ids'] = batch['speaker_ids'].to(device)

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
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                    do_step = (batch_idx % 2 == 1)
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
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.optimizer.param_groups[0]['lr']
                )
                # self._log_predictions(**batch)
                # self._log_spectrogram(batch["spectrogram"])
                self._log_scalars(self.train_metrics) 
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker, do_step=True):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            if do_step == False:
                self.optimizer.zero_grad()
            outputs = self.model(**batch, predict_speaker=True)
            batch.update(outputs)
        else:
            outputs = self.model(**batch, predict_speaker=False)
            batch.update(outputs)

        if is_train:
            batch["loss"] = self.criterion(**batch)
            batch["loss"].backward()
            self._clip_grad_norm()
            if do_step:
                self.optimizer.step()
            metrics.update("loss", batch["loss"].item())

        for met in self.metrics:
            # this template clearly didnt expect different metrics for train/val
            if met.name == 'pesq' and is_train:
                continue
            if met.name == 'speaker acc' and not is_train:
                continue
            with torch.no_grad():
                metric_val = met(**batch)
                batch[met.name] = metric_val
                metrics.update(met.name, metric_val)
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(batch['si_sdr'].item())
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            # self._log_predictions(**batch)
            # self._log_spectrogram(batch["spectrogram"])
            self._log_audio(batch)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    # def _log_predictions(
    #         self,
    #         text,
    #         log_probs,
    #         log_probs_length,
    #         audio_path,
    #         examples_to_log=10,
    #         *args,
    #         **kwargs,
    # ):
    #     # TODO: implement logging of beam search results
    #     if self.writer is None:
    #         return
    #     argmax_inds = log_probs.cpu().argmax(-1).numpy()
    #     argmax_inds = [
    #         inds[: int(ind_len)]
    #         for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
    #     ]
    #     argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
    #     argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]
    #     tuples = list(zip(argmax_texts, text, argmax_texts_raw, audio_path))
    #     shuffle(tuples)
    #     rows = {}
    #     for pred, target, raw_pred, audio_path in tuples[:examples_to_log]:
    #         target = BaseTextEncoder.normalize_text(target)
    #         wer = calc_wer(target, pred) * 100
    #         cer = calc_cer(target, pred) * 100

    #         rows[Path(audio_path).name] = {
    #             "target": target,
    #             "raw prediction": raw_pred,
    #             "predictions": pred,
    #             "wer": wer,
    #             "cer": cer,
    #         }
    #     self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))

    # def _log_spectrogram(self, spectrogram_batch):
    #     spectrogram = random.choice(spectrogram_batch.cpu())
    #     image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
    #     self.writer.add_image("spectrogram", ToTensor()(image))

    def _log_audio(self, batch):
        batch_size = batch['audios']['mix'].shape[0]
        ind = random.randint(0, batch_size - 1)
        
        self.writer.add_audio('mix', batch['audios']['mix'][ind, :, :], sample_rate=16000)
        self.writer.add_audio('ref', batch['audios']['refs'][ind, :, :], sample_rate=16000)
        self.writer.add_audio('target', batch['audios']['targets'][ind, :, :], sample_rate=16000)
        self.writer.add_audio('s1', batch['s1'][ind, :, :], sample_rate=16000)
        self.writer.add_audio('s2', batch['s2'][ind, :, :], sample_rate=16000)
        self.writer.add_audio('s3', batch['s3'][ind, :, :], sample_rate=16000)

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
