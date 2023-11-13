from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality


class PESQMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')

    def __call__(self, s1: Tensor, audios, **kwargs):
        pred = s1.cpu()
        targ = audios['targets'].cpu()
        # if abs(pred.shape[-1] - targ.shape[-1]) < 5:
        #     pred = pred[:, :, :min(pred.shape[-1], targ.shape[-1])]
        #     targ = targ[:, :, :min(pred.shape[-1], targ.shape[-1])]
        return self.pesq(pred, targ)
