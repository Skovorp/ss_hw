from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio


class SiSdrMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_sdr = ScaleInvariantSignalDistortionRatio().to(torch.cuda.current_device())

    def __call__(self, s1: Tensor, audios, **kwargs):
        pred = s1
        targ = audios['targets']
        # if abs(pred.shape[-1] - targ.shape[-1]) < 5:
        #     pred = pred[:, :, :min(pred.shape[-1], targ.shape[-1])]
        #     targ = targ[:, :, :min(pred.shape[-1], targ.shape[-1])]
        return self.si_sdr(pred, targ)
