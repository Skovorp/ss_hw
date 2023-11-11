from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric


class SpeakerAccuracyMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, speaker_logits: Tensor, speaker_ids: Tensor, **kwargs):
        return (speaker_logits.argmax(dim=1) == speaker_ids).float().mean()
