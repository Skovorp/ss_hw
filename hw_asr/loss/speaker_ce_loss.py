import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss


class SpeakerCrossEntropyLoss(CrossEntropyLoss):
    def forward(self, speaker_logits, speaker_ids, **batch) -> Tensor:
        return super().forward(speaker_logits, speaker_ids)