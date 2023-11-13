import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from hw_asr.loss.speaker_ce_loss import SpeakerCrossEntropyLoss
from torch import nn
from torch.linalg import vector_norm


class MixedLoss(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super().__init__()
        self.ce_loss = SpeakerCrossEntropyLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


    def forward(self, **batch) -> Tensor:
        return self.ce_loss(**batch) * self.gamma + self.forward_sisdr(**batch)
    
    def forward_sisdr(self, s1, s2, s3, audios, **batch):
        short = self.calc_sisdr(s1[:, 0, :], audios['targets'][:, 0, :])
        medium = self.calc_sisdr(s2[:, 0, :], audios['targets'][:, 0, :])
        long = self.calc_sisdr(s3[:, 0, :], audios['targets'][:, 0, :])

        return (1 - self.alpha - self.beta) * short + self.alpha * medium + self.beta * long

    @staticmethod
    def calc_sisdr(pred, target):
        k = (pred * target).sum(1, keepdim=True) / (target * target).sum(1, keepdim=True)
        res = vector_norm(k * target, dim=1) / vector_norm(k * target - pred, dim=1)
        res = -20 * torch.log10(res)
        res = res.mean()
        return res
