from torch import nn
import torch
import torch.nn.functional as F
import numpy as np


class ResNetBlock(nn.Module):
    def __init__(self, resnet_input_channels, resnet_hidden_channels, **kwargs):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(resnet_input_channels, resnet_hidden_channels, 1),
            nn.BatchNorm1d(resnet_hidden_channels),
            nn.PReLU(),
            nn.Conv1d(resnet_hidden_channels, resnet_input_channels, 1),
            nn.BatchNorm1d(resnet_input_channels),
        )
        self.prelu = nn.PReLU()
        self.pool = nn.MaxPool1d(3)

    def forward(self, x):
        x = x + self.block(x)
        x = self.prelu(x)
        x = self.pool(x)
        return x


class TCNBlock(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x, speaker_emb):
        return F.relu(x)


class MySpExModel(nn.Module):
    def __init__(self, encoder_params, tcn_params, resnet_params, n_train_speakers, **batch):
        super().__init__()

        # print(encoder_params, tcn_params, resnet_params, n_train_speakers, batch)
        self.encoder_params = encoder_params
        self.tcn_params = tcn_params
        self.resnet_params = resnet_params
        self.n_train_speakers = n_train_speakers

        # encoder
        self.short_encoder = nn.Conv1d(1, out_channels=encoder_params['feature_dim'], kernel_size=encoder_params['short_len'], 
                                       stride=encoder_params['short_len'] // 2)
        self.medium_encoder = nn.Conv1d(1, out_channels=encoder_params['feature_dim'], kernel_size=encoder_params['medium_len'], 
                                        stride=encoder_params['short_len'] // 2)
        self.long_encoder = nn.Conv1d(1, out_channels=encoder_params['feature_dim'], kernel_size=encoder_params['long_len'], 
                                      stride=encoder_params['short_len'] // 2)

        encoded_dim = encoder_params['feature_dim'] * 3

        # speaker encoder
        self.speaker_norm = nn.LayerNorm(encoded_dim)
        self.speaker_encoder_block = nn.Sequential(
            nn.Conv1d(encoded_dim, resnet_params['resnet_input_channels'], 1),
            nn.Sequential(*[ResNetBlock(**resnet_params) for i in range(resnet_params['n'])]),
            nn.Conv1d(resnet_params['resnet_input_channels'], resnet_params['resnet_input_channels'], 1) 
        )
        self.proj_speaker = nn.Linear(resnet_params['resnet_input_channels'], n_train_speakers)
        
        # speech extractor
        self.mix_norm = nn.LayerNorm(encoded_dim)
        self.mix_1d = nn.Conv2d(encoded_dim, encoded_dim, (1, 1)) 
        self.speaker_tcn_stack = nn.ModuleList([TCNBlock() for i in range(tcn_params['n'])])
        


    def forward(self, audios, predict_speaker, **batch):
        ref_encoded = self.forward_encoder(audios['refs']) # (batch, 3 * encoder_feature_dim, shorter_len)
        mix_encoded = self.forward_encoder(audios['mix'])  # (batch, 3 * encoder_feature_dim, shorter_len)

        speaker_emb = self.speaker_encoder(ref_encoded)

        res = {}
        if predict_speaker:
            res['speaker_logits'] = self.proj_speaker(speaker_emb)

        return res

    def forward_encoder(self, wave):
        padded_for_medium = F.pad(wave, (0, self.encoder_params['medium_len'] - self.encoder_params['short_len']))
        padded_for_long = F.pad(wave, (0, self.encoder_params['long_len'] - self.encoder_params['short_len']))

        short = self.short_encoder(wave)   # (batch, encoder_feature_dim, shorter_len)
        medium = self.medium_encoder(padded_for_medium) # (batch, encoder_feature_dim, shorter_len)
        long = self.long_encoder(padded_for_long)     # (batch, encoder_feature_dim, shorter_len)
        cat_emb = torch.cat([short, medium, long], 1) # (batch, 3 * encoder_feature_dim, shorter_len)
        return F.relu(cat_emb)

    def speaker_encoder(self, encoded):
        encoded = self.speaker_norm(encoded.transpose(1, 2)).transpose(1, 2)
        encoded = self.speaker_encoder_block(encoded)
        return encoded.mean(2) 

    def speaker_extractor(self, encoded, speaker_emb):
        encoded = self.mix_norm(encoded.transpose(1, 2)).transpose(1, 2)
        encoded = self.mix_1d(encoded)
        for tcn in self.speaker_tcn_stack:
            encoded = tcn(encoded, speaker_emb)

        # return mask_short, mask_medium, mask_long

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
