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

class GlobalNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-6, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.eps = eps  
        
        nn.init.ones_(self.scale)
        nn.init.zeros_(self.bias)


    def forward(self, x):
        mean = x.mean(dim=(1, 2), keepdim=True)
        variance = x.var(dim=(1, 2), keepdim=True, unbiased=False)
        
        normalized_inputs = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * normalized_inputs + self.bias



class StackedTCN(nn.Module):
    def __init__(self, blocks_in_stack, tcn_input_channels, tch_hidden_channels, user_emb_dim, **kwargs):
        super().__init__()
        blocks = [TCNBlock(tcn_input_channels, tch_hidden_channels, needs_concat=(i==0), dilation=2**i, user_emb_dim=user_emb_dim) 
                  for i in range(blocks_in_stack)]
        self.stack = nn.ModuleList(blocks)
    
    def forward(self, x, user_emb):
        for i, layer in enumerate(self.stack):
            if i == 0:
                x = layer(x, user_emb)
            else:
                x = layer(x)
        return x


class TCNBlock(nn.Module):
    def __init__(self, tcn_input_channels, tch_hidden_channels, needs_concat, dilation, user_emb_dim=None, **kwargs):
        super().__init__()

        if needs_concat:
            assert user_emb_dim is not None, "need user_emb_dim!"

        self.needs_concat = needs_concat
        if needs_concat:
            self.first_conv = nn.Conv1d(tcn_input_channels + user_emb_dim, tch_hidden_channels, 1)
        else:
            self.first_conv = nn.Conv1d(tcn_input_channels, tch_hidden_channels, 1)
        
        self.stack = nn.Sequential(
            nn.PReLU(),
            GlobalNorm(tch_hidden_channels),
            nn.Conv1d(tch_hidden_channels, tch_hidden_channels, kernel_size=3, 
                      dilation=dilation, groups=tch_hidden_channels, padding=((3 - 1) * dilation) // 2),
            nn.PReLU(),
            GlobalNorm(tch_hidden_channels),
            nn.Conv1d(tch_hidden_channels, tcn_input_channels, 1),
        )

    def forward(self, x, speaker_emb=None):
        residual = x
        if self.needs_concat:
            assert speaker_emb is not None, "pass speaker_emb!"
            # print('speaker_emb.shape raw', speaker_emb.shape)
            # print('x shape', x.shape)
            speaker_emb = torch.unsqueeze(speaker_emb, 2).expand(-1, -1, x.shape[2])
            x = torch.cat([speaker_emb, x], 1)
            # print("cat shape", x.shape)
        x = self.first_conv(x)
        x = self.stack(x)
        # print("passed shape", x.shape)
        return residual + x


class SpeechDecoder(nn.Module):
    def __init__(self, input_channels, encoder_channels, encoder_params):
        super().__init__()
        self.cnn = nn.Conv1d(input_channels, encoder_channels, 1)
        self.relu = nn.ReLU()
        self.deconv = nn.ConvTranspose1d()

    def forward(self, extracted_speech, encoded_mix):
        extracted_speech = self.cnn(extracted_speech)
        extracted_speech = self.relu(extracted_speech)
        res = extracted_speech * encoded_mix
        res = self.deconv(res)
        return res 


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
        self.mix_1d = nn.Conv1d(encoded_dim, tcn_params['tcn_input_channels'], 1) 
        self.speaker_tcn_stack = nn.ModuleList([StackedTCN(user_emb_dim=resnet_params['resnet_input_channels'], **tcn_params) for i in range(tcn_params['n_stacks'])])
        
        # decoder
        self.decoder_conv_short = nn.Conv1d(tcn_params['tcn_input_channels'], encoder_params['feature_dim'], 1)
        self.decoder_conv_medium = nn.Conv1d(tcn_params['tcn_input_channels'], encoder_params['feature_dim'], 1)
        self.decoder_conv_long = nn.Conv1d(tcn_params['tcn_input_channels'], encoder_params['feature_dim'], 1)
        
        self.deconv_short = nn.ConvTranspose1d(encoder_params['feature_dim'], 1, kernel_size=encoder_params['short_len'], 
                                       stride=encoder_params['short_len'] // 2)
        self.deconv_medium = nn.ConvTranspose1d(encoder_params['feature_dim'], 1, kernel_size=encoder_params['medium_len'], 
                                        stride=encoder_params['short_len'] // 2)
        self.deconv_long = nn.ConvTranspose1d(encoder_params['feature_dim'], 1, kernel_size=encoder_params['long_len'], 
                                        stride=encoder_params['short_len'] // 2)

    def forward(self, audios, predict_speaker, **batch):
        ref_encoded, (y1, y2, y3) = self.forward_encoder(audios['refs'], True) # (batch, 3 * encoder_feature_dim, shorter_len)
        mix_encoded = self.forward_encoder(audios['mix'])  # (batch, 3 * encoder_feature_dim, shorter_len)

        speaker_emb = self.speaker_encoder(ref_encoded)

        res = {}
        if predict_speaker:
            res['speaker_logits'] = self.proj_speaker(speaker_emb)

        extracted = self.speaker_extractor(mix_encoded, speaker_emb)
        s1, s2, s3 = self.decode(extracted, y1, y2, y3)
        res['s1'] = s1
        res['s2'] = s2
        res['s3'] = s3

        assert s1.shape == s2.shape == s3.shape == audios['mix'].shape, f"\
            s1.shape {s1.shape} | s2.shape {s2.shape} | s3.shape {s3.shape} | mix.shape {audios['mix'].shape}"

        return res

    def forward_encoder(self, wave, return_separate=False):
        padded_for_medium = F.pad(wave, (0, self.encoder_params['medium_len'] - self.encoder_params['short_len']))
        padded_for_long = F.pad(wave, (0, self.encoder_params['long_len'] - self.encoder_params['short_len']))

        short = self.short_encoder(wave)   # (batch, encoder_feature_dim, shorter_len)
        medium = self.medium_encoder(padded_for_medium) # (batch, encoder_feature_dim, shorter_len)
        long = self.long_encoder(padded_for_long)     # (batch, encoder_feature_dim, shorter_len)
        cat_emb = torch.cat([short, medium, long], 1) # (batch, 3 * encoder_feature_dim, shorter_len)

        if return_separate:
            return F.relu(cat_emb), (F.relu(short), F.relu(medium), F.relu(long))
        else:
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
        return encoded
    
    def decode(self, x, y1, y2, y3):
        short = F.relu(self.decoder_conv_short(x)) * y1
        medium = F.relu(self.decoder_conv_medium(x)) * y2
        long = F.relu(self.decoder_conv_long(x)) * y3

        # print("y1.shape", y1.shape)
        # print("y2.shape", y2.shape)
        # print("y3.shape", y3.shape)

        short = self.deconv_short(short)
        medium = self.deconv_medium(medium)[:, :, :short.shape[2]]
        long = self.deconv_long(long)[:, :, :short.shape[2]]
        return short, medium, long

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + f"\nTrainable parameters: {params:,}"
