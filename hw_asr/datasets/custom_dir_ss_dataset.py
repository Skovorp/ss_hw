import logging
import random
from typing import List, Dict

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from hw_asr.utils.parse_config import ConfigParser
import os
import json

logger = logging.getLogger(__name__)


class CustomDirSSDataset(Dataset):
    def __init__(
            self,
            path,
            config_parser: ConfigParser,
            wave_augs=None,
            spec_augs=None,
            limit=None,
    ):
        self.path = path
        self.config_parser = config_parser
        self.wave_augs = wave_augs
        self.spec_augs = spec_augs
        self.log_spec = config_parser["preprocessing"]["log_spec"]

        index = self._get_or_load_index(path)
        self._assert_index_is_valid(index)
        self.all_speakers = self.get_all_speakers(index)
        index = self._cut_index(index, limit)

        self._index: List[Dict] = index

    @staticmethod
    def _cut_index(index, limit):
        if limit is None:
            return index 
        else:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            return index[:limit]

    def __getitem__(self, ind):
        ind_res = self._index[ind]
        audio_paths = {
            'mix': ind_res['mix'],
            'refs': ind_res['mix'],
            'targets': ind_res['targets'],
        }
        audio_waves = self.load_audios(audio_paths)
        audio_waves, audio_specs = self.process_waves(audio_waves)
        return {
            "audios": audio_waves,
            "spectrograms": audio_specs,
            "audio_paths": audio_paths,
            "speaker_id": self.all_speakers.index(ind_res['original_speaker_id']),
            'original_speaker_id': ind_res['original_speaker_id'],
        }

    def __len__(self):
        return len(self._index)

    def load_audios(self, paths: Dict[str, str]):
        res = {}
        for key, path in paths.items():
            audio_tensor, sr = torchaudio.load(path)
            audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
            target_sr = self.config_parser["preprocessing"]["sr"]
            if sr != target_sr:
                audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
            res[key] = audio_tensor
        return res

    def process_waves(self, audio_tensor_waves: Dict[str, torch.tensor]):
        res_waves = {}
        res_specs = {}
        for key, audio_tensor_wave in audio_tensor_waves.items():
            with torch.no_grad():
                if self.wave_augs is not None:
                    audio_tensor_wave = self.wave_augs(audio_tensor_wave)
                wave2spec = self.config_parser.init_obj(
                    self.config_parser["preprocessing"]["spectrogram"],
                    torchaudio.transforms,
                )
                audio_tensor_spec = wave2spec(audio_tensor_wave)
                if self.spec_augs is not None:
                    audio_tensor_spec = self.spec_augs(audio_tensor_spec)
                if self.log_spec:
                    audio_tensor_spec = torch.log(audio_tensor_spec + 1e-5)
            res_waves[key] = audio_tensor_wave
            res_specs[key] = audio_tensor_spec
        return res_waves, res_specs


    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert 'mix' in entry, 'no mix'
            assert 'refs' in entry, 'no refs'
            assert 'targets' in entry, 'no targets'
            assert 'original_speaker_id' in entry, 'no speaker_orig_id'

    def get_all_speakers(self, index):
        return list(set([el['original_speaker_id'] for el in index]))

    def _get_or_load_index(self, path):
        index_path = os.path.join(path, "index.json")
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index = json.load(f)
        else:
            index = self._create_index(path)
            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, path):
        mix_path = os.path.join(path, 'mix')
        refs_path = os.path.join(path, 'refs')
        targets_path = os.path.join(path, 'targets')

        assert os.path.exists(mix_path) and os.path.exists(mix_path) and os.path.exists(mix_path), 'bad folders: should be mix + refs + targets'

        mix_files = sorted(os.listdir(mix_path))
        refs_files = sorted(os.listdir(refs_path))
        targets_files = sorted(os.listdir(targets_path))

        assert len(mix_files) == len(refs_files) == len(targets_files), f'number of files in folders differs. len(mix_files) == len(refs_files) == len(targets_files): {len(mix_files)} == {len(refs_files)} == {len(targets_files)}'
        assert set([f.split('-')[0] for f in mix_files]) == set([f.split('-')[0] for f in refs_files]) == set([f.split('-')[0] for f in targets_files]), "ids dont match in folders"

        index = []
        for i in range(len(mix_files)):
            original_speaker_id = refs_files[i].split('_')[0]
            index.append({
                'mix': os.path.join(path, 'mix', mix_files[i]),
                'refs': os.path.join(path, 'refs', refs_files[i]),
                'targets': os.path.join(path, 'targets', targets_files[i]),
                'original_speaker_id': original_speaker_id,
            })
        return index