import logging
from typing import List
import torch.nn.functional as F
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    # 'audios', 'spectrograms', 'audio_paths', speaker_id, original_speaker_id
    result_batch = {
        'audios': {'mix': [], 'refs': [], 'targets': []},
        'spectrograms': {'mix': [], 'refs': [], 'targets': []},
        'audio_paths': {'mix': [], 'refs': [], 'targets': []},
        'speaker_ids': [],
        'original_speaker_ids': [],
    }

    max_spectrogram_len = {
        'mix': max(item['spectrograms']['mix'].size(2) for item in dataset_items),
        'refs': max(item['spectrograms']['refs'].size(2) for item in dataset_items),
        'targets': max(item['spectrograms']['targets'].size(2) for item in dataset_items)
    }

    max_wave_len = {
        'mix': max(item['audios']['mix'].size(1) for item in dataset_items),
        'refs': max(item['audios']['refs'].size(1) for item in dataset_items),
        'targets': max(item['audios']['targets'].size(1) for item in dataset_items)
    }


    for item in dataset_items:
        result_batch['speaker_ids'].append(item['speaker_id'])
        result_batch['original_speaker_ids'].append(item['original_speaker_id'])
        for key in ['mix', 'refs', 'targets']:
            spectrogram_pad = (0, max_spectrogram_len[key] - item['spectrograms'][key].size(2))
            result_batch['spectrograms'][key].append(F.pad(item['spectrograms'][key], spectrogram_pad, 'constant', 0))

            wave_pad = (0, max_wave_len[key] - item['audios'][key].size(1))
            result_batch['audios'][key].append(F.pad(item['audios'][key], wave_pad, 'constant', 0))

            result_batch['audio_paths'][key].append(item['audio_paths'][key])

    for key in ['mix', 'refs', 'targets']:
        result_batch['spectrograms'][key] = torch.stack(result_batch['spectrograms'][key])
        result_batch['audios'][key] = torch.stack(result_batch['audios'][key])
    result_batch['speaker_ids'] = torch.LongTensor(result_batch['speaker_ids'])

    return result_batch