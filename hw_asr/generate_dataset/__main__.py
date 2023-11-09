from hw_asr.datasets.librispeech_dataset import LibrispeechDataset
from hw_asr.generate_dataset.generate_dataset import LibriSpeechSpeakerFiles, MixtureGenerator
import os

# load both datasets
try:
    _ = LibrispeechDataset('train-clean-100')
except Exception:
    pass

try:
    _ = LibrispeechDataset('test-clean')
except Exception:
    pass


path_train = '/home/ubuntu/ss_hw/data/datasets/librispeech/train-clean-100'
path_val = '/home/ubuntu/ss_hw/data/datasets/librispeech/test-clean'

path_mixtures_train = '/home/ubuntu/ss_hw/data/datasets/mixed_librispeech/train-clean-100'
path_mixtures_val = '/home/ubuntu/ss_hw/data/datasets/mixed_librispeech/test-clean'

speakersTrain = [el.name for el in os.scandir(path_train)]
speakersVal = [el.name for el in os.scandir(path_val)]

speakers_files_train = [LibriSpeechSpeakerFiles(i, path_train, audioTemplate="*.flac") for i in speakersTrain]
speakers_files_val = [LibriSpeechSpeakerFiles(i, path_val, audioTemplate="*.flac") for i in speakersVal]


print("train | total speakers :  ", len(speakers_files_train))
print("train | total recordings :", sum([len(s.files) for s in speakers_files_train]))
print("val   | total speakers :  ", len(speakers_files_val))
print("val   | total recordings :", sum([len(s.files) for s in speakers_files_val]))


# mix ??
mixer_train = MixtureGenerator(speakers_files_train,
                                path_mixtures_train,
                                nfiles=20_000,
                                test=False)

mixer_val = MixtureGenerator(speakers_files_val,
                                path_mixtures_val,
                                nfiles=1500,
                                test=True)

import warnings
warnings.filterwarnings("ignore")


print("\nMixing train:")
mixer_train.generate_mixes(snr_levels=[0, 3, 5],
                           num_workers=8,
                           trim_db=20,
                           vad_db=20,
                           audioLen=3)

print("\nMixing test:")
mixer_val.generate_mixes(snr_levels=[0],
                           num_workers=8,
                           trim_db=None,
                           vad_db=None,
                           audioLen=3)



                    