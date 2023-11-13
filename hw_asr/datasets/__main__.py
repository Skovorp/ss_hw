from hw_asr.datasets.custom_dir_ss_dataset import CustomDirSSDataset
from hw_asr.utils.parse_config import ConfigParser
from tqdm import tqdm

config_parser = ConfigParser.get_test_configs()
dataset_test = CustomDirSSDataset('/home/ubuntu/ss_hw/data/datasets/mixed_librispeech/test-clean', config_parser)
dataset_train = CustomDirSSDataset('/home/ubuntu/ss_hw/data/datasets/mixed_librispeech/train-clean-100', config_parser)

for i, el in tqdm(enumerate(dataset_train)):
    assert el['audios']['mix'].shape == el['audios']['targets'].shape, i
print("ok")