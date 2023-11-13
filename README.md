# ASR project barebones

changing names to ss everywhere was to much, so it's called asr everywhere

## Installation guide

```shell
pip install -r ./requirements.txt
```

load model from google drive: 
```
gdown https://drive.google.com/drive/folders/1Wd4--Bwi3sLFB-YKBhaQinioxkqVxisl
```

you can test the model by calling:
``` 
python test.py -c /home/ubuntu/ss_hw/hw_asr/configs/for_test.json -r "/home/ubuntu/ss_hw/saved/models/traing config/1113_180604/checkpoint-epoch26.pth" -t "/home/ubuntu/ss_hw/data/datasets/mixed_librispeech/test-clean"
```
where you substitute path to loaded checkpoint and target dataset. keep the test config!