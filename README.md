# DiffWave

* [Kong, Z., Ping, W., Huang, J., Zhao, K., & Catanzaro, B. (2020). Diffwave: A versatile diffusion model for audio synthesis. arXiv preprint arXiv:2009.09761.](https://arxiv.org/abs/2009.09761)


* Some `Swish` activation function is changed to `Mish`.

# Used datasets

| Dataset   | Language | License                                | Dataset address                                                                                      
|-----------|----------|----------------------------------------|-----------------------------------------------------------------------------------------------------|
| Emotion   | Korean   | CC-BY-NC-SA-4.0                        | https://github.com/emotiontts/emotiontts_open_db                                                    |
| KSS       | Korean   | CC-BY-NC-SA-4.0                        | https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset                      |
| AIHub     | Korean   | Conditionally commercial use available | https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=542 |
| LJSpeech  | English  | Public Domain                          | https://keithito.com/LJ-Speech-Dataset/                                                             | 
| VCTK      | English  | CC-BY-4.0                              | https://datashare.is.ed.ac.uk/handle/10283/2651                                                     | 
| LibriTTS  | English  | CC-BY-4.0                              | https://openslr.org/60/                                                                             | 

# Generate patterns
```
python Pattern_Generate.py [parameters]
```

## Parameters

* Main parameters are the paths of dataset.
    * -emo <path>
    * -kss <path>
    * -aihub <path>
    * -lj <path>
    * -vctk <path>    
    * -libri <path>
* -evalr
    * Set the evaluation pattern ratio.
    * Default is `0.001`.
* -evalm
    * Set the evaluation pattern minimum of each speaker.
    * Default is `1`.
* -mw
    * The number of threads used to create the pattern
    * Default is `2`.

# Train
## Single GPU
```
python Train.py [parameters]
```

* -hp <path>
    * The hyper parameter's path.
* -s <int>
    * The resume step parameter.
    * Default is 0.
    * When this parameter is 0, model try to find the latest checkpoint in checkpoint path.

## Multi GPU example
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=32 python -m torch.distributed.launch --nproc_per_node=8 --master_port=54321 Train.py --hyper_parameters Hyper_Parameters.yaml
```

* Before run, please modify the `Use_Multi_GPU` and `Device` in [hyper parameters](./Hyper_Parameters.yaml)

# Inference
* Please refer [Inference.py](./Inference.py) and [Inference.ipynb](./Inference.ipynb)

# Checkpoint
|Main parameters                 | Hyper parameter                        | Checkpoint                     |
|--------------------------------|----------------------------------------|--------------------------------|
|64 diffusion size and 20 stacks | [Here](./Exp/Hyper_Parameters_64.yaml) | [Here](./Exp/Checkpoint_64.pt) |
