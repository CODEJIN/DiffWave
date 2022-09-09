import torch
import logging, yaml, sys, pickle
import matplotlib.pyplot as plt
from typing import List

from Modules.Diffusion import Diffusion
from Datasets import Inference_Dataset as Dataset, Inference_Collater as Collater
from Arg_Parser import Recursive_Parse

import matplotlib as mpl
# 유니코드 깨짐현상 해결
mpl.rcParams['axes.unicode_minus'] = False
# 나눔고딕 폰트 적용
plt.rcParams["font.family"] = 'NanumGothic'

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )

class Inferencer:
    def __init__(
        self,
        hp_path: str,
        checkpoint_path: str,        
        batch_size= 1
        ):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        self.hp = Recursive_Parse(yaml.load(
            open(hp_path, encoding='utf-8'),
            Loader=yaml.Loader
            ))

        self.model = Diffusion(self.hp).to(self.device)

        self.Load_Checkpoint(checkpoint_path)
        self.batch_size = batch_size

    def Load_Checkpoint(self, path: str):
        state_dict = torch.load(path, map_location= 'cpu')
        self.model.load_state_dict(state_dict['Model'])        
        self.steps = state_dict['Steps']

        self.model.eval()

        logging.info('Checkpoint loaded at {} steps.'.format(self.steps))

    @torch.no_grad()
    def Inference(self, features: torch.Tensor, feature_lengths: List[int]):
        features = features.to(self.device, non_blocking=True)
        
        audios, *_ = self.model(
            conditions= features
            )
        audios = audios.clamp(-1.0, 1.0)

        audios = [
            audio[:length * self.hp.Sound.Frame_Shift].cpu().numpy()
            for audio, length in zip(audios, feature_lengths)
            ]

        return audios