from argparse import Namespace
import torch
import numpy as np
import pickle, os, logging, librosa
from typing import Dict, List

from meldataset import mel_spectrogram, spectrogram

def Feature_Stack(features, max_length: int= None):
    max_feature_length = max_length or max([feature.shape[0] for feature in features])
    features = np.stack(
        [np.pad(feature, [[0, max_feature_length - feature.shape[0]], [0, 0]], constant_values= -10.0) for feature in features],
        axis= 0
        )
    return features

def Audio_Stack(audios):
    max_audio_length = max([audio.shape[0] for audio in audios])
    audios = np.stack(
        [np.pad(audio, [0, max_audio_length - audio.shape[0]]) for audio in audios],
        axis= 0
        )
    return audios

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pattern_path: str,
        metadata_file: str,
        feature_length_min: int,
        feature_length_max: int,
        accumulated_dataset_epoch: int= 1,
        augmentation_ratio: float= 0.0
        ):
        super().__init__()
        self.pattern_path = pattern_path
        
        metadata_dict = pickle.load(open(
            os.path.join(pattern_path, metadata_file).replace('\\', '/'), 'rb'
            ))
        
        self.patterns = []
        max_pattern_by_speaker = max([
            len(patterns)
            for patterns in metadata_dict['File_List_by_Speaker_Dict'].values()
            ])
        for patterns in metadata_dict['File_List_by_Speaker_Dict'].values():
            ratio = float(len(patterns)) / float(max_pattern_by_speaker)
            if ratio < augmentation_ratio:
                patterns *= int(np.ceil(augmentation_ratio / ratio))
            self.patterns.extend(patterns)

        self.patterns = [
            x for x in self.patterns
            if all([
                metadata_dict['Mel_Length_Dict'][x] >= feature_length_min,
                metadata_dict['Mel_Length_Dict'][x] <= feature_length_max
                ])
            ] * accumulated_dataset_epoch

    def __getitem__(self, idx):
        path = os.path.join(self.pattern_path, self.patterns[idx]).replace('\\', '/')
        pattern_dict = pickle.load(open(path, 'rb'))
        
        return pattern_dict['Mel'], pattern_dict['Audio']

    def __len__(self):
        return len(self.patterns)

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        hyper_parameters: Namespace,
        sources: List[str],
        ):
        super().__init__()
        self.hp = hyper_parameters
        
        self.patterns = []
        for index, source in enumerate(sources):
            if not os.path.exists(source):
                logging.warning('The source path of index {} is incorrect. This index is ignoired.'.format(index))
                continue

            self.patterns.append(source)

    def __getitem__(self, idx):
        source = self.patterns[idx]        
        audio, _ = librosa.load(source, sr= self.hp.Sound.Sample_Rate)
        audio = librosa.util.normalize(audio) * 0.95
        
        feature = mel_spectrogram(
            y= torch.from_numpy(audio).float().unsqueeze(0),
            n_fft= self.hp.Sound.N_FFT,
            num_mels= self.hp.Sound.Mel_Dim,
            sampling_rate= self.hp.Sound.Sample_Rate,
            hop_size= self.hp.Sound.Frame_Shift,
            win_size= self.hp.Sound.Frame_Length,
            fmin= self.hp.Sound.Mel_F_Min,
            fmax= self.hp.Sound.Mel_F_Max,
            ).squeeze(0).T.numpy()
        
        return feature, source

    def __len__(self):
        return len(self.patterns)

    def __len__(self):
        return len(self.patterns)


class Collater:
    def __init__(
        self,
        pattern_length: int,
        hop_length: int
        ):
        self.pattern_length = pattern_length
        self.hop_length = hop_length

    def __call__(self, batch):
        features, audios  = zip(*batch)

        feature_list, audio_list = [], []
        for feature, audio in zip(features, audios):
            feature_pad = max(0, self.pattern_length - feature.shape[0])
            audio_pad = max(0, self.pattern_length * self.hop_length - audio.shape[0])

            feature = np.pad(
                feature,
                [[int(np.floor(feature_pad / 2)), int(np.ceil(feature_pad / 2))], [0, 0]],
                mode= 'reflect'
                )
            audio = np.pad(
                audio,
                [int(np.floor(audio_pad / 2)), int(np.ceil(audio_pad / 2))],
                mode= 'reflect'
                )
            feature_list.append(feature)
            audio_list.append(audio)

        offsets = [
            np.random.randint(0, feature.shape[0] - self.pattern_length + 1)
            for feature in feature_list
            ]
            
        features = Feature_Stack([
            feature[offset:offset+self.pattern_length]
            for feature, offset in zip(feature_list, offsets)
            ])
        audios = Audio_Stack([
            audio[offset * self.hop_length:(offset+self.pattern_length) * self.hop_length]
            for audio, offset in zip(audio_list, offsets)
            ])

        features = torch.FloatTensor(features).permute(0, 2, 1)   # [Batch, Feature_d, Featpure_t]
        audios = torch.FloatTensor(audios)  # [Batch, Audio_t]

        return features, audios

class Inference_Collater:
    def __call__(self, batch):
        features, sources = zip(*batch)
        
        feature_lengths = np.array([feature.shape[0] for feature in features])

        features = Feature_Stack(features)
        
        features = torch.FloatTensor(features).permute(0, 2, 1)   # [Batch, Feature_d, Feature_t]
        feature_lengths = torch.LongTensor(feature_lengths) # [Batch]
        
        return features, feature_lengths, sources