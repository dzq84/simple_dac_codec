import os
import random
import torch
import torchaudio
import yaml
import argparse
from torch.utils.data import Dataset, DataLoader
from model.codec import CODEC
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

class AudioDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, segment_length=1):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.audio_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.wav') or filename.endswith('.mp3'):
                    self.audio_files.append(os.path.join(dirpath, filename))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        audio, sr = torchaudio.load(audio_path, normalize=True)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            audio = resampler(audio)

        num_samples = self.segment_length * self.sample_rate
        if audio.size(1) >= num_samples:
            start = random.randint(0, audio.size(1) - num_samples)
            audio = audio[:, start:start + num_samples]
        else:
            pad = (0, num_samples - audio.size(1))
            audio = torch.nn.functional.pad(audio, pad)

        return audio


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CODEC model using PyTorch Lightning')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    args = parser.parse_args()

    # Load the configuration from the YAML file
    config = load_config(args.config)

    # Create datasets
    traindataset = AudioDataset(root_dir=config['train_dir'], 
                                sample_rate=config['sample_rate'], 
                                segment_length=config['segment_length_train'])

    testdataset = AudioDataset(root_dir=config['test_dir'], 
                               sample_rate=config['sample_rate'], 
                               segment_length=config['segment_length_test'])

    # Create dataloaders
    traindataloader = DataLoader(traindataset, 
                                 batch_size=config['batch_size'], 
                                 shuffle=True, 
                                 num_workers=config['num_workers'])

    testdataloader = DataLoader(testdataset, 
                                batch_size=config['batch_size'], 
                                shuffle=True, 
                                num_workers=config['num_workers'])

    # Initialize model and logger
    codec = CODEC()
    logger = TensorBoardLogger(save_dir=config['log_dir'], name=config['log_name'])

    # Initialize the trainer
    trainer = Trainer(accelerator='gpu', devices=config['devices'], logger=logger)

    # Start training
    trainer.fit(codec, traindataloader)
