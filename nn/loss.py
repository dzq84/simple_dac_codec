import torch
import torch.nn.functional as F
from typing import List
from utils.tools import batch_mel,batch_stft

def L1Loss(x,y):
    loss = F.l1_loss(x, y)
    return loss

def multi_scale_stft_loss(x: torch.Tensor, 
                          y: torch.Tensor, 
                          window_lengths: List[int] = [2048, 512], 
                          clamp_eps: float = 1e-5, 
                          mag_weight: float = 1.0, 
                          log_weight: float = 1.0, 
                          pow: float = 2.0) -> torch.Tensor:
    loss = 0.0
    for w in window_lengths: 
        x_stft = batch_stft(x,w)
        y_stft = batch_stft(y,w)
        x_magnitude = torch.abs(x_stft)
        y_magnitude = torch.abs(y_stft)
        log_magnitude_loss = F.l1_loss(
            x_magnitude.clamp(min=clamp_eps).pow(pow).log10(),
            y_magnitude.clamp(min=clamp_eps).pow(pow).log10(),
        )
        magnitude_loss = F.l1_loss(x_magnitude, y_magnitude)
        loss += log_weight * log_magnitude_loss + mag_weight * magnitude_loss
        
    return loss
    
def mel_spectrogram_loss(
    x: torch.Tensor, y: torch.Tensor, 
    sample_rate: int = 16000,
    n_mels: List[int] = [5, 10, 20, 40, 80, 160, 320], 
    window_lengths: List[int] = [32, 64, 128, 256, 512, 1024, 2048],
    clamp_eps: float = 1e-5, 
    mag_weight: float = 0.0, 
    log_weight: float = 1.0, 
    pow: float = 1.0, 
    mel_fmin: List[float] = [0, 0, 0, 0, 0, 0, 0], 
    mel_fmax: List[float] = None,):

    if mel_fmax is None:
        mel_fmax = [sample_rate / 2] * len(n_mels)
    
    loss = 0.0
    for n_mel, win_len, fmin, fmax in zip(n_mels, window_lengths, mel_fmin, mel_fmax):
        
        x_mel = batch_mel(x,sample_rate=sample_rate,window_length=win_len,n_mels=n_mel,f_min=fmin,f_max=fmax)
        y_mel = batch_mel(y,sample_rate=sample_rate,window_length=win_len,n_mels=n_mel,f_min=fmin,f_max=fmax)

        log_loss = F.l1_loss(
            x_mel.clamp(min=clamp_eps).pow(pow).log10(),
            y_mel.clamp(min=clamp_eps).pow(pow).log10(),
        )

        mag_loss = F.l1_loss(
            x_mel, y_mel
        )

        loss += log_weight * log_loss + mag_weight * mag_loss
    
    return loss
    
def discriminator_loss(d_fake, d_real):

    d_fake = d_fake
    
    loss_d = 0
    for x_fake, x_real in zip(d_fake, d_real):
        loss_d += torch.mean(x_fake[-1] ** 2)
        loss_d += torch.mean((1 - x_real[-1]) ** 2)
    
    return loss_d

def generator_loss(d_fake, d_real):
    
    loss_g = 0
    for x_fake in d_fake:
        loss_g += torch.mean((1 - x_fake[-1]) ** 2)
    
    loss_feature = 0
    for i in range(len(d_fake)):
        for j in range(len(d_fake[i]) - 1):
            loss_feature += L1Loss(d_fake[i][j], d_real[i][j].detach())
    
    return loss_g, loss_feature
