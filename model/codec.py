import math
from typing import List
from typing import Union

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import librosa

from nn.layers import Snake1d
from nn.layers import WNConv1d
from nn.layers import WNConvTranspose1d
from nn.quantize import ResidualVectorQuantize
import nn.loss as closs
from utils.tools import audio_to_mel

import lightning as L
from model.discriminator import Discriminator

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class CODEC(L.LightningModule):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 16000,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
        )

        self.generator= nn.ModuleList([
            self.encoder,
            self.quantizer,
            self.decoder
        ])
        self.discriminator = Discriminator()
        self.sample_rate = sample_rate
        self.apply(init_weights)

        self.automatic_optimization = False  

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data
    
    def training_step(self, batch, batch_idx):

        opt_g, opt_d = self.optimizers()

        audio_data = batch
        audio_data = self.preprocess(audio_data, self.sample_rate)

        z = self.encoder(audio_data)
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(z)
        
        audio_data_hat = self.decoder(z)

        d_real = self.discriminator(audio_data)
        d_fake = self.discriminator(audio_data_hat.detach())

        d_loss = closs.discriminator_loss(d_fake,d_real)
        loss_d = d_loss

        opt_d.zero_grad()
        self.manual_backward(loss_d)    
        opt_d.step()

        d_real = self.discriminator(audio_data)
        d_fake = self.discriminator(audio_data_hat)

        mel_loss = closs.mel_spectrogram_loss(audio_data,audio_data_hat)
        g_loss , f_loss = closs.generator_loss(d_fake,d_real)
        loss_g = 0.25 * commitment_loss + codebook_loss + 15 * mel_loss + g_loss + 2 * f_loss

        opt_g.zero_grad()
        self.manual_backward(loss_g)        
        opt_g.step()

        self.log('train/loss_d', loss_d, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/loss_g', loss_g, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/mel_loss', mel_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/commitment_loss', commitment_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/codebook_loss', codebook_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/f_loss', f_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx % 100 == 0:
            self.logger.experiment.add_audio('train/original_audio', audio_data[0], self.global_step, self.sample_rate)
            self.logger.experiment.add_audio('train/generated_audio', audio_data_hat[0], self.global_step, self.sample_rate)
            
            mel_original = audio_to_mel(audio_data[0].unsqueeze(0), self.sample_rate)
            mel_generated = audio_to_mel(audio_data_hat[0].unsqueeze(0), self.sample_rate)
            
            fig, axes = plt.subplots(1, 2, figsize=(20, 4))
            mel_original_db = mel_original.squeeze().detach().cpu().numpy()
            mel_generated_db = mel_generated.squeeze().detach().cpu().numpy()

            img = librosa.display.specshow(mel_original_db, sr=self.sample_rate, hop_length=512, x_axis='time', y_axis='mel', ax=axes[0])
            axes[0].set_title('Original Audio Mel-Spectrogram')
            fig.colorbar(img, ax=axes[0], format='%+2.0f dB')

            img = librosa.display.specshow(mel_generated_db, sr=self.sample_rate, hop_length=512, x_axis='time', y_axis='mel', ax=axes[1])
            axes[1].set_title('Generated Audio Mel-Spectrogram')
            fig.colorbar(img, ax=axes[1], format='%+2.0f dB')

            self.logger.experiment.add_figure('train/mel_spectrograms', fig, self.global_step)
            plt.close(fig)

    def configure_optimizers(self):
        
        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=0.0001, betas=(0.8, 0.99))
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=0.0001, betas=(0.8, 0.99))

        sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=0.999996)
        sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=0.999996)

        return [opt_g, opt_d], [{'scheduler': sch_g, 'interval': 'step'}, {'scheduler': sch_d, 'interval': 'step'}]