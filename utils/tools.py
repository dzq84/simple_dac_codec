import torch
import torchaudio
import torchaudio.transforms as T

def audio_to_mel(audio, sample_rate):
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    ).to(audio.device)
    mel = mel_spectrogram(audio)
    # 将Power Spectrogram转为DB单位
    mel_db = T.AmplitudeToDB()(mel)
    return mel_db

def batch_stft(audio_data, window_length=None):

    window = torch.hann_window(window_length) 
    batch_size, num_channels, _ = audio_data.shape

    # STFT
    stft_data = torch.stft(
        audio_data.reshape(-1, audio_data.size(-1)), # Flatten batch and channel
        n_fft=window_length,
        hop_length=window_length//4,
        win_length=window_length,
        window=window.to(audio_data.device),
        normalized=True, 
        onesided=True,
        return_complex=True,
        center=False
    )

    # Reshape back to (B, C, F, T) format
    _, freq_bins, time_frames = stft_data.shape
    stft_data = stft_data.view(batch_size, num_channels, freq_bins, time_frames)
    
    return stft_data

def batch_mel(audio_data, sample_rate=44100, window_length=None, n_mels=128, f_min=0.0, f_max=None):

    window = torch.hann_window(window_length).to(audio_data.device)
    batch_size, num_channels, _ = audio_data.shape

    # Compute STFT
    stft_data = torch.stft(
        audio_data.reshape(-1, audio_data.size(-1)),  # Flatten batch and channel
        n_fft=window_length,
        hop_length=window_length // 4,
        win_length=window_length,
        window=window,
        normalized=True,
        onesided=True,
        return_complex=True,
        center=False
    )

    # Convert STFT to power spectrogram
    power_spec = stft_data.abs() ** 2

    # Create Mel filter
    mel_scale = torchaudio.transforms.MelScale(n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max, n_stft=window_length//2 + 1,norm='slaney').to(audio_data.device)
    
    # Apply Mel filter
    mel_spec = mel_scale(power_spec)

    # Reshape back to (B, C, F, T) format
    _, n_mels, time_frames = mel_spec.shape
    mel_spec = mel_spec.view(batch_size, num_channels, n_mels, time_frames)
    
    return mel_spec