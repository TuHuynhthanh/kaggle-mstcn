from pathlib import Path
import random
import numpy as np
import soundfile as sf
import torch


def list_wavs(root):
    root = Path(root)
    return sorted([p for p in root.rglob('*.wav')])


def read_wav(path, sr=16000):
    import librosa
    wav, file_sr = sf.read(str(path), dtype='float32')
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if file_sr != sr:
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    return wav.astype(np.float32)


def fix_length(wav, length):
    if len(wav) >= length:
        start = random.randint(0, len(wav) - length)
        return wav[start:start + length]
    return np.pad(wav, (0, length - len(wav)))


def mix_at_snr(clean, noise, snr_db):
    clean_power = np.mean(clean ** 2) + 1e-8
    noise_power = np.mean(noise ** 2) + 1e-8
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    noisy = clean + noise
    peak = max(1.0, np.max(np.abs(noisy)), np.max(np.abs(clean)))
    return (noisy / peak).astype(np.float32), (clean / peak).astype(np.float32)


def stft_torch(wav, n_fft=512, hop=256, win_length=512):
    window = torch.hann_window(win_length, device=wav.device)
    return torch.stft(wav, n_fft=n_fft, hop_length=hop, win_length=win_length,
                      window=window, return_complex=True)


def wav_to_targets(noisy, clean, eps=1e-8):
    # input tensors: [T]
    Y = stft_torch(noisy)
    X = stft_torch(clean)
    noisy_pow = (Y.abs() ** 2).clamp_min(eps)
    clean_pow = (X.abs() ** 2).clamp_min(eps)
    noisy_lps = torch.log(noisy_pow).transpose(0, 1)  # [frames,257]
    clean_lps = torch.log(clean_pow).transpose(0, 1)
    irm = (X.abs() / (Y.abs() + eps)).clamp(0.0, 1.0).transpose(0, 1)
    return noisy_lps, clean_lps, irm


def lps_irm_to_wave(noisy_wav, pred_lps, pred_irm, sr=16000):
    # pred_lps/pred_irm: [frames,257]
    wav = torch.tensor(noisy_wav, dtype=torch.float32)
    Y = stft_torch(wav)
    mag_lps = torch.exp(0.5 * pred_lps.transpose(0, 1))
    mag_irm = Y.abs() * pred_irm.transpose(0, 1)
    mag = 0.5 * (mag_lps + mag_irm)
    enhanced = mag * torch.exp(1j * torch.angle(Y))
    window = torch.hann_window(512)
    out = torch.istft(enhanced, n_fft=512, hop_length=256, win_length=512,
                      window=window, length=len(noisy_wav))
    return out.detach().cpu().numpy().astype(np.float32)
