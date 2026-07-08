import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from audio_utils import list_wavs, read_wav, fix_length, mix_at_snr, wav_to_targets


class OnTheFlyMixDataset(Dataset):
    def __init__(self, clean_dir, noise_dir, segment_seconds=2.0, sr=16000, snr_min=-5, snr_max=15, max_items=None):
        self.clean_files = list_wavs(clean_dir)
        self.noise_files = list_wavs(noise_dir)
        if max_items:
            self.clean_files = self.clean_files[:max_items]
        assert self.clean_files, f'No wav files in clean_dir={clean_dir}'
        assert self.noise_files, f'No wav files in noise_dir={noise_dir}'
        self.sr = sr
        self.seg_len = int(segment_seconds * sr)
        self.snr_min, self.snr_max = snr_min, snr_max

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean = fix_length(read_wav(self.clean_files[idx], self.sr), self.seg_len)
        noise = fix_length(read_wav(random.choice(self.noise_files), self.sr), self.seg_len)
        snr = random.uniform(self.snr_min, self.snr_max)
        noisy, clean = mix_at_snr(clean, noise, snr)
        return wav_to_targets(torch.tensor(noisy), torch.tensor(clean))


class PairedSpeechDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, segment_seconds=2.0, sr=16000, max_items=None):
        clean_files = list_wavs(clean_dir)
        noisy_files = list_wavs(noisy_dir)
        clean_map = {p.name: p for p in clean_files}
        pairs = []
        for n in noisy_files:
            if n.name in clean_map:
                pairs.append((clean_map[n.name], n))
        if not pairs:
            # fallback: pair by sorted order
            pairs = list(zip(clean_files, noisy_files))
        if max_items:
            pairs = pairs[:max_items]
        assert pairs, f'No clean/noisy wav pairs found: {clean_dir}, {noisy_dir}'
        self.pairs = pairs
        self.sr = sr
        self.seg_len = int(segment_seconds * sr)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        clean_path, noisy_path = self.pairs[idx]
        clean = fix_length(read_wav(clean_path, self.sr), self.seg_len)
        noisy = fix_length(read_wav(noisy_path, self.sr), self.seg_len)
        return wav_to_targets(torch.tensor(noisy), torch.tensor(clean))
