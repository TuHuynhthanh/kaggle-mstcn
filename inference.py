import argparse
from pathlib import Path
import torch
import soundfile as sf
from model import MSTCNSE
from audio_utils import read_wav, wav_to_targets, lps_irm_to_wave


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--noisy_wav', required=True)
    p.add_argument('--out_wav', default='/kaggle/working/enhanced.wav')
    args = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MSTCNSE().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    noisy = read_wav(args.noisy_wav)
    dummy_clean = noisy.copy()
    noisy_lps, _, _ = wav_to_targets(torch.tensor(noisy), torch.tensor(dummy_clean))
    with torch.no_grad():
        pred_lps, pred_irm = model(noisy_lps.unsqueeze(0).to(device))
    enhanced = lps_irm_to_wave(noisy, pred_lps[0].cpu(), pred_irm[0].cpu())
    sf.write(args.out_wav, enhanced, 16000)
    print('Saved:', args.out_wav)


if __name__ == '__main__':
    main()
