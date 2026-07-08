import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import MSTCNSE, count_parameters
from dataset import OnTheFlyMixDataset, PairedSpeechDataset


def make_dataset(args, split):
    if args.mode == 'paired':
        clean = args.clean_train if split == 'train' else args.clean_val
        noisy = args.noisy_train if split == 'train' else args.noisy_val
        return PairedSpeechDataset(clean, noisy, args.segment_seconds, max_items=args.max_items)
    clean = args.clean_train if split == 'train' else args.clean_val
    noise = args.noise_train if split == 'train' else args.noise_val
    return OnTheFlyMixDataset(clean, noise, args.segment_seconds, max_items=args.max_items)


def run_epoch(model, loader, opt, device, train=True):
    model.train(train)
    total = 0.0
    for noisy_lps, clean_lps, irm in tqdm(loader, leave=False):
        noisy_lps = noisy_lps.to(device)
        clean_lps = clean_lps.to(device)
        irm = irm.to(device)
        with torch.set_grad_enabled(train):
            pred_lps, pred_irm = model(noisy_lps)
            loss = F.mse_loss(pred_lps, clean_lps) + F.mse_loss(pred_irm, irm)
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
        total += loss.item() * noisy_lps.size(0)
    return total / len(loader.dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['mix','paired'], default='mix')
    p.add_argument('--clean_train', required=True)
    p.add_argument('--clean_val', required=True)
    p.add_argument('--noise_train')
    p.add_argument('--noise_val')
    p.add_argument('--noisy_train')
    p.add_argument('--noisy_val')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--segment_seconds', type=float, default=2.0)
    p.add_argument('--max_items', type=int, default=None)
    p.add_argument('--out', default='/kaggle/working/runs/mstcn_se2')
    p.add_argument('--no_multiscale', action='store_true')
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_set = make_dataset(args, 'train')
    val_set = make_dataset(args, 'val')
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    model = MSTCNSE(use_multiscale=not args.no_multiscale).to(device)
    print('Device:', device)
    print('Trainable parameters:', count_parameters(model))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = 1e9
    history = []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, opt, device, True)
        va = run_epoch(model, val_loader, opt, device, False)
        print(f'Epoch {epoch:03d}: train_loss={tr:.6f} val_loss={va:.6f}')
        history.append((epoch, tr, va))
        torch.save({'model': model.state_dict(), 'args': vars(args)}, out / 'last.pt')
        if va < best:
            best = va
            torch.save({'model': model.state_dict(), 'args': vars(args)}, out / 'best.pt')
    with open(out / 'history.csv', 'w') as f:
        f.write('epoch,train_loss,val_loss\n')
        for e, tr, va in history:
            f.write(f'{e},{tr},{va}\n')


if __name__ == '__main__':
    main()
