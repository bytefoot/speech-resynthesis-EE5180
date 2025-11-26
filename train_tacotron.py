import argparse
import json
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

# Ensure this path matches your file structure
from textless.vocoders.hifigan.model import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator

class UnitAudioDataset(Dataset):
    def __init__(self, json_path, segment_size=8192):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.segment_size = segment_size
        self.hop_length = 320 # 16000Hz / 50Hz = 320 hop

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        wav, sr = torchaudio.load(item['audio_path'])
        if wav.size(0) > 1: wav = wav.mean(dim=0, keepdim=True)
        
        units = torch.LongTensor(item['units'])
        f0 = torch.FloatTensor(item['f0'])

        # Random Slicing
        if wav.size(1) > self.segment_size:
            max_audio_start = wav.size(1) - self.segment_size
            audio_start = torch.randint(0, max_audio_start, (1,)).item()
            
            # Align Unit Start
            unit_start = audio_start // self.hop_length
            unit_len = self.segment_size // self.hop_length
            
            # Boundary Check
            unit_start = min(unit_start, len(units) - unit_len)
            
            # Re-align audio to exact unit boundary
            audio_start = unit_start * self.hop_length

            wav = wav[:, audio_start : audio_start + self.segment_size]
            units = units[unit_start : unit_start + unit_len]
            f0 = f0[unit_start : unit_start + unit_len]
        else:
            # Pad
            wav = F.pad(wav, (0, self.segment_size - wav.size(1)))
            pad_len = (self.segment_size // self.hop_length) - len(units)
            if pad_len > 0:
                units = F.pad(units, (0, pad_len))
                f0 = F.pad(f0, (0, pad_len))

        return units, f0, wav

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Init Models (No num_speakers arg)
    generator = Generator(vocab_size=args.vocab_size).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    optim_g = AdamW(generator.parameters(), lr=args.lr, betas=(0.8, 0.99))
    optim_d = AdamW(list(mpd.parameters()) + list(msd.parameters()), lr=args.lr, betas=(0.8, 0.99))
    
    scheduler_g = ExponentialLR(optim_g, gamma=0.999)
    scheduler_d = ExponentialLR(optim_d, gamma=0.999)

    train_ds = UnitAudioDataset(args.train_json)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    global_step = 0
    
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint)
        generator.load_state_dict(ckpt['generator'])
        print(f"Resumed from {args.resume_checkpoint}")

    print("Starting Training...")
    for epoch in range(args.epochs):
        generator.train()
        for batch_idx, (units, f0, wav) in enumerate(train_dl):
            units, f0, wav = units.to(device), f0.to(device), wav.to(device)
            
            # --- Discriminator Step ---
            optim_d.zero_grad()
            
            # Generator Forward (Units + F0 only)
            y_g_hat = generator(units, f0) 

            # MPD & MSD Loss
            y_df_hat_r, y_df_hat_g, _, _ = mpd(wav, y_g_hat.detach())
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(wav, y_g_hat.detach())
            
            loss_d = 0
            for dr, dg in zip(y_df_hat_r + y_ds_hat_r, y_df_hat_g + y_ds_hat_g):
                loss_d += torch.mean((1 - dr)**2) + torch.mean(dg**2)
            
            loss_d.backward()
            optim_d.step()

            # --- Generator Step ---
            optim_g.zero_grad()
            
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(wav, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(wav, y_g_hat)
            
            loss_fm = 0
            for dr, dg in zip(fmap_f_r + fmap_s_r, fmap_f_g + fmap_s_g):
                for rl, gl in zip(dr, dg):
                    loss_fm += torch.mean(torch.abs(rl - gl))
            
            loss_gen = 0
            for dg in y_df_hat_g + y_ds_hat_g:
                loss_gen += torch.mean((1 - dg)**2)
                
            loss_g_total = loss_gen + 2.0 * loss_fm
            
            loss_g_total.backward()
            optim_g.step()

            if global_step % args.log_interval == 0:
                print(f"Epoch {epoch} | Step {global_step} | D: {loss_d.item():.4f} | G: {loss_g_total.item():.4f}")
            
            global_step += 1

        scheduler_g.step()
        scheduler_d.step()

        if (epoch + 1) % args.save_interval == 0:
            torch.save({'generator': generator.state_dict()}, os.path.join(args.checkpoint_dir, f"ckpt_{epoch+1}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--vocab_size", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train(args)