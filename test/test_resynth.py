import argparse
import torch
import torchaudio
from textless.data.speech_encoder import SpeechEncoder
from textless.vocoders.hifigan.model import Generator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense_model", type=str, default="hubert-base-ls960")
    parser.add_argument("--vocab_size", type=int, default=100)
    parser.add_argument("--input", required=True, help="Input wav file")
    parser.add_argument("--output", required=True, help="Output wav file")
    parser.add_argument("--checkpoint", required=True, help="Path to HiFiGAN checkpoint")
    return parser.parse_args()

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Init Encoder
    print("Loading SpeechEncoder...")
    encoder = SpeechEncoder.by_name(
        dense_model_name=args.dense_model,
        quantizer_model_name="kmeans",
        vocab_size=args.vocab_size,
        need_f0=True,
        deduplicate=False,
    ).to(device)

    # 2. Init Generator (No num_speakers)
    print("Loading Generator...")
    vocoder = Generator(vocab_size=args.vocab_size).to(device)
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'generator' in ckpt:
        vocoder.load_state_dict(ckpt['generator'])
    else:
        vocoder.load_state_dict(ckpt)
    
    vocoder.eval()
    vocoder.remove_weight_norm()

    # 3. Process Input
    print(f"Processing {args.input}...")
    wav, sr = torchaudio.load(args.input)
    if wav.ndim == 2: wav = wav.mean(dim=0, keepdim=True)
    
    wav = encoder.maybe_resample(wav, sr).to(device)

    # 4. Infer
    with torch.no_grad():
        encoded = encoder(wav)
        units = encoded["units"].unsqueeze(0) # [1, T]
        f0 = encoded["f0"].unsqueeze(0)       # [1, T]

        # Forward pass (Units + F0 only)
        audio = vocoder(units, f0)

    # 5. Save
    print(f"Saving to {args.output}")
    torchaudio.save(args.output, audio.cpu(), encoder.expected_sample_rate)

if __name__ == "__main__":
    main(get_args())