"""
Zero-Shot Speaker Embedding Extraction
Extract speaker embeddings from any audio file for voice cloning
"""
import sys
import os
import json
import torch
import numpy as np
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from speaker.models.lstm import LSTMSpeakerEncoder
from speaker.config import SpeakerEncoderConfig
from speaker.utils.audio import AudioProcessor


def load_config(config_path):
    """Load configuration from JSON file (with comment support)"""
    import re
    with open(config_path, 'r', encoding='utf-8') as f:
        # Read the file and remove comments
        content = f.read()
        # Remove single-line comments (// ...)
        content = re.sub(r'//.*\n', '\n', content)
        # Remove multi-line comments (/* ... */)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        # Parse the cleaned JSON
        config_dict = json.loads(content)
    return config_dict


def extract_speaker_embedding(
    audio_path,
    model_path='speaker_pretrain/best_model.pth.tar',
    config_path='speaker_pretrain/config.json',
    output_path=None,
    use_cuda=True
):
    """
    Extract speaker embedding from audio file
    
    Args:
        audio_path: Path to input audio file
        model_path: Path to pretrained speaker encoder model
        config_path: Path to model configuration
        output_path: Path to save embedding (optional, if None will return embedding)
        use_cuda: Whether to use GPU
        
    Returns:
        embedding: 256-dim numpy array representing speaker identity
    """
    
    # Load configuration
    print(f"Loading config from: {config_path}")
    config_dict = load_config(config_path)
    config = SpeakerEncoderConfig(config_dict)
    config.from_dict(config_dict)
    
    # Initialize model
    print(f"Initializing speaker encoder model...")
    speaker_encoder = LSTMSpeakerEncoder(
        config.model_params["input_dim"],
        config.model_params["proj_dim"],
        config.model_params["lstm_dim"],
        config.model_params["num_lstm_layers"],
    )
    
    # Load pretrained weights
    print(f"Loading pretrained weights from: {model_path}")
    speaker_encoder.load_checkpoint(model_path, eval=True, use_cuda=use_cuda)
    
    # Initialize audio processor
    print(f"Initializing audio processor...")
    audio_processor = AudioProcessor(**config.audio)
    # Enable audio preprocessing for better embeddings
    audio_processor.do_sound_norm = True
    audio_processor.do_trim_silence = True
    
    # Load and process audio
    print(f"Loading audio from: {audio_path}")
    waveform = audio_processor.load_wav(
        audio_path, 
        sr=audio_processor.sample_rate
    )
    
    # Extract mel-spectrogram
    print(f"Extracting mel-spectrogram...")
    mel_spec = audio_processor.melspectrogram(waveform)
    spec_tensor = torch.from_numpy(mel_spec.T)
    
    if use_cuda:
        spec_tensor = spec_tensor.cuda()
    spec_tensor = spec_tensor.unsqueeze(0)
    
    # Extract embedding
    print(f"Extracting speaker embedding...")
    with torch.no_grad():
        embedding = speaker_encoder.compute_embedding(spec_tensor)
    
    # Convert to numpy
    embedding = embedding.detach().cpu().numpy().squeeze()
    
    print(f"✓ Extracted {embedding.shape[0]}-dimensional embedding")
    print(f"  Embedding stats: min={embedding.min():.4f}, max={embedding.max():.4f}, norm={np.linalg.norm(embedding):.4f}")
    
    # Save if output path provided
    if output_path:
        np.save(output_path, embedding, allow_pickle=False)
        print(f"✓ Saved embedding to: {output_path}")
    
    return embedding


def compare_speakers(embedding1, embedding2):
    """
    Compare similarity between two speaker embeddings using cosine similarity
    
    Args:
        embedding1, embedding2: 256-dim numpy arrays
        
    Returns:
        similarity: Cosine similarity score (0-1, higher = more similar)
    """
    # Normalize embeddings
    emb1_norm = embedding1 / np.linalg.norm(embedding1)
    emb2_norm = embedding2 / np.linalg.norm(embedding2)
    
    # Compute cosine similarity
    similarity = np.dot(emb1_norm, emb2_norm)
    return similarity


def mix_speakers(embeddings, weights=None):
    """
    Mix multiple speaker embeddings with optional weights
    
    Args:
        embeddings: List of 256-dim numpy arrays
        weights: List of weights (optional, defaults to equal weights)
        
    Returns:
        mixed_embedding: Weighted average of embeddings
    """
    embeddings = np.array(embeddings)
    
    if weights is None:
        weights = np.ones(len(embeddings)) / len(embeddings)
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights
    
    # Weighted average
    mixed = np.sum(embeddings * weights[:, np.newaxis], axis=0)
    
    return mixed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract speaker embeddings for zero-shot voice cloning",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        'audio_path',
        type=str,
        help='Path to input audio file (WAV format recommended)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Path to save speaker embedding (.npy file)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='speaker_pretrain/best_model.pth.tar',
        help='Path to pretrained speaker encoder model'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='speaker_pretrain/config.json',
        help='Path to model configuration'
    )
    
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Use CPU instead of GPU'
    )
    
    parser.add_argument(
        '--compare',
        type=str,
        help='Compare with another speaker embedding file (.npy)'
    )
    
    args = parser.parse_args()
    
    # Extract embedding
    print("\n" + "="*60)
    print("ZERO-SHOT SPEAKER EMBEDDING EXTRACTION")
    print("="*60 + "\n")
    
    embedding = extract_speaker_embedding(
        audio_path=args.audio_path,
        model_path=args.model,
        config_path=args.config,
        output_path=args.output,
        use_cuda=not args.cpu
    )
    
    # Optional: Compare with another speaker
    if args.compare:
        print(f"\nComparing with: {args.compare}")
        other_embedding = np.load(args.compare)
        similarity = compare_speakers(embedding, other_embedding)
        print(f"Cosine similarity: {similarity:.4f}")
        
        if similarity > 0.9:
            print("→ Very similar speakers (likely same person)")
        elif similarity > 0.7:
            print("→ Similar speakers (might be same person)")
        elif similarity > 0.5:
            print("→ Moderately similar speakers")
        else:
            print("→ Different speakers")
    
    print("\n" + "="*60)
    print("DONE! Use this embedding for voice conversion:")
    print(f"  python svc_inference.py --spk {args.output} --wave input.wav ...")
    print("="*60 + "\n")
