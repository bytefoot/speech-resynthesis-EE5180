import argparse
import glob
import json
import os
import random
import time
import torch
import torchaudio
import math
import signal
import sys
from tqdm import tqdm
from torch.multiprocessing import Pool, set_start_method, Manager

# Note: Import SpeechEncoder inside worker to avoid pickling issues
from textless.data.speech_encoder import SpeechEncoder

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", type=str, required=True, help="Directory containing wav files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save json splits")
    parser.add_argument("--dense_model", type=str, default="hubert-base-ls960")
    parser.add_argument("--vocab_size", type=int, default=100)
    parser.add_argument("--split_ratio", type=str, default="0.9,0.05,0.05")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel processes")
    return parser.parse_args()

def worker_process(packed_args):
    """
    Worker process to handle a chunk of files.
    """
    gpu_id, files, args, worker_id = packed_args
    
    # Check if we should skip files that are already done? 
    # (Simplified approach: We process the chunk assigned. The filtering happens in main.)

    device = torch.device(f'cuda:{gpu_id}')
    
    try:
        # Load Model (Silence stdout during load to keep progress bar clean)
        # sys.stdout = open(os.devnull, 'w') 
        encoder = SpeechEncoder.by_name(
            dense_model_name=args.dense_model,
            quantizer_model_name="kmeans",
            vocab_size=args.vocab_size,
            need_f0=True,
            deduplicate=False,
        ).to(device)
        # sys.stdout = sys.__stdout__ # Restore
    except Exception as e:
        return [] # Fail gracefully if model load fails

    results = []
    
    # We use a simple loop instead of tqdm here to avoid conflict with main process signal handling
    # The main process tracks overall progress.
    for i, wav_path in enumerate(files):
        try:
            wav, sr = torchaudio.load(wav_path)
            if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
            
            wav = encoder.maybe_resample(wav, sr).to(device)
            
            with torch.no_grad():
                encoded = encoder(wav)
            
            units = encoded["units"].cpu().tolist()
            f0 = encoded["f0"].cpu().tolist()
            
            entry = {
                "audio_path": wav_path,
                "units": units,
                "f0": f0,
                "duration": wav.shape[1] / encoder.expected_sample_rate
            }
            results.append(entry)
            
        except Exception as e:
            # print(f"[Worker {worker_id}] Error on {wav_path}: {e}")
            continue
            
    return results

def save_checkpoint(data, output_dir):
    """Saves the current progress to a temporary file."""
    temp_path = os.path.join(output_dir, "processed_checkpoint.json")
    with open(temp_path, 'w') as f:
        json.dump(data, f)
    print(f"\n[Checkpointer] Progress saved! {len(data)} files processed. Resume by running the script again.")

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "processed_checkpoint.json")
    
    # 1. Load Previous Progress
    processed_data = []
    processed_files = set()
    
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}. Loading...")
        with open(checkpoint_path, 'r') as f:
            processed_data = json.load(f)
        processed_files = set(item['audio_path'] for item in processed_data)
        print(f"Resuming with {len(processed_data)} files already done.")

    # 2. Find Files
    print("Scanning directory...")
    all_wav_files = glob.glob(os.path.join(args.wav_dir, "**/*.wav"), recursive=True)
    
    # Filter out already processed files
    wav_files_to_do = [f for f in all_wav_files if f not in processed_files]
    
    if not wav_files_to_do:
        print("All files already processed! Proceeding to split generation.")
        total_data = processed_data
    else:
        print(f"Total files: {len(all_wav_files)}. Left to process: {len(wav_files_to_do)}")
        
        # Sort/Shuffle deterministically
        wav_files_to_do.sort()
        random.seed(42)
        random.shuffle(wav_files_to_do)

        # 3. Split into Chunks
        chunk_size = math.ceil(len(wav_files_to_do) / args.workers)
        chunks = [wav_files_to_do[i:i + chunk_size] for i in range(0, len(wav_files_to_do), chunk_size)]
        
        process_args = [(0, chunk, args, i) for i, chunk in enumerate(chunks)]

        # 4. Run with Ctrl+C Handling
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        print("Starting workers... (Press Ctrl+C to pause/save)")
        
        results_new = []
        try:
            with Pool(processes=args.workers) as pool:
                # We use imap_unordered to get results as they finish (chunk by chunk)
                # allowing us to save progress if interrupted between chunks, 
                # though usually chunks finish near the end.
                # A better way for fine-grained saving is difficult with Pool.map, 
                # so we catch the KeyboardInterrupt on the MAIN process.
                
                iterator = pool.imap_unordered(worker_process, process_args)
                
                # Progress bar for CHUNKS (not individual files, to keep UI clean)
                with tqdm(total=len(chunks), desc="Processing Chunks") as pbar:
                    for chunk_result in iterator:
                        results_new.extend(chunk_result)
                        pbar.update(1)
                        
                        # Optional: Intermediate save every X chunks could go here
                        
        except KeyboardInterrupt:
            print("\n\n!!! CAUGHT CTRL+C !!!")
            print("Terminating workers...")
            # The pool context manager usually handles termination, but we ensure we save what we have.
            # Combine old data with whatever new data we gathered before the crash
            total_data = processed_data + results_new
            save_checkpoint(total_data, args.output_dir)
            sys.exit(0)
            
        # Combine old and new
        total_data = processed_data + results_new
        save_checkpoint(total_data, args.output_dir) # Save final checkpoint

    # 5. Generate Final Splits
    print("\nGenerating final Train/Test/Val splits...")
    
    # Shuffle the TOTAL dataset before splitting
    random.shuffle(total_data)

    r_train, r_val, _ = map(float, args.split_ratio.split(','))
    n_train = int(len(total_data) * r_train)
    n_val = int(len(total_data) * r_val)
    
    splits = {
        "train": total_data[:n_train],
        "val": total_data[n_train:n_train+n_val],
        "test": total_data[n_train+n_val:]
    }

    for name, data in splits.items():
        out_path = os.path.join(args.output_dir, f"{name}.txt")
        with open(out_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
                
        print(f"Saved {name} split ({len(data)} items)")
    
    # Optional: Delete checkpoint if successfully finished
    # os.remove(checkpoint_path) 

if __name__ == "__main__":
    main()