import torchaudio
from models import dispatch_dense_model, dispatch_quantizer
from data.speech_encoder import SpeechEncoder
from vocoders.tacotron2.vocoder import TacotronVocoder


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dense_model_name",
        type=str,
        default="hubert-base-ls960",
        choices=["hubert-base-ls960", "cpc-big-ll6k"],
        help="Dense representation model",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=50,
        help="Vocabulary size used for resynthesis",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input audio file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output audio file.",
    )
    parser.add_argument(
        "--decoder_steps",
        type=int,
        default=100,
        help="Maximal number of decoder steps",
    )

    args = parser.parse_args()
    return args


def get_compression_rate(dense_model, units, wave, vocab_size, sample_rate):
    import numpy as np

    assert units.ndim == 1
    assert wave.ndim == 1

    time_in_seconds = wave.numel() / sample_rate

    uniform_token_entropy = np.log2(vocab_size)
    # calculated on LL-6k train
    unigram_token_entropy = {
        "hubert-base-ls960": {
            50: 5.458528917634601,
            100: 6.44513268276806,
            200: 7.477069233162813,
        },
        "cpc-big-ll6k": {
            50: 5.428271158461133,
            100: 6.413083187885448,
            200: 7.44253841579776,
        },
    }[dense_model][vocab_size]

    uniform_bps = uniform_token_entropy * units.size(0) / time_in_seconds
    unigram_entropy = unigram_token_entropy * units.size(0) / time_in_seconds

    return uniform_bps, unigram_entropy


def main(args):
    dense_model_name = args.dense_model_name
    quantizer_name = "kmeans"

    dense_model = dispatch_dense_model(dense_model_name)
    quantizer_model = dispatch_quantizer(
        dense_model_name, quantizer_name, args.vocab_size
    )

    encoder = SpeechEncoder(
        dense_model=dense_model,
        quantizer_model=quantizer_model,
        need_f0=True,
        deduplicate=True,
        f0_normalizer=None,
        f0_quantizer=None,
    ).cuda()

    waveform, input_sample_rate = torchaudio.load(args.input)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)

    waveform = encoder.maybe_resample(waveform, input_sample_rate)
    encoded = encoder(waveform.cuda())

    units = encoded[
        "units"
    ]
    vocoder = TacotronVocoder.by_name(
        dense_model_name,
        quantizer_name,
        args.vocab_size,
    ).cuda()

    audio = vocoder(units)

    torchaudio.save(
        args.output, audio.cpu().float().unsqueeze(0), vocoder.output_sample_rate
    )

    uniform_bps, learned_bps = get_compression_rate(
        dense_model_name, units, waveform, args.vocab_size, encoder.expected_sample_rate
    )

    print(
        f"Audio of length {round(waveform.size(0) / 16_000, 1)} seconds represented as {units.numel()} tokens"
    )
    print(
        f"\tAssuming uniform token distribution: {round(uniform_bps, 1)} bits per second"
    )
    print(
        f"\tAssuming unigram token distribution estimated on LL-6K train: {round(learned_bps, 1)} bits per second"
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
