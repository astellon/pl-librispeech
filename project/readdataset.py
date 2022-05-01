import io
import random
import time

import soundfile
import torch
import torchaudio
import webdataset

from tqdm import tqdm


def flac_handler_soundfile(b):
    return torch.from_numpy(soundfile.read(io.BytesIO(b))[0])


def flac_handler_torchaudio(b):
    return torchaudio.load(io.BytesIO(b))[0]


def transform(data, crop_length=int(1.5 * 16000)):
    waveform, spk = data

    length = waveform.size(0)

    if length < crop_length:
        result = torch.nn.functional.pad(waveform, (0, crop_length - length))
    elif length == crop_length:
        result = waveform
    else:
        start = random.randint(0, len(waveform) - crop_length - 1)
        result = waveform[start:start+crop_length]

    return result, spk


def main():
    # url = "data/webdataset/shards/librespeech-train-{000000..000058}.tar"
    # url = "data/webdataset/tar/librespeech-train.tar"
    url = "https://storage.cloud.google.com/librispeech-webdataset/librespeech-train-{000000..000058}.tar"

    dataset = webdataset.WebDataset(url, shardshuffle=True) \
        .shuffle(1000) \
        .decode(webdataset.handle_extension("flac", flac_handler_soundfile)) \
        .to_tuple("waveform.flac", "speaker.id") \
        .map(transform) \
        .batched(16)

    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=None)

    start_time = time.time()

    count = 0

    for data in tqdm(dataloader):
        waveform, speaker_id = data
        count += waveform.size(0)

    end_time = time.time()

    print(f'Read {count} files in {end_time - start_time}s from {url}.')


if __name__ == "__main__":
    main()
