import io
import random

import pytorch_lightning as lightning
import soundfile
import torch
import webdataset


def flac_handler_soundfile(b):
    return torch.from_numpy(soundfile.read(io.BytesIO(b))[0].astype('float32'))


def random_crop(data, crop_length=int(1.5 * 16000)):
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


class LibriSpeech(lightning.LightningDataModule):
    def __init__(self, url_train, url_dev, url_test, batch_size):
        super().__init__()
        self.url_train = url_train
        self.url_dev = url_dev
        self.url_test = url_test
        self.batch_size = batch_size

    def make_dataset(self, url, label='speaker.id'):
        transform = random_crop

        return webdataset.WebDataset(url) \
            .shuffle(1000) \
            .decode(webdataset.handle_extension("flac", flac_handler_soundfile)) \
            .to_tuple("waveform.flac", label) \
            .map(transform) \
            .batched(self.batch_size)

    def setup(self, stage):
        self.trainset = self.make_dataset(self.url_train)
        self.devset = self.make_dataset(self.url_dev, '__key__')
        self.testset = self.make_dataset(self.url_test, '__key__')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset, num_workers=4, batch_size=None)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.devset, num_workers=4, batch_size=None)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.testset, num_workers=4, batch_size=None)
