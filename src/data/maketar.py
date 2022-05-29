import os
import glob

import torch
import torchaudio
import webdataset


class LibriSpeech(torch.utils.data.Dataset):
    def __init__(self, path: str, split: str = 'train') -> None:
        self.fn = []
        self.text = []
        self.spk = []
        self.chap = []
        self.uttr = []

        urls = [url for url in os.listdir(path) if url.startswith(split)]

        for url in urls:
            # path/split/speaker id/chapter id/filename
            files = glob.glob(os.path.join(path, url, '*', '*', '*.flac'))

            for file in files:
                basename, _ = os.path.splitext(os.path.basename(file))
                spk, chap, uttr = basename.split('-')

                self.fn.append(file)
                self.spk.append(int(spk))
                self.chap.append(int(chap))
                self.uttr.append(int(uttr))

                text = os.path.join(path, url, spk, chap, spk + '-' + chap + '.trans.txt')

                with open(text) as ft:
                    for line in ft:
                        name, text = line.strip().split(" ", 1)
                        if basename == name:
                            break
                    else:
                        # Translation not found
                        raise FileNotFoundError("Translation not found for " + basename)

                self.text.append(text)

    def __len__(self):
        return len(self.fn)

    def __getitem__(self, index: int):
        return (
            self.fn[index],
            self.text[index],
            self.spk[index],
            self.chap[index],
            self.uttr[index]
        )


def make_dataset(root, folder_in_archive='LibriSpeech', split='train'):
    urls = [url for url in os.listdir(os.path.join(root, folder_in_archive)) if url.startswith(split)]

    dataset = LibriSpeech(os.path.join(root, folder_in_archive), split)
    sampler = torch.utils.data.RandomSampler(dataset)

    dst = os.path.join(root, 'webdataset', 'tar', f'librespeech-{split}.tar')

    with webdataset.TarWriter(dst) as sink:
        for i in sampler:
            # dont need raw waveform
            fn, text, spk, chap, utter = dataset[i]

            with open(fn, 'rb') as f:
                data = f.read()

            sample = {
                "__key__": f'{spk}-{chap}-{utter}',
                "waveform.flac": data,
                "text.txt": text,
                "speaker.id": spk,
            }

            sink.write(sample)


def main():
    make_dataset('data', split='train')
    make_dataset('data', split='dev')
    make_dataset('data', split='test')


if __name__ == '__main__':
    main()
