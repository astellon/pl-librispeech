import torch
import torchaudio
import pytorch_lightning as lightning

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data.librispeech import LibriSpeech
from model.ecapa import ECAPATDNN
from model.svmodel import SVSystem
from model.aamsoftmax import AAMsoftmax


def main():
    # datamodule = LibriSpeech(
    #     'data/materials/webdataset/shards/librespeech-train-{000000..000058}.tar',
    #     'data/materials/webdataset/shards/librespeech-dev-000000.tar',
    #     'data/materials/webdataset/shards/librespeech-test-000000.tar',
    #     batch_size=128,
    # )

    datamodule = LibriSpeech(
        'gs://librispeech-webdataset/librespeech-train-{000000..000058}.tar',
        'gs://librispeech-webdataset/librespeech-dev-000000.tar',
        'gs://librispeech-webdataset/librespeech-test-000000.tar',
        batch_size=128,
    )

    fbank = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=512,
        win_length=int(0.025 * 16000),
        hop_length=int(0.010 * 16000),
        n_mels=80
    )

    embedding = ECAPATDNN()
    loss = AAMsoftmax(2338, 0.2, 30)

    model = SVSystem(
        fbank,
        embedding,
        loss,
        dev_trials='data/materials/dev-trials.txt',
        test_trials='data/materials/test-trials.txt'
    )

    # callbacks
    checkpoint_callback = ModelCheckpoint(dirpath="data/artifacts/checkpoints", save_top_k=5, monitor="val_eer")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # logger
    wandb_logger = WandbLogger(name='librispeech', save_dir='data/artifacts/wandb')

    trainer = lightning.Trainer(
        accelerator='gpu',
        logger=wandb_logger,
        callbacks=[lr_monitor, checkpoint_callback],
        num_sanity_val_steps=-1,
        max_steps=130_000 * 4
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == '__main__':
    main()
