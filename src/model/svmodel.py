import pytorch_lightning as lightning
import torch
from misc.eer import compute_eer, compute_scores, read_trials


class SVSystem(lightning.LightningModule):
    def __init__(
        self,
        fbank,
        embedding,
        loss,
        dev_trials=None,
        test_trials=None,
    ) -> None:
        super().__init__()

        self.fbank = fbank
        self.embedding = embedding
        self.loss = loss

        self.dev_trials = dev_trials
        self.test_trials = test_trials

    def forward(self, x) -> torch.Tensor:
        x = self.fbank(x)
        x = self.embedding(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=1e-8,
            max_lr=1e-3,
            step_size_up=130_000 // 2,
            step_size_down=130_000 // 2,
            mode='triangular2',
            cycle_momentum=False
        )

        lr_scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step"
        }

        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        emb = self(x)
        loss = self.loss(emb, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, key = batch
        emb = self(x)
        emb = {key[i]: emb[i] for i in range(len(x))}
        return emb

    def validation_epoch_end(self, output_results):
        # merge all dicts in `emb`
        emb = {}
        for output in output_results:
            emb |= output

        labels, trials = read_trials(self.dev_trials)
        scores = compute_scores(emb, trials)
        eer, threshold = compute_eer(scores, labels)

        self.log('val_eer', eer)

    def test_step(self, batch, batch_idx):
        x, key = batch
        emb = self(x)
        emb = {key[i]: emb[i] for i in range(len(x))}
        return emb

    def test_epoch__end(self, output_results):
        # merge all dicts in `emb`
        emb = {}
        for output in output_results:
            emb |= output

        labels, trials = read_trials(self.dev_trials)
        scores = compute_scores(emb, trials)
        eer, threshold = compute_eer(scores, labels)

        self.log('test_eer', eer)
