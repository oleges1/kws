import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch import nn
import torch
import torch.nn.functional as F
import os
import sys
import torchaudio
import numpy as np

from data.collate import no_pad_collate
from data.speechcommands import CommandsDataset
from data.transforms import *


class KWSModel(pl.LightningModule):
    def __init__(
           self,
           model,
           lr,
           in_channels,
           batch_size
        ):
        super(KWSModel, self).__init__()
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.mel_spectrogramer_train = Compose([
            ToGpu('cuda' if torch.cuda.is_available() else 'cpu'),
            NormalizedMelSpectrogram(
                sample_rate=16000,
                n_mels=in_channels,
                normalize='touniform'
            ).to('cuda' if torch.cuda.is_available() else 'cpu'),
            MaskSpectrogram(
                frequency_mask_max_percentage=0.2,
                time_mask_max_percentage=0.1,
                probability=1.0
            ),
            AddLengths(),
            Pad()
        ])
        self.mel_spectrogramer_val = Compose([
            ToGpu('cuda' if torch.cuda.is_available() else 'cpu'),
            NormalizedMelSpectrogram(
                sample_rate=16000,
                n_mels=in_channels,
                normalize='touniform'
            ).to('cuda' if torch.cuda.is_available() else 'cpu'),
            AddLengths(),
            Pad()
        ])

    def forward(self, batch):
        return self.model(batch['audio'], batch['lengths'])

    def training_step(self, batch, batch_nb):
        # REQUIRED
        batch = self.mel_spectrogramer_train(batch)
        y_hat = self(batch)
        y = batch['target']
        loss = F.cross_entropy(y_hat, y)
        y_pred = F.softmax(y_hat, dim=1).argmax(dim=1)
        acc = (y_pred == y).float().mean()
        fa = torch.bitwise_and(y_pred != y, y == 0).float().sum()
        fr = torch.bitwise_and(y_pred != y, y != 0).float().sum()
        length_zeros = batch['lengths'][y == 0].sum()
        length_nonzeros = batch['lengths'][y != 0].sum()

        self.logger.experiment.log({
            'train_loss': loss, 'train_acc': acc,
            'train_fa': fa / (y == 0).float().sum(), 'train_fr': fr / (y != 0).float().sum(),
            'train_fa_per_len': fa / length_zeros, 'train_fr_per_len': fr / length_nonzeros
        })
        return {'train_loss': loss, 'train_acc': acc}


    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        batch = self.mel_spectrogramer_val(batch)
        y_hat = self(batch)
        y = batch['target']
        loss = F.cross_entropy(y_hat, y)
        y_pred = F.softmax(y_hat, dim=1).argmax(dim=1)
        acc = (y_pred == y).float().mean()
        fa = torch.bitwise_and(y_pred != y, y == 0).float().sum()
        fr = torch.bitwise_and(y_pred != y, y != 0).float().sum()
        length_zeros = batch['lengths'][y == 0].sum()
        length_nonzeros = batch['lengths'][y != 0].sum()
        return {
            'val_loss': loss, 'val_acc': acc, 'fa': fa, 'fr': fr,
            'length_zeros': length_zeros, 'length_nonzeros': length_nonzeros,
            'zeros': (y == 0).float().sum(), 'non_zeros': (y != 0).float().sum()
        }

    def validation_epoch_end(self, outputs):
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_fa = torch.stack([x['fa'] for x in outputs]).sum() / torch.stack([x['zeros'] for x in outputs]).sum()
        avg_fr = torch.stack([x['fr'] for x in outputs]).sum() / torch.stack([x['non_zeros'] for x in outputs]).sum()

        avg_fa_per_len = torch.stack([x['fa'] for x in outputs]).sum() / torch.stack([x['length_zeros'] for x in outputs]).sum()
        avg_fr_per_len = torch.stack([x['fr'] for x in outputs]).sum() / torch.stack([x['length_nonzeros'] for x in outputs]).sum()

        self.logger.experiment.log({
            'avg_val_loss': avg_loss, 'avg_val_acc': avg_acc,
            'avg_val_fa': avg_fa, 'avg_val_fr': avg_fr,
            'avg_val_fa_per_len': avg_fa_per_len, 'avg_val_fr_per_len': avg_fr_per_len
        })

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # dataset:

    def prepare_data(self):
        CommandsDataset(train=True, download=True)

    def train_dataloader(self):
        train_transforms = Compose([
            ToNumpy(),
            AudioSqueeze(),
            AddGaussianNoise(
                min_amplitude=0.001,
                max_amplitude=0.015,
                p=0.5
            ),
            TimeStretch(
                min_rate=0.8,
                max_rate=1.25,
                p=0.5
            ),
            PitchShift(
                min_semitones=-4,
                max_semitones=4,
                p=0.5
            )
        ])
        dataset_train = CommandsDataset(train=True, download=False, transforms=train_transforms)
        dataset_train = torch.utils.data.DataLoader(dataset_train,
                              batch_size=self.batch_size, collate_fn=no_pad_collate, shuffle=True, num_workers=4)
        return dataset_train

    def val_dataloader(self):
        dataset_val = CommandsDataset(train=False, download=False, transforms=Compose([ToNumpy(), AudioSqueeze()]))
        dataset_val = torch.utils.data.DataLoader(dataset_val,
                              batch_size=self.batch_size, collate_fn=no_pad_collate, num_workers=4)
        return dataset_val
