import argparse
import os
import yaml
from easydict import EasyDict as edict
from model import CRNNEncoder, AttentionNet
from trainer import KWSModel
from utils import fix_seeds
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


def train(config):
    fix_seeds(seed=config.train.seed)

    crnn = CRNNEncoder(
        in_channels=config.model.get('in_channels', 42),
        hidden_size=config.model.get('hidden_size', 16),
        dropout=config.model.get('dropout', 0.1),
        cnn_layers=config.model.get('cnn_layers', 2),
        rnn_layers=config.model.get('rnn_layers', 2),
        kernel_size=config.model.get('kernel_size' 9)
    )
    model = AttentionNet(
        crnn,
        hidden_size=config.model.get('hidden_size', 16),
        num_classes=config.model.get('num_classes', 3)
    )
    pl_model = KWSModel(
        model, lr=config.train.get('lr', 4e-5),
         in_channels=config.model.get('in_channels', 42),
         batch_size=config.train.get('batch_size', 32)
    )
    wandb_logger = WandbLogger(name=config.train.get('experiment_name', 'final_run'), project='kws-attention', log_model=True)
    wandb_logger.log_hyperparams(config)
    wandb_logger.watch(model, log='all', log_freq=100)
    trainer = pl.Trainer(max_epochs=config.train.get('max_epochs', 15), logger=wandb_logger, gpus=config.train.get('gpus', 1))
    trainer.fit(pl_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training model.')
    parser.add_argument('--config', default='configs/train.yml',
                        help='path to config file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = edict(yaml.safe_load(stream))
    train(config)
