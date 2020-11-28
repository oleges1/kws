# KWS
Pytorch implementation for [Attention-based End-to-End Models for Small-Footprint Keyword Spotting](https://arxiv.org/abs/1803.10916).
Training with Pytorch-Lightning, inference on sliding window - streaming mode.

To run training you may use: train.py and set some config.
To run inference you may use: infer.py choose config, model weights, audio.

[Wandb logs](https://wandb.ai/oleges/kws-attention) of my training on SpeechCommands dataset.

Possible updates:
- Multi-head attention and orthogonality regularization from [ORTHOGONALITY CONSTRAINED MULTI-HEAD ATTENTION
FOR KEYWORD SPOTTING](https://arxiv.org/pdf/1910.04500.pdf)
