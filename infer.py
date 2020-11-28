import argparse
import os
import yaml
from easydict import EasyDict as edict
from model import CRNNEncoder, AttentionNet
from data.transforms import *
from utils import fix_seeds

import torchaudio
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def visualize(preds):
  for id, label in zip([1, 2], ['marvin', 'sheila']):
      plt.plot(preds[:, 0, id], label=label)

  plt.xlabel('time')
  plt.ylabel('prob')
  plt.legend()
  plt.savefig('predictions.png')


# run in streaming mode:
def run_streaming_on_file(model, file_audio, T):
    model.streaming_mode = True
    model.last_h = None

    waveform, sample_rate = torchaudio.load(file_audio)
    data_transforms = Compose([
        ToNumpy(),
        AudioSqueeze(),
        ToGpu(device='cuda' if torch.cuda.is_available() else 'cpu', batched=False),
        NormalizedMelSpectrogram(
            sample_rate=16000,
            n_mels=42,
            normalize='touniform',
            batched=False
        ).to('cuda' if torch.cuda.is_available() else 'cpu')
    ])
    data = data_transforms({'audio' : waveform, 'sample_rate': sample_rate})
    melspec = data['audio'][None]
    preds = []
    for indx in range(melspec.shape[-1] - T):
        slice_melspec = melspec[..., indx:indx+T]
        with torch.no_grad():
            raw_probs = model(slice_melspec, torch.tensor([T]).to('cuda' if torch.cuda.is_available() else 'cpu'))
            y_pred = F.softmax(raw_probs, dim=1).cpu().numpy()
        preds.append(y_pred)
    return np.array(preds)

def infer(config, weights_path, audio, T):
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
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    preds = run_streaming_on_file(model, audio, T)
    labels = np.array(['marvin', 'sheila'])
    mask = preds[:, 0, [1, 2]].max(dim=0) > 0.75
    if len(labels[mask]) > 0:
        print('keywords found!')
        print(labels[mask])
    else:
        print('keywords not found!')
    visualize(preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script.')
    parser.add_argument('--config', default='configs/train.yml',
                        help='path to config file')
    parser.add_argument('--weights_path',
                        help='path to weights of trained model')
    parser.add_argument('--audio',
                        help='path to audio to infer on')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = edict(yaml.safe_load(f))
    infer(config, args.weights_path, args.audio, config.get('inference', {}).get('T', 50))
