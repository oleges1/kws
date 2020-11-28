import torch
import torch.nn.functional as F
import os
import sys
import torchaudio
import numpy as np

class CommandsDataset(torch.utils.data.Dataset):
    def __init__(
          self, train=True, transforms=None,
          targets=['marvin', 'sheila'], download=True,
          datadir="speech_commands"
        ):
        if download:
            os.system("wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz -O speech_commands_v0.01.tar.gz")
            # alternative url: https://www.dropbox.com/s/j95n278g48bcbta/speech_commands_v0.01.tar.gz?dl=1
            os.system(f"mkdir {datadir} && tar -C {datadir} -xvzf speech_commands_v0.01.tar.gz 1> log")
        self.samples_by_target = {
            cls: [os.path.join(datadir, cls, name) for name in os.listdir("./speech_commands/{}".format(cls)) if name.endswith('.wav')]
            for cls in os.listdir(datadir)
            if os.path.isdir(os.path.join(datadir, cls))
        }
        self.train = train
        self.transforms = transforms
        self.target2id = {target: i for i, target in enumerate(targets)}
        self.id2target = {i: target for target, i in self.target2id.items()}

        if self.train:
            self.samples_by_target = {
                cls: audios[:int(0.75 * len(audios))] for cls, audios in self.samples_by_target.items()
            }
        else:
            self.samples_by_target = {
                cls: audios[int(0.75 * len(audios)):] for cls, audios in self.samples_by_target.items()
            }
        self.samples = []
        for cls, audios in self.samples_by_target.items():
            for audio in audios:
                self.samples.append((audio, self.target2id.get(cls, -1) + 1))


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        file_audio, target_id = self.samples[idx]

        # Load audio
        waveform, sample_rate = torchaudio.load(file_audio)
        if self.transforms is not None:
            return self.transforms({'audio' : waveform, 'sample_rate': sample_rate, 'target': target_id})
        return {'audio' : waveform, 'sample_rate': sample_rate, 'target': target_id}
