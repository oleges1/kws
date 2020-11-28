# model:
from torch import nn
import torch
import torch.nn.functional as F


class CRNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        cnn_layers=2,
        rnn_layers=2,
        hidden_size=16,
        kernel_size=9,
        dropout=0.2
    ):
      super(CRNNEncoder, self).__init__()
      self.kernel_size = kernel_size
      self.cnn_layers = cnn_layers

      prev_channels = in_channels
      channels = hidden_size * 25 if cnn_layers > 0 else None
      layers = []
      for _ in range(cnn_layers):
          layers.append(nn.Conv1d(
              prev_channels, channels, kernel_size=kernel_size,
              padding=(kernel_size - 1) // 2, bias=False
          ))
          layers.append(nn.BatchNorm1d(channels))
          layers.append(nn.ReLU())
          prev_channels = channels

      self.cnn_net = nn.Sequential(
          *layers
      ) if len(layers) > 0 else None

      self.rnn = nn.GRU(
          input_size=prev_channels,
          hidden_size=hidden_size,
          num_layers=rnn_layers,
          dropout=dropout,
          batch_first=True
      )


    def forward(
        self,
        input,
        last_h=None
    ):
      # input (batch_size, max_length, hidden_size)
      if last_h is None:
          if self.cnn_net is not None:
              input = self.cnn_net(input.permute(0, 2, 1)).permute(0, 2, 1)
          output, h = self.rnn(input)
          return output, h
      # streaming mode:
      part_input = input[:, -self.kernel_size * self.cnn_layers:]
      if self.cnn_net is not None:
            part_input = self.cnn_net(part_input.permute(0, 2, 1)).permute(0, 2, 1)
      output, h = self.rnn(part_input[:, [-1]], last_h)
      return output, h


class AttentionNet(nn.Module):
    def __init__(
        self,
        net,
        hidden_size,
        num_classes,
        streaming_mode=False
    ):
        super(AttentionNet, self).__init__()
        self.hidden_size = hidden_size
        self.streaming_mode = streaming_mode
        self.last_h = None

        self.net = net
        self.W_mem2hidden = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        self.w_global = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        input,
        length
    ):
        input = input.permute(0, 2, 1)
        # input (batch_size, max_length, hidden_size)
        # length (batch_size, )
        if self.streaming_mode:
            if self.last_h is not None:
                temp_out, self.last_h = self.net(input, self.last_h)
                out = torch.cat([self.prev_out[:, 1:], temp_out], dim=1)
            else:
                out, self.last_h = self.net(input)
            self.prev_out = out
        else:
            out, _ = self.net(input) # (batch_size, max_length, hidden_size)


        batch_size = out.size(0)
        max_length = out.size(1)

        # mask (batch_size, max_length)
        mask = torch.arange(max_length, device=length.device,
                        dtype=length.dtype)[None, :] < length[:, None]

        # if out.shape[1] > 0:
        scores = self.W_mem2hidden(out)
        scores = torch.tanh(scores)
        scores = self.v(scores).squeeze(2) # (batch_size, max_length)
        scores = scores.masked_fill(mask, -1e20) # (batch_size, max_length)
        attn_weights = F.softmax(scores, dim=1) # (batch_size, max_length)
        attn_weights = attn_weights.unsqueeze(1) # (batch_size, 1,  max_length)
        context = torch.matmul(attn_weights, out).squeeze(1) # (batch_size, hidden_size)

        w_t = self.w_global(context)
        return w_t
