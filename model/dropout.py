import torch
import torch.nn as nn
from torch.autograd import Variable


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5, seq_lens=None):
        if not self.training or not dropout:
            return x
        if seq_lens == None:
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
            mask = Variable(m, requires_grad=False) / (1 - dropout)
            mask = mask.expand_as(x)
            return mask * x
        else:
            x, _ = nn.utils.rnn.pad_packed_sequence(x)
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
            mask = Variable(m, requires_grad=False) / (1 - dropout)
            mask = mask.expand_as(x)
            x = mask * x
            return nn.utils.rnn.pack_padded_sequence(x, seq_lens)


class WeightDropout(nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDropout, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        if hasattr(module, 'batch_first'):
            self.batch_first = module.batch_first
        else:
            self.batch_first = False
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda: mask = mask.cuda()
                mask = nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)
