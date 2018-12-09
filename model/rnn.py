import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from .util import reorder_sequence, reorder_lstm_states
from .embed_regularize import embedded_dropout
from .dropout import WeightDropout


def lstm_encoder(sequence, lstm, seq_lens=None, init_states=None, embedding=None,
                 dropoute=0.1, dropouti=0.65, lockdrop=None, isTraining=True):
    """ functional LSTM encoder (sequence is [b, t]/[b, t, d],
    lstm should be rolled lstm)"""
    batch_size = sequence.size(0)
    if not lstm.batch_first:
        sequence = sequence.transpose(0, 1)

    # embedding dropout & input dropout
    if embedding is not None:
        emb_sequence = embedded_dropout(embedding, sequence, dropout=dropoute if isTraining else 0)
        emb_sequence = lockdrop(emb_sequence, dropouti)
    else:
        emb_sequence = sequence

    if seq_lens:
        assert batch_size == len(seq_lens)
        sort_ind = sorted(range(len(seq_lens)),
                          key=lambda i: seq_lens[i], reverse=True)
        seq_lens = [seq_lens[i] for i in sort_ind]
        emb_sequence = reorder_sequence(emb_sequence, sort_ind,
                                        lstm.batch_first)

    if init_states is None:
        device = sequence.device
        init_states = init_lstm_states(lstm, batch_size, device)
    else:
        init_states = (init_states[0].contiguous(),
                       init_states[1].contiguous())

    if seq_lens:
        packed_seq = nn.utils.rnn.pack_padded_sequence(emb_sequence,
                                                       seq_lens)
        packed_out, final_states = lstm(packed_seq, init_states, seq_lens)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out)

        back_map = {ind: i for i, ind in enumerate(sort_ind)}
        reorder_ind = [back_map[i] for i in range(len(seq_lens))]
        lstm_out = reorder_sequence(lstm_out, reorder_ind, lstm.batch_first)
        final_states = reorder_lstm_states(final_states, reorder_ind)
    else:
        lstm_out, final_states = lstm(emb_sequence, init_states)

    return lstm_out, final_states


def init_lstm_states(lstm, batch_size, device):
    n_layer = lstm.num_layers*(2 if lstm.bidirectional else 1)
    n_hidden = lstm.hidden_size

    states = (torch.zeros(n_layer, batch_size, n_hidden).to(device),
              torch.zeros(n_layer, batch_size, n_hidden).to(device))
    return states


class StackedLSTMCells(nn.Module):
    """ stack multiple LSTM Cells"""
    def __init__(self, cells, dropout=0.0):
        super().__init__()
        self._cells = nn.ModuleList(cells)
        self._dropout = dropout

    def forward(self, input_, state):
        """
        Arguments:
            input_: FloatTensor (batch, input_size)
            states: tuple of the H, C LSTM states
                FloatTensor (num_layers, batch, hidden_size)
        Returns:
            LSTM states
            new_h: (num_layers, batch, hidden_size)
            new_c: (num_layers, batch, hidden_size)
        """
        hs = []
        cs = []
        for i, cell in enumerate(self._cells):
            s = (state[0][i, :, :], state[1][i, :, :])
            h, c = cell(input_, s)
            hs.append(h)
            cs.append(c)
            input_ = F.dropout(h, p=self._dropout, training=self.training)

        new_h = torch.stack(hs, dim=0)
        new_c = torch.stack(cs, dim=0)

        return new_h, new_c

    @property
    def hidden_size(self):
        return self._cells[0].hidden_size

    @property
    def input_size(self):
        return self._cells[0].input_size

    @property
    def num_layers(self):
        return len(self._cells)

    @property
    def bidirectional(self):
        return self._cells[0].bidirectional


class MultiLayerLSTMCells(StackedLSTMCells):
    """
    This class is a one-step version of the cudnn LSTM
    , or multi-layer version of LSTMCell
    """
    def __init__(self, input_size, hidden_size, num_layers,
                 bias=True, dropout=0.0, wdrop=0.5):
        """ same as nn.LSTM but without (bidirectional)"""

        cells = []
        cells.append(nn.LSTMCell(input_size, hidden_size, bias))
        for _ in range(num_layers-1):
            cells.append(nn.LSTMCell(hidden_size, hidden_size, bias))

        super().__init__(cells, dropout)

    @property
    def bidirectional(self):
        return False

    def reset_parameters(self):
        for cell in self._cells:
            # xavier initilization
            gate_size = self.hidden_size / 4
            for weight in [cell.weight_ih, cell.weight_hh]:
                for w in torch.chunk(weight, 4, dim=0):
                    init.xavier_normal_(w)
            #forget bias = 1
            for bias in [cell.bias_ih, cell.bias_hh]:
                torch.chunk(bias, 4, dim=0)[1].data.fill_(1)

    @staticmethod
    def convert(lstm):
        """ convert from a cudnn LSTM"""
        lstm_cell = MultiLayerLSTMCells(
            lstm.input_size, lstm.hidden_size,
            lstm.num_layers, dropout=lstm.dropout)
        for i, cell in enumerate(lstm_cell._cells):
            cell.weight_ih.data.copy_(getattr(lstm, 'weight_ih_l{}'.format(i)))
            cell.weight_hh.data.copy_(getattr(lstm, 'weight_hh_l{}'.format(i)))
            cell.bias_ih.data.copy_(getattr(lstm, 'bias_ih_l{}'.format(i)))
            cell.bias_hh.data.copy_(getattr(lstm, 'bias_hh_l{}'.format(i)))
        return lstm_cell


class MultiLayerLSTMCells_abs_enc(nn.Module):
    """ stack multiple LSTM Cells"""
    def __init__(self, input_size, hidden_size, num_layers,
                 bias=True, dropout=0.4, wdrop=0.5, dropouth=0.3, bidirectional=False, lockdrop=None):
        super().__init__()
        # weight_drop
        bi_multiplier = 2 if bidirectional else 1
        cells = [nn.LSTM(input_size if l == 0 else hidden_size * bi_multiplier,
                         hidden_size,
                         num_layers=1,
                         bias=bias,
                         dropout=0,
                         bidirectional=bidirectional) for l in range(num_layers)]
        cells = [WeightDropout(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in cells]

        self._cells = nn.ModuleList(cells)
        self._input_size = input_size
        self._hidden_size = hidden_size
        self.dropout = dropout
        self.nlayers = num_layers
        self.batch_first = self._cells[0].batch_first
        self.lockdrop = lockdrop
        self.dropouth = dropouth
        self._bi = bidirectional

    def forward(self, input_, state, seq_lens=None):
        """
        Arguments:
            input_: FloatTensor (batch, input_size)
            states: tuple of the H, C LSTM states
                FloatTensor (num_layers, batch, hidden_size)
        Returns:
            LSTM states
            new_h: (num_layers, batch, hidden_size)
            new_c: (num_layers, batch, hidden_size)
        """
        hs = []
        cs = []
        output = input_
        for i, cell in enumerate(self._cells):
            if self._bi:
                s = (state[0][i*2:(i+1)*2, :, :], state[1][i*2:(i+1)*2, :, :])
            else:
                s = (state[0][i, :, :], state[1][i, :, :])
            output, (h, c) = cell(input_, s)
            hs.append(h)
            cs.append(c)
            if i != self.nlayers - 1:
                input_ = self.lockdrop(output, self.dropouth, seq_lens)
            else:
                input_ = output

        new_h = torch.cat(hs)
        new_c = torch.cat(cs)
        output = self.lockdrop(output, self.dropout, seq_lens)

        return output, (new_h, new_c)

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def input_size(self):
        return self._input_size

    @property
    def num_layers(self):
        return len(self._cells)

    @property
    def bidirectional(self):
        return self._bi


class MultiLayerLSTMCells_abs_dec(nn.Module):
    """ stack multiple LSTM Cells"""
    def __init__(self, input_size, hidden_size, num_layers,
                 bias=True, wdrop=0.5):
        super().__init__()
        # weight_drop
        cells = [nn.LSTMCell(input_size if l == 0 else hidden_size,
                             hidden_size,
                             bias=bias) for l in range(num_layers)]
        cells = [WeightDropout(rnn, ['weight_hh'], dropout=wdrop) for rnn in cells]

        self._cells = nn.ModuleList(cells)

    def forward(self, input_, state, dropout_mask=None):
        """
        Arguments:
            input_: FloatTensor (batch, input_size)
            states: tuple of the H, C LSTM states
                FloatTensor (num_layers, batch, hidden_size)
        Returns:
            LSTM states
            new_h: (num_layers, batch, hidden_size)
            new_c: (num_layers, batch, hidden_size)
        """
        hs = []
        cs = []
        for i, cell in enumerate(self._cells):
            s = (state[0][i, :, :], state[1][i, :, :])
            h, c = cell(input_, s)
            hs.append(h)
            cs.append(c)
            if self.training:
                input_ = dropout_mask[i] * h
            else:
                input_ = h

        new_h = torch.stack(hs, dim=0)
        new_c = torch.stack(cs, dim=0)

        return new_h, new_c

    @property
    def hidden_size(self):
        return self._cells[0].hidden_size

    @property
    def input_size(self):
        return self._cells[0].input_size

    @property
    def num_layers(self):
        return len(self._cells)
