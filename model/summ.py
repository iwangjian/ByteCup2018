import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
from .rnn import lstm_encoder
from .rnn import MultiLayerLSTMCells_abs_dec, MultiLayerLSTMCells_abs_enc
from .attention import step_attention
from .util import sequence_mean, len_mask
from .dropout import LockedDropout


INIT = 1e-2


class Seq2SeqSumm(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_hidden, bidirectional, n_layer,
                 dropoute=0.1, dropouti=0.65, dropout=0.4, wdrop=0.5, dropouth=0.3):
        super().__init__()
        # embedding weight parameter is shared between encoder, decoder,
        # and used as final projection layer to vocab logit
        # can initialize with pretrained word vectors
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lockdrop = LockedDropout()
        self.dropoute = dropoute # dropout to remove words from embedding layer (0 = no dropout)
        self.dropouti = dropouti # dropout for input embedding layers (0 = no dropout)
        self.dropout = dropout # dropout applied to layers (0 = no dropout)
        self.wdrop = wdrop # amount of weight dropout to apply to the RNN hidden to hidden matrix
        self.dropouth = dropouth # dropout for rnn layers (0 = no dropout)
        self.emb_size = emb_dim

        # self._enc_lstm = nn.LSTM(
        #     emb_dim, n_hidden, n_layer,
        #     bidirectional=bidirectional, dropout=dropout
        # )
        self._enc_lstm = MultiLayerLSTMCells_abs_enc(
            emb_dim, n_hidden, n_layer,
            dropout=self.dropout, wdrop=self.wdrop, dropouth=self.dropouth,
            bidirectional=bidirectional, lockdrop=self.lockdrop
        )

        # initial encoder LSTM states are learned parameters
        state_layer = n_layer * (2 if bidirectional else 1)
        self._init_enc_h = nn.Parameter(
            torch.Tensor(state_layer, n_hidden)
        )
        self._init_enc_c = nn.Parameter(
            torch.Tensor(state_layer, n_hidden)
        )
        init.uniform_(self._init_enc_h, -INIT, INIT)
        init.uniform_(self._init_enc_c, -INIT, INIT)

        # self._dec_lstm = MultiLayerLSTMCells(
        #     2 * emb_dim, n_hidden, n_layer, dropout=dropout
        # )
        self._dec_lstm = MultiLayerLSTMCells_abs_dec(
            2 * emb_dim, n_hidden, n_layer,
            wdrop=self.wdrop
        )

        # project encoder final states to decoder initial states
        enc_out_dim = n_hidden * (2 if bidirectional else 1)
        self._dec_h = nn.Linear(enc_out_dim, n_hidden, bias=False)
        self._dec_c = nn.Linear(enc_out_dim, n_hidden, bias=False)
        # multiplicative attention
        self._attn_wm = nn.Parameter(torch.Tensor(enc_out_dim, n_hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        init.xavier_normal_(self._attn_wm)
        init.xavier_normal_(self._attn_wq)
        # project decoder output to emb_dim, then
        # apply weight matrix from embedding layer
        self._projection = nn.Sequential(
            nn.Linear(2*n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, emb_dim, bias=False)
        )
        # functional object for easier usage
        self._decoder = AttentionalLSTMDecoder(
            self._embedding, self._dec_lstm,
            self._attn_wq, self._projection,
            dropouth=self.dropouth
        )

    def forward(self, article, art_lens, abstract):
        attention, init_dec_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        logit = self._decoder((attention, mask), init_dec_states, abstract)
        return logit

    def encode(self, article, art_lens=None):
        size = (
            self._init_enc_h.size(0),
            len(art_lens) if art_lens else 1,
            self._init_enc_h.size(1)
        )
        init_enc_states = (
            self._init_enc_h.unsqueeze(1).expand(*size),
            self._init_enc_c.unsqueeze(1).expand(*size)
        )

        # enc_art, final_states = lstm_encoder(
        #     article, self._enc_lstm, art_lens,
        #     init_enc_states, self._embedding
        # )
        enc_art, final_states = lstm_encoder(
            article, self._enc_lstm, art_lens,
            init_enc_states, self._embedding,
            dropoute=self.dropoute, dropouti=self.dropouti, lockdrop=self.lockdrop,
            isTraining=self.training
        )

        if self._enc_lstm.bidirectional:
            h, c = final_states
            final_states = (
                torch.cat(h.chunk(2, dim=0), dim=2),
                torch.cat(c.chunk(2, dim=0), dim=2)
            )
        init_h = torch.stack([self._dec_h(s)
                              for s in final_states[0]], dim=0)
        init_c = torch.stack([self._dec_c(s)
                              for s in final_states[1]], dim=0)
        init_dec_states = (init_h, init_c)
        attention = torch.matmul(enc_art, self._attn_wm).transpose(0, 1)
        init_attn_out = self._projection(torch.cat(
            [init_h[-1], sequence_mean(attention, art_lens, dim=1)], dim=1
        ))
        return attention, (init_dec_states, init_attn_out)

    def batch_decode(self, article, art_lens, go, eos, max_len):
        """ greedy decode support batching"""
        batch_size = len(art_lens)
        attention, init_dec_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        attention = (attention, mask)
        tok = torch.LongTensor([go]*batch_size).to(article.device)
        outputs = []
        attns = []
        states = init_dec_states

        # locked dropout
        m = init_dec_states[0][0].data.new(init_dec_states[0][0].size(0),
                                           init_dec_states[0][0].size(1),
                                           self.emb_size * 2
                                           ).bernoulli_(1 - self.dropouth)
        dropout_mask = Variable(m, requires_grad=False) / (1 - self.dropouth)

        for i in range(max_len):
            tok, states, attn_score = self._decoder.decode_step(
                tok, states, attention, dropout_mask)
            outputs.append(tok[:, 0])
            attns.append(attn_score)
        return outputs, attns

    def decode(self, article, go, eos, max_len):
        attention, init_dec_states = self.encode(article)
        attention = (attention, None)
        tok = torch.LongTensor([go]).to(article.device)
        outputs = []
        attns = []
        states = init_dec_states

        # locked dropout
        m = init_dec_states[0][0].data.new(init_dec_states[0][0].size(0),
                                           init_dec_states[0][0].size(1),
                                           self.emb_size * 2
                                           ).bernoulli_(1 - self.dropouth)
        dropout_mask = Variable(m, requires_grad=False) / (1 - self.dropouth)

        for i in range(max_len):
            tok, states, attn_score = self._decoder.decode_step(
                tok, states, attention, dropout_mask)
            if tok[0, 0].item() == eos:
                break
            outputs.append(tok[0, 0].item())
            attns.append(attn_score.squeeze(0))
        return outputs, attns

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)
        # self._embedding.weight.requires_grad = False


class AttentionalLSTMDecoder(object):
    def __init__(self, embedding, lstm, attn_w, projection, dropouth):
        super().__init__()
        self._embedding = embedding
        self._lstm = lstm
        self._attn_w = attn_w
        self._projection = projection
        self._dropouth = dropouth

    def __call__(self, attention, init_states, target):
        max_len = target.size(1)
        states = init_states
        logits = []

        # locked dropout
        m = states[0][0].data.new(states[0][0].size(0), # n_layer
                                  states[0][0].size(1), # batch_size
                                  states[0][0].size(2)  # embedding_size * 2
                                  ).bernoulli_(1 - self._dropouth)
        dropout_mask = Variable(m, requires_grad=False) / (1 - self._dropouth)

        for i in range(max_len):
            tok = target[:, i:i+1]
            logit, states, _ = self._step(tok, states, attention, dropout_mask)
            logits.append(logit)
        logit = torch.stack(logits, dim=1)
        return logit

    def _step(self, tok, states, attention, dropout_mask):
        prev_states, prev_out = states
        lstm_in = torch.cat(
            [self._embedding(tok).squeeze(1), prev_out],
            dim=1
        )
        states = self._lstm(lstm_in, prev_states, dropout_mask)
        lstm_out = states[0][-1]
        query = torch.mm(lstm_out, self._attn_w)
        attention, attn_mask = attention
        context, score = step_attention(
            query, attention, attention, attn_mask)
        dec_out = self._projection(torch.cat([lstm_out, context], dim=1))
        states = (states, dec_out)
        logit = torch.mm(dec_out, self._embedding.weight.t())
        return logit, states, score

    def decode_step(self, tok, states, attention, dropout_mask):
        logit, states, score = self._step(tok, states, attention, dropout_mask)
        out = torch.max(logit, dim=1, keepdim=True)[1]
        return out, states, score
