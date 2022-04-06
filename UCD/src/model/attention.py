import torch
import torch.nn as nn
import torch.nn.functional as F


class WordDefnAttention(nn.Module):
    """
    Vanilla attention mechanism for fusing embeddings for word definitions.
    """
    def __init__(self, config):
        super(WordDefnAttention, self).__init__()
        self.config = config
        # Attention Layer
        self.attn_layer = torch.nn.Linear(
            2 * self.config.PRETRAINED_CONDGEN_EMBED_DIM + self.config.PRETRAINED_SENT_EMBED_DIM,
            self.config.PRETRAINED_SENT_EMBED_DIM
        )
        self.v = torch.nn.Parameter(torch.rand(self.config.PRETRAINED_SENT_EMBED_DIM))
        self.v.data.normal_(mean=0, std=1./self.v.size(0)**(1./2.))

    def compute_attn_score(self, hidden_states, encoder_output):
        # hidden: batch_size, max_src_seq_len, de_hidden_size*2
        # encoder_output: batch_size, max_src_seq_len, en_hidden_size
        # batch_size, max_src_seq_len, de_hidden_size
        energy = torch.tanh(self.attn_layer(torch.cat([hidden_states, encoder_output], 2)))

        # batch_size, de_hidden_size, max_src_seq_len
        energy = energy.transpose(2, 1)
        # batch_size, 1, de_hidden_size
        v = self.v.repeat(encoder_output.shape[0], 1).unsqueeze(1)
        # batch_size, 1, max_src_seq_len
        energy = torch.bmm(v, energy)
        # batch_size, max_src_seq_len
        return energy.squeeze(1)

    def forward(self, hidden_states, encoder_output, src_seq_lens):
        # hidden: (h, c)
        # h, c: 1, batch_size, de_hidden_size
        hidden_states = torch.unsqueeze(hidden_states, 0)
        # encoder_output: batch_size, max_src_seq_len, en_hidden_size
        # src_lens: batch_size
        # batch_size, de_hidden_size*2
        hidden_states = torch.cat((hidden_states, hidden_states), 2).squeeze(0)
        # batch_size, max_src_seq_len, de_hidden_size*2
        hidden_states = hidden_states.repeat(encoder_output.shape[1], 1, 1).transpose(0, 1)
        # batch_size, max_src_seq_len
        attn_energies = self.compute_attn_score(hidden_states, encoder_output)
        # max_src_seq_len
        idx = torch.arange(end=encoder_output.shape[1], dtype=torch.float, device=self.config.DEVICE)
        # batch_size, max_src_seq_len
        idx = idx.unsqueeze(0).expand(attn_energies.shape)
        # batch size, max_src_seq_len
        src_lens = src_seq_lens.unsqueeze(-1).expand(attn_energies.shape)
        mask = idx.long() < src_lens
        attn_energies[~mask] = float('-inf')
        # batch_size, 1, max_src_seq_len
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
