import torch
import torch.nn.functional as F

from torch import nn


# palabra anterior o <SOS>       -->
# [features_imagen]              --> attention -->  decoder --> [0,0......,1,0,0,0,0,0....0]


class Decoder(nn.Module):
    def __init__(self, image_features_dim,vocab_size, embed_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # hidden_size * 2 => because we now have the encoder_states which are
        # states for backward and forward states
        self.rnn = nn.GRU(image_features_dim + embed_size, hidden_size, num_layers, batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab_size)

        self.relu = nn.ReLU()

    def init_hidden(self, batch_size:int):
        # (num_layers * num_directions, batch, hidden_size)
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

    def forward(self, context_vector, word, hidden_state=None):
        """
        It's important to remember that we compute one time step at a time

        word => (vocab_size)
        image_features => (embed_size) ?
        """
        # embeddings => (bsz, 1, embed_size)
        embeddings = self.embed(word)

        # embeddings => (bsz, embed_size)
        embeddings = embeddings.squeeze(1)

        rnn_input = torch.cat((context_vector, embeddings), dim=-1)

        outputs, hidden_state = self.rnn(rnn_input.unsqueeze(0), hidden_state)
        outputs = self.relu(self.linear(outputs))
        predictions = outputs.squeeze(0)

        return predictions, hidden_state
