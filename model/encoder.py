from torch import nn
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)

        self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), 
                          bidirectional=True)
        
        # Since we're using a bidirectional LSTM, we will map
        # forward and backward hidden states to just one
        self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, x):
        # Convert word indexes to embeddings
        embedded = self.embedding(x)

        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        # 
        # encoder_states: output of shape (seq_len, batch, num_directions * hidden_size)
        # tensor containing the output features (h_t) from the last layer of the LSTM, for each t.
        #
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        # c_n of shape (num_layers * num_directions, batch, hidden_size)
        encoder_states, (hidden, cell) = self.rnn(embedded)
        
        # (2, N, hidden_size)
        # the hidden contains the most right state
        hidden = self.fc_hidden(
            torch.cat((hidden[0:1], hidden[1:2]),dim=2)
        )
        cell = self.fc_hidden(
            torch.cat((cell[0:1], cell[1:2]),dim=2)
        )

        # the encoder states in the Attention paper is the h_j
        return encoder_states, hidden, cell