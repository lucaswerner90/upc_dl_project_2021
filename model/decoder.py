import torch
import torch.nn.functional as F

from torch import nn


#palabra anterior o <SOS>       -->
#[features_imagen]              --> attention -->  decoder --> [0,0......,1,0,0,0,0,0....0]



class Decoder(nn.Module):
	def __init__(self, vocab_size, embed_size, hidden_size, num_layers = 1):
		super(Decoder, self).__init__()
		self.embed = nn.Embedding(vocab_size, embed_size)

        # hidden_size * 2 => because we now have the encoder_states which are
        # states for backward and forward states
		self.lstm = nn.LSTM((hidden_size * 2) + embed_size, hidden_size, num_layers)
		self.linear = nn.Linear(hidden_size, vocab_size)

        # we send the hidden states from the encoder (hidden_size * 2 because it's bidirectional)
        # and the previous state of the decoder (which also has hidden_size dimensions)
        self.energy = nn.Linear(hidden_size * 3, 1)

        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

		self.dropout = nn.Dropout(.5)

	def forward(self, image_features, word, encoder_states, prev_hidden_state, prev_cell_state):
        """
        It's important to remember that we compute one time step at a time

        word => (vocab_size)
        image_features => 
        encoder_states => (seq_len, batch, num_directions * hidden_size)
        """
        seq_len = encoder_states.shape[0]
        h_reshaped = prev_hidden_state.repeat(seq_len, 1, 1)

        # (vocab_size) => (1, vocab_size)
        word = word.unsqueeze(0)

		embeddings = self.embed(word)
		embeddings = torch.cat((image_features.unsqueeze(0),embeddings),dim=0)

        energy = self.relu(self.energy(
            torch.cat((h_reshaped, encoder_states), dim=-1)
        ))
        # attention => (seq_len, batch_size, 1)
        attention = self.softmax(energy)

        # Encoder states: (seq_len, batch, num_directions * hidden_size)
        # attention: (seq_len, batch_size, 1)
        # torch.bmm
        
        # (batch_size, 1, seq_length)
        attention = attention.permute(1,2,0)

        encoder_states = encoder_states.permute(1,0,2)

        # (vocab_size, 1, hidden_size*2) => (vocab_size, batch_size, hidden_size * 2)
        context_vector = torch.bmm(attention, encoder_states).permute(1,0,2)

        rnn_input = torch.cat((context_vector, embeddings), dim=2)

		outputs, (hidden, cell) = self.lstm(rnn_input, (prev_hidden_state,prev_cell_state))
		outputs = self.linear(outputs)
        
        predictions = outputs.squeeze(0)

		return predictions, hidden, cell