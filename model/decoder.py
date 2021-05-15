import torch
import torch.nn.functional as F

from torch import nn


#                                    palabra anterior o <SOS>       -->
#[features_imagen, embedding_palabra] --> attention                 -->  decoder --> [0,0......,1,0,0,0,0,0....0]




class Decoder(nn.Module):
    """
    This class is the attention based decoder that I have mentioned earlier.
    The ‘attn’ layer is used to calculate the value of e<ᵗ,ᵗ’> 
    which is the small neural network mentioned above. 
    This layer calculates the importance of that word, 
    by using the previous decoder hidden state and the hidden state of the encoder
    at that particular time step. 
    The ‘lstm’ layer takes in concatenation of vector obtained
    by having a weighted sum according to attention weights and the previous word outputted.
    The final layer is added to map the output feature space into the size of vocabulary,
    and also add some non-linearity while outputting the word.
    The ‘init_hidden’ function is used in the same way as in the encoder.
    """

    def __init__(self, hidden_size, output_size, vocab_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.attn = nn.Linear(hidden_size + output_size, 1)
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        # 
        # input_size: The number of expected features in the input `x`
        # hidden_size: The number of features in the hidden state `h`
        # 
        self.lstm = nn.LSTM(hidden_size + vocab_size, output_size)
        self.final = nn.Linear(output_size, vocab_size)

    def init_hidden(self):
        """
        It is a tuple because the hidden state of the LSTM contains 2 vectors:
        (h_0, c_0)

        * h_0 (num_layers * num_directions, batch, hidden_size):
        tensor containing the initial hidden state for each element in the batch.
        If the LSTM is bidirectional, num_directions should be 2, else it should be 1

        * c_0 (num_layers * num_directions, batch, hidden_size)
        tensor containing the initial cell state for each element in the batch.

        """

        return (torch.zeros(1, 1, self.output_size), torch.zeros(1, 1, self.output_size))

    def forward(self, previous_decoder_hidden_state, encoder_outputs, input):
        weights = []

        # We need to calculate the attention value for each of the encoder outputs
        for i in range(len(encoder_outputs)):

            # concat both the previous decoder hidden state and
            # one of the vectors from the encoder output.
            
            hidden_state, *_ = previous_decoder_hidden_state #(num_layers * num_directions, batch, hidden_size)
            attention_input = torch.cat((hidden_state[0], encoder_outputs[i]),dim=1)
            
            # computes the attention for that specific position of the encoder output
            # and using the previous hidden state
            attention_output = self.attn(attention_input)

            # for each attention weight that we calculate we append it to a list
            # this is basically the weight that each encoder output is being given
            # in relation to the timestep
            weights.append(attention_output)

        # normalize the weights using the softmax
        # so all of them are between 0 and 1
        normalized_weights = F.softmax(input = torch.cat(weights, 1), dim=1)

        # since we're doing self attention we need to multiply the attention weights
        # with the respective encoder outputs
        attn_applied = torch.bmm(normalized_weights.unsqueeze(1),
                                 encoder_outputs.view(1, -1, self.hidden_size))

        #  input_lstm = (seq_len, batch, input_size): tensor containing the features of the input sequence.
        input_lstm = torch.cat((attn_applied[0], input[0]), dim=1)

        # output = (seq_len, batch, num_directions * hidden_size)
        # hidden = (num_layers * num_directions, batch, hidden_size)
        output, hidden_state = self.lstm(input_lstm.unsqueeze(0), previous_decoder_hidden_state)

        # we finally return the vector that contains all the words in our language
        # with the probability for each of them
        output = self.final(output[0])

        return output, hidden_state, normalized_weights
