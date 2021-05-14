import torch
import torch.nn.functional as F

from torch import nn


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

        self.lstm = nn.LSTM(hidden_size + vocab_size, output_size)
        self.final = nn.Linear(output_size, vocab_size)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.output_size), torch.zeros(1, 1, self.output_size))

    def forward(self, decoder_hidden, encoder_outputs, input):
        """
        The forward function of the decoder takes the decoder’s previous hidden state, 
        encoder outputs and the previous word outputted. ‘weights’ list is used to store the attention weights. 
        Now, as we need to calculate attention weight for each encoder output, we iterate through them and pass 
        them through the ‘attn’ layer along with decoder’s previous hidden state by concatenating them and store 
        them in the ‘weights’ list. Once, we have these weights, we scale them in range (0,1) by applying softmax 
        activation to them. To calculate the weighted sum, we use batch matrix multiplication to multiply 
        attention vector of size (1,1, len(encoder_outputs)) and encoder_outputs of size (1, len(encoder_outputs), hidden_size) 
        to obtain the size of vector hidden_size is the weighted sum. We pass the concatenation of obtained vector and the previous 
        word outputted through the decoder LSTM, along with previous hidden states. The output of this LSTM is passed through the 
        linear layer and mapped to vocabulary length to output actual words. We take argmax of this vector to obtain the word 
        (the last step should be done in the main function).
        """
        weights = []
        for i in range(len(encoder_outputs)):
            print(decoder_hidden[0][0].shape)
            print(encoder_outputs[0].shape)
            weights.append(
                self.attn(
                    torch.cat(
                        (decoder_hidden[0][0], encoder_outputs[i]),
                        dim=1
                    )
                )
            )

        normalized_weights = F.softmax(torch.cat(weights, 1), 1)
        attn_applied = torch.bmm(normalized_weights.unsqueeze(1),
                                 encoder_outputs.view(1, -1, self.hidden_size))

        # if we are using embedding, use embedding of input here instead
        input_lstm = torch.cat((attn_applied[0], input[0]), dim=1)

        output, hidden = self.lstm(input_lstm.unsqueeze(0), decoder_hidden)

        output = self.final(output[0])

        return output, hidden, normalized_weights
