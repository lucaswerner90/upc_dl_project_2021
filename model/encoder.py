import torch
from torch import nn

class Encoder(nn.Module):
	"""
	Encoder

	This class is the Encoder for the attention network that is similar to the vanilla encoders. 
	In the ‘__init__’ function we just store the parameters and create an LSTM layer. 
	In the forward function, we just pass the input through the LSTM with the provided hidden state. 
	The ‘init_hidden’ function is to be called before passing sentence through the LSTM to initialize the hidden state. 
	Note that, the hidden state has to be two vectors, as LSTMs have two vectors i.e. hidden activation and the memory cell, 
	in contrast with GRUs that is used in the PyTorch Tutorial. 
	The first dimension of the hidden state is 2 for bidirectional LSTM (as bidirectional LSTMs are two LSTMs, 
	one of which inputs the words in a forward manner, while the other one takes the words in reverse order) 
	the second dimension is the batch size, which we take here to be 1 and the last one is the desired output size. 
	Note that, I haven’t added any embedding for simplicity of the code.
	"""
	def __init__(self, input_size, hidden_size, bidirectional=True):
		super(Encoder,self).__init__()
		self.hidden_size = hidden_size
		self.input_size = input_size
		self.bidirectional = bidirectional

		self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=self.bidirectional)
	
	def forward(self,inputs, hidden):
		output, hidden = self.lstm(inputs.view(1,1,self.input_size), hidden)
		return output, hidden
	
	def init_hidden(self):
		return (torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size),
			torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size))