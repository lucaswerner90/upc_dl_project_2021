import torch
from torch import nn

class Encoder(nn.Module):
	def __init__(self, input_size, hidden_size, bidirectional=True):
		super(Encoder,self).__init__()
		self.hidden_size = hidden_size
		self.input_size = input_size
		self.bidirectional = bidirectional

		self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, bidirectional=self.bidirectional, num_layers = 1)
	
	def forward(self, inputs, hidden):
		"""
		For the LSTM:

		inputs: (seq_len, batch, input_size)
		hidden: tuple (h_0, c_0) => 	(num_layers * num_directions, batch, hidden_size) ,
										(num_layers * num_directions, batch, hidden_size)
		"""
		output, hidden_state = self.lstm(inputs.view(1,1,self.input_size), hidden)

		# output: (seq_len, batch, num_directions * hidden_size)
		# hidden: (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t = seq_len.
		return output, hidden_state
	
	def init_hidden(self):
		# (num_layers * num_directions, batch, hidden_size), (num_layers * num_directions, batch, hidden_size)
		return (torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size),
			torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size))


if __name__ == "__main__":
	rnn = nn.LSTM(input_size=10, hidden_size=20, num_layers=2)
	input = torch.randn(5, 3, 10)
	h0 = torch.randn(2, 3, 20)
	c0 = torch.randn(2, 3, 20)
	output, (hn, cn) = rnn(input, (h0, c0))
	print(output) # (5,3,20)
	print(hn.shape,cn.shape) # (2,3,20)