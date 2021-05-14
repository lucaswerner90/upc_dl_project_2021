import torch
from torch import nn

# https://medium.com/intel-student-ambassadors/implementing-attention-models-in-pytorch-f947034b3e66
from encoder import Encoder
from decoder import Decoder

class ImageCaptioningModel(nn.Module):
	def __init__(self):
		super(ImageCaptioningModel, self).__init__()
	
	def forward(self,X):
		return X




if __name__ == "__main__":
	"""
	For sake of testing the code, let’s create an encoder ‘c’ with an input size of 10 
	and output size of the encoder as 20 and make this LSTM bidirectional. 
	We pass a random vector of size 10, and pass the hidden state to get output, as vectors ‘a’ and ‘b’. 
	Note that, a.shape gives a tensor of size (1,1,40) as the LSTM is bidirectional; 
	two hidden states are obtained which are concatenated by PyTorch to obtain eventual hidden state 
	which explains the third dimension in the output which is 40 instead of 20. 
	
	Also, the hidden state ‘b’ is a tuple two vectors i.e. the activation and the memory cell. 
	The shape of each of these vectors is (2,1,20) (the first dimension is 2 due to the bidirectional nature of LSTMs).

	Now we create an attention-based decoder with hidden size = 40 if the encoder is bidirectional, 
	else 20 as we see that if they LSTM is bidirectional then outputs of LSTMs are concatenated, 
	25 as the LSTM output size and 30 as the vocab size. 
	We pass a vector [a,a] i.e. two similar encoder outputs, just for the sake of understanding through the decoder. 
	Also, assume that <SOS> token is all zeros. We see that shape of hidden state and output are (1,1,25) 
	while weights are (0.5, 0.5) as we pass the same vector ‘a’ through the network.
	Thus, using this model we could get text datasets, and use them for sequence to sequence modeling.
	"""
	bidirectional = False
	encod = Encoder(input_size=10,hidden_size=20,bidirectional=bidirectional)
	a,b = encod(torch.randn(10), encod.init_hidden())
	decod = Decoder(hidden_size=20*(1+bidirectional), output_size=25, vocab_size=30)
	y, z, w = decod(decod.init_hidden(), torch.cat((a,a)), torch.zeros(1,1, 30)) #Assuming <SOS> to be all zeros