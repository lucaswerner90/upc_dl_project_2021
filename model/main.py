import torch
from torch import nn

from encoder import Encoder
from decoder import Decoder

class ImageCaptioningModel(nn.Module):
	def __init__(self, input_size, bidirectional_encoder=False, vocab_size = 50):
		super(ImageCaptioningModel, self).__init__()
		self.encoder = Encoder(input_size=input_size,hidden_size=20, bidirectional=bidirectional_encoder)
		self.decoder = Decoder(hidden_size=20*(1+bidirectional_encoder), output_size=25, vocab_size=vocab_size)
	
	def forward(self,image, caption):
		pass


if __name__ == "__main__":
	bidirectional = False
	vocab_size = 30
	encod = Encoder(input_size=10,hidden_size=20,bidirectional=bidirectional)
	decod = Decoder(hidden_size=20*(1+bidirectional), output_size=25, vocab_size=vocab_size)
	encoder_output, hidden_state = encod(torch.randn(10), encod.init_hidden())

	output, hidden, normalized_weights = decod(decod.init_hidden(), encoder_output, torch.zeros(1,1, vocab_size)) # Assuming <SOS> to be all zeros
	print(output.shape)