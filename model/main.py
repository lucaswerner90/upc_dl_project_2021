import torch
from torch import nn

from encoder import Encoder
from decoder import Decoder

class ImageCaptioningModel(nn.Module):
	def __init__(self, input_size, embed_size, bidirectional_encoder=False, vocab_size = 50):
		super(ImageCaptioningModel, self).__init__()
		self.encoder = Encoder(vocab_size, hidden_size)
		self.decoder = Decoder(vocab_size, embed_size, hidden_size)
	
	def forward(self,image, caption):
		pass


if __name__ == "__main__":
	vocab_size = 30
	hidden_size = 20
	embed_size = 10
	encod = Encoder(vocab_size, hidden_size)
	decod = Decoder(vocab_size, embed_size, hidden_size)