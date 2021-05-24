import torch
from torch import nn
from model.attention import Attention

#from encoder import Encoder
from model.encoder import Encoder
from model.decoder import Decoder

class ImageCaptioningModel(nn.Module):
	def __init__(self, embed_size:int, vocab_size:int, caption_max_length:int):
		super(ImageCaptioningModel, self).__init__()
		self.encoder = Encoder(embed_size)
		self.attention = Attention(encoder_dim=embed_size, decoder_dim=embed_size, attention_dim=256)
		self.decoder = Decoder(vocab_size, embed_size, embed_size)
		self.caption_max_length = caption_max_length
	
	def forward(self, images, captions):
		"""
		images => (batch_size, channels, W, H)
		captions => (batch_size, captions_length)
		"""
		images_features = self.encoder(images)
		hidden = self.decoder.init_hidden(images.shape[0])

		for i in range(captions.shape[-1]):
			alphas, weighted_features = self.attention.forward(images_features, hidden)
			predictions, hidden = self.decoder.forward(weighted_features,captions[:,i:i+1], hidden)
			print(captions[:,i:i+1])
			print(predictions)
			print(alphas)
		return predictions, hidden, alphas