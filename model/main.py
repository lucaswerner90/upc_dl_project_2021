import torch
from torch import nn
from model.attention import Attention

#from encoder import Encoder
from model.encoder import Encoder
from model.decoder import Decoder

class ImageCaptioningModel(nn.Module):
	def __init__(self, embed_size:int, vocab_size:int, caption_max_length:int):
		super(ImageCaptioningModel, self).__init__()
		self.encoder = Encoder()
		self.attention = Attention(image_features_dim=512, decoder_hidden_state_dim=embed_size, attention_dim=256)
		self.decoder = Decoder(vocab_size, embed_size, embed_size)
		self.caption_max_length = caption_max_length
	
	def forward(self, images, captions, initial_hidden=None):
		"""
		images => (batch_size, channels, W, H)
		captions => (batch_size, captions_length)
		initial_hidden => (seq_len = 1, bsz_size, captions_length)
		"""

		images_features = self.encoder(images)
		bsz, *_ = images_features.shape
		hidden = self.decoder.init_hidden(images.shape[0]) if initial_hidden == None else initial_hidden
		timesteps = captions.shape[-1]
		predicted_captions = []
		attention_weights = []

		for i in range(timesteps):
			alphas, weighted_features = self.attention.forward(images_features, hidden)
			words = captions[:,i:i+1]
			predictions_t, hidden = self.decoder.forward(weighted_features, words, hidden)
			predicted_captions.append(predictions_t)
			attention_weights.append(alphas)
		output = torch.cat(predicted_captions).reshape(bsz, -1, timesteps)
		return output, attention_weights