import torch
from torch import nn
from model.attention import Attention

#from encoder import Encoder
from model.encoder import Encoder
from model.decoder import Decoder
from dataset.vocabulary import Vocabulary

class ImageCaptioningModel(nn.Module):
	def __init__(self, image_features_dim:int,embed_size:int, vocab_size:int, caption_max_length:int,device):
		super(ImageCaptioningModel, self).__init__()
		self.encoder = Encoder(device)
		self.attention = Attention(image_features_dim=image_features_dim, decoder_hidden_state_dim=embed_size, attention_dim=256,device=device)
		self.decoder = Decoder(vocab_size, embed_size, embed_size,device=device)
		self.caption_max_length = caption_max_length
	
	def forward(self, images, captions,device, initial_hidden=None):
		"""
		images => (batch_size, channels, W, H)
		captions => (batch_size, captions_length)
		initial_hidden => (seq_len = 1, bsz_size, captions_length)
		"""

		images_features = self.encoder(images)
		bsz, *_ = images_features.shape
		hidden = self.decoder.init_hidden(images.shape[0]).to(device) if initial_hidden == None else initial_hidden
		timesteps = captions.shape[-1]-1
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
	
	def inference(self, image, vocab:Vocabulary):
		image_features = self.encoder(image)
		hidden = self.decoder.init_hidden(image.shape[0]).to(device)
		alphas, weighted_features = self.attention.forward(image_features, hidden)
		word = torch.IntTensor(vocab.word_to_index['<START>']).to(device)
		sentence = [word.tolist()]
		
		for i in range(self.caption_max_length):
			predictions_t, hidden = self.decoder.forward(weighted_features, word, hidden)
			word = torch.argmax(predictions_t, dim=-1)
			sentence.append(word.tolist())
			if word==vocab.word_to_index['<END>']:
				break
		return sentence