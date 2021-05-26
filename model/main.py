import torch
from torch import nn
from model.attention import Attention

#from encoder import Encoder
from model.encoder import Encoder
from model.decoder import Decoder
from dataset.vocabulary import Vocabulary

class ImageCaptioningModel(nn.Module):
	def __init__(self, image_features_dim:int,embed_size:int, vocab:Vocabulary, caption_max_length:int,attention_dim):
		super(ImageCaptioningModel, self).__init__()
		self.vocab = vocab
		self.vocab_size = len(self.vocab.word_to_index)
		self.encoder = Encoder()
		self.attention = Attention(image_features_dim=image_features_dim, decoder_hidden_state_dim=embed_size, attention_dim=attention_dim)
		self.decoder = Decoder(image_features_dim=image_features_dim,vocab_size=self.vocab_size,hidden_size=embed_size,embed_size=embed_size)
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
		timesteps = captions.shape[-1]-1
		attention_weights = []
		predictions = torch.zeros(size=(timesteps,bsz,self.vocab_size))
		for i in range(timesteps):
			alphas, weighted_features = self.attention.forward(images_features, hidden)
			words = captions[:,i:i+1]
			predictions_t, hidden = self.decoder.forward(weighted_features, words, hidden)
			predictions[i] = predictions_t
			attention_weights.append(alphas)
		return predictions.permute(1,2,0), attention_weights
	
	def inference(self, image, vocab:Vocabulary):
		image_features = self.encoder(image)
		hidden = self.decoder.init_hidden(image.shape[0])
		
		if torch.cuda.is_available():
			word = torch.cuda.IntTensor([vocab.word_to_index['<START>']])
		else:
			word = torch.tensor(vocab.word_to_index['<START>'])
		sentence = [word.tolist()]
		
		for i in range(self.caption_max_length):
			alphas, weighted_features = self.attention.forward(image_features, hidden)
			predictions_t, hidden = self.decoder.forward(weighted_features, word, hidden)
			word = torch.argmax(predictions_t, dim=-1)
			sentence.append(word.tolist())
			
			if word[0].tolist()==vocab.word_to_index['<END>']:
				break
		return sentence