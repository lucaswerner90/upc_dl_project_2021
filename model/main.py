import torch
from torch import nn
from model.attention import Attention

#from encoder import Encoder
from model.encoder import Encoder_VGG16
from model.decoder import Decoder
from dataset.vocabulary import Vocabulary
import math

### S'ha de revisar això  

class PositionalEncoding(nn.Module):

	def __init__(self, d_model, dropout=0.1, max_len=5000):   ## d_mode es la dimensió dels embeddings
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		r"""Inputs of forward function
		Args:
			x: the sequence fed to the positional encoder model (required).
		Shape:
			x: [sequence length, batch size, embed dim]
			output: [sequence length, batch size, embed dim]
		Examples:
			>>> output = pos_encoder(x)
		"""

		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)


class ImageCaptioningModel(nn.Module):
	def __init__(self, image_features_dim:int,embed_size:int, vocab:Vocabulary, caption_max_length:int,attention_dim):
		super(ImageCaptioningModel, self).__init__()
		self.embed_size = embed_size
		self.vocab = vocab
		self.vocab_size = len(self.vocab.word_to_index)
		self.encoder = Encoder_VGG16()
		self.adap_encoder = nn.Linear(image_features_dim,embed_size)

		self.embed_layer = nn.Embedding(self.vocab_size, embed_size)
		
		self.tgt_mask = None    ## OJU!!
		self.pos_encoder = PositionalEncoding(embed_size, dropout=0.1)

		decoderlayers = nn.TransformerDecoderLayer(embed_size,nhead=4,dropout=0.1)
		self.decoder = nn.TransformerDecoder(decoderlayers,num_layers=4)
		self.caption_max_length = caption_max_length
		self.linear = nn.Linear(embed_size,self.vocab_size)
	

	## Me l'he d emirar aquest subsequent mask ##
	def _generate_square_subsequent_mask(self, sz):
		mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)  # Lower triangular matrix with ones.
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask

## M'ho he de mirar també
	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)


	def forward(self, images, captions, has_mask=True):
		"""
		images => (batch_size, channels, W, H)
		captions => (batch_size, captions_length)
		initial_hidden => (seq_len = 1, bsz_size, captions_length)
		"""
		tgt_seq_len = captions.shape[1]    #  aquí hi vull posar la longitud de la seqüencia. No sé si està bé.  Es podria utilitzar el len(captions)
		# No tinc om
		if has_mask:
			device = images.device
			if self.tgt_mask is None or self.tgt_mask.size(0) != tgt_seq_len:
				mask = self._generate_square_subsequent_mask(tgt_seq_len).to(device)    #  Es podria utilitzar el len(captions)
				self.tgt_mask = mask
		else:
			self.tgt_mask = None


		images_features = self.encoder(images)
		bsz, *_ = images_features.shape
		images_features = images_features.permute(2,0,1)

		images_features = self.adap_encoder(images_features)

		embeddings = self.embed_layer(captions)
		embeddings = embeddings * math.sqrt(self.embed_size)
		pos_embeddings = self.pos_encoder(embeddings)
		pos_embeddings = pos_embeddings.permute(1,0,2)

		## La dimensió de pos-embeddings hauria de ser  Seq_len,batch,embed_size
		x = self.decoder(pos_embeddings,images_features)
		x = self.linear(x)

		return x


#
# 		hidden = self.decoder.init_hidden(images.shape[0]) if initial_hidden == None else initial_hidden

#		timesteps = captions.shape[-1]-1
#		attention_weights = []
#		predictions = torch.zeros(size=(timesteps,bsz,self.vocab_size))
#		for i in range(timesteps):
#			alphas, weighted_features = self.attention.forward(images_features, hidden)
#			words = captions[:,i:i+1]
#			predictions_t, hidden = self.decoder.forward(weighted_features, words, hidden)
#			predictions[i] = predictions_t
#			attention_weights.append(alphas)
		return predictions.permute(1,2,0), attention_weights
	
	def inference(self, image):
		image_features = self.encoder(image)
		hidden = self.decoder.init_hidden(image.shape[0])
		
		if torch.cuda.is_available():
			word = torch.cuda.IntTensor([self.vocab.word_to_index['<START>']])
		else:
			word = torch.tensor(self.vocab.word_to_index['<START>'])
		sentence = [word.item()]
		attention_weights=[]
		for i in range(self.caption_max_length):
			alphas, weighted_features = self.attention.forward(image_features, hidden)
			predictions_t, hidden = self.decoder.forward(weighted_features, word, hidden)
			word = torch.argmax(predictions_t, dim=-1)
			if word[0].item()==self.vocab.word_to_index['<END>']:
				break
			sentence.append(word.item())
			attention_weights.append(alphas)
		if torch.cuda.is_available():
			sentence=torch.cuda.IntTensor(sentence[1:])
		else:
			sentence=torch.tensor(sentence[1:])
		sentence=self.vocab.generate_caption(sentence)
		return sentence, attention_weights