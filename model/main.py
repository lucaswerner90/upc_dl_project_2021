import os
import torch
from torch import nn

#from encoder import Encoder
from model.encoder import Encoder_VGG16, Encoder_ViT_Pretrained
from dataset.vocabulary import Vocabulary
from model.transformer.decoder import TransformerDecoder

class ImageCaptioningModel(nn.Module):
	def __init__(self, image_features_dim:int,embed_size:int, vocab:Vocabulary, caption_max_length:int,decoder_num_layers=1):
		super(ImageCaptioningModel, self).__init__()
		self.vocab = vocab
		self.vocab_size = len(self.vocab.word_to_index)
		self.encoder = Encoder_VGG16()
		self.decoder = TransformerDecoder(image_features_dim,vocab_size=self.vocab_size,embed_size=embed_size,num_layers=decoder_num_layers)
		self.caption_max_length = caption_max_length

	def forward(self, images, captions):
		"""
		images => (batch_size, channels, W, H)
		captions => (batch_size, captions_length)
		"""
		images_features = self.encoder.forward(images)
		predictions = self.decoder.forward(images_features, captions)
		return predictions

	def generate(self,image):
		image_features = self.encoder.forward(image)
		
		output = torch.LongTensor([self.vocab.word_to_index['<START>']])\
			.expand(image_features.size(0),1).to(image_features.device)
		
		for _ in range(self.caption_max_length):
			next_word=self.decoder.forward(image_features,output).argmax(-1)[:,-1]
			if next_word.item()==self.vocab.word_to_index['<END>']:
				break
			output = torch.cat([output,next_word.unsqueeze(0)],dim=1)
		return output.squeeze(0)
			

	def save_model(self,epoch):
		string_fn='transformer_model_epoch_'+str(epoch)+'.pth'
		filename = os.path.join('model','trained_models',string_fn)
		dirname = os.path.join('model','trained_models')
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		model_state = {
		'epoch':epoch,
		'model':self.state_dict()
		}
		torch.save(model_state, filename)

class ViTImageCaptioningModel(nn.Module):
	def __init__(self,embed_size:int, vocab:Vocabulary, caption_max_length:int,decoder_num_layers=1):
		super(ViTImageCaptioningModel, self).__init__()
		self.vocab = vocab
		self.vocab_size = len(self.vocab.word_to_index)
		self.encoder =  Encoder_ViT_Pretrained()
		image_features_dim = self.encoder.pretrained_model.encoder.config.hidden_size
		self.decoder = TransformerDecoder(image_features_dim,vocab_size=self.vocab_size,embed_size=embed_size,num_layers=decoder_num_layers)
		self.caption_max_length = caption_max_length

	def forward(self, images, captions):
		"""
		images => (batch_size, channels, W, H)
		captions => (batch_size, captions_length)
		"""
		images_features = self.encoder.forward(images)
		predictions = self.decoder.forward(images_features, captions)
		return predictions

	def generate(self,image):
		image_features = self.encoder.forward(image)
		
		output = torch.LongTensor([self.vocab.word_to_index['<START>']])\
			.expand(image_features.size(0),1).to(image_features.device)
		
		for _ in range(self.caption_max_length):
			next_word=self.decoder.forward(image_features,output).argmax(-1)[:,-1]
			if next_word.item()==self.vocab.word_to_index['<END>']:
				break
			output = torch.cat([output,next_word.unsqueeze(0)],dim=1)
		return output.squeeze(0)
			

	def save_model(self,epoch):
		string_fn='ViT_model_epoch_'+str(epoch)+'.pth'
		filename = os.path.join('model','trained_models',string_fn)
		dirname = os.path.join('model','trained_models')
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		model_state = {
		'epoch':epoch,
		'model':self.state_dict()
		}
		torch.save(model_state, filename)
