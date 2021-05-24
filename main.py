import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from dataset import Flickr8kDataset
from caps_collate import CapsCollate
from model.main import ImageCaptioningModel
from train import train


def main():
	data_path = "./dataset"
	#data_path = "/local/Datasets/8KFlicker/archive"
	# Setting hyperparameters
	hparams = {
		'BATCH_SIZE': 256,
		# NUM_WORKER = 4,
		'VOCAB_SIZE': 5000,
		'reduce': True,
		'NUM_EPOCHS': 10,
		'max_length': 25,
		'embed_size': 300,
		'attention_dim': 256,
		'encoder_dim': 512,
		'decoder_dim': 512,
		'learning_rate': 1e-3,
		'image_size': 224,
		'bidirectional_encoder': False,
		'196_emb': False,
		'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
		'log_interval': 100,
	}
	# Choosing image transformations
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Resize(hparams['image_size']),
		transforms.CenterCrop(hparams['image_size']),
		# The normalize parameters depends on the model we're gonna use
		# If we apply transfer learning from a model that used ImageNet, then
		# we should use the ImageNet values to normalize the dataset
		# Otherwise we could just normalize the values between -1 and 1 using the
		# standard mean and standard deviation
		# transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	])
	# Initializing & splitting dataset
	dataset = Flickr8kDataset(dataset_folder=data_path,transform=transform,reduce=hparams['reduce'],max_size=hparams['VOCAB_SIZE'])
	train_split, test_split = random_split(dataset,[32364,8091]) #80% train, 20% test

	# Initializing DataLoaders
	train_loader = DataLoader(train_split,batch_size=hparams['BATCH_SIZE'], collate_fn=CapsCollate(pad_idx=dataset.vocab.word_to_index['<PAD>'],batch_first=True))
	test_loader = DataLoader(test_split,batch_size=hparams['BATCH_SIZE'], collate_fn=CapsCollate(pad_idx=dataset.vocab.word_to_index['<PAD>'],batch_first=True))
	
	# instantiating model
	model = ImageCaptioningModel(embed_size=hparams['embed_size'],bidirectional_encoder=hparams['bidirectional_encoder'],num_emb_196=hparams['num_emb_196'],vocab_size=len(dataset.vocab.word_to_index),max_length=hparams['max_length'])

	# select optimizer and criterion
	optimizer = optim.Adam(model.parameters(),lr=hparams['learning_rate'])
	criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.word_to_index['<PAD>'])

	# LET'S TRAIN!!!
	train(num_epochs=hparams['NUM_EPOCHS'],model=model,train_loader=train_loader,test_loader=test_loader,optimizer=optimizer,criterion=criterion,device=hparams['device'],log_interval=['log_interval'])

if __name__ == "__main__":
	print(
	"""
	This script is gonna execute the whole program.
	That means, creating the dataset, training the model and executing the model
	"""
	)
	main()
