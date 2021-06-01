import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from dataset.main import Flickr8kDataset
from dataset.caps_collate import CapsCollate
from model.main import ImageCaptioningModel
from train import train
import json

CONFIGURATION = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cwd = os.getcwd()

with open(os.path.join(cwd,'config.json')) as f:
	CONFIGURATION = json.load(f)
	
hparams = CONFIGURATION['HPARAMS']
hparams['device'] = device

def main():
	# Choosing image transformations
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Resize((hparams['IMAGE_SIZE'],hparams['IMAGE_SIZE'])),
		# The normalize parameters depends on the model we're gonna use
		# If we apply transfer learning from a model that used ImageNet, then
		# we should use the ImageNet values to normalize the dataset.
		# Otherwise we could just normalize the values between -1 and 1 using the 
		# standard mean and standard deviation
		transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
	])

    train_split = Flickr8kDataset(transform=transform,reduce=True, vocab_max_size=hparams['VOCAB_SIZE'],split='train')
    test_split = Flickr8kDataset(transform=transform,reduce=True, vocab_max_size=hparams['VOCAB_SIZE'],split='test')
	eval_split = Flickr8kDataset(transform=transform,reduce=True, vocab_max_size=hparams['VOCAB_SIZE'],split='eval')


	# Test the dataloader
	model = ImageCaptioningModel(
		embed_size=hparams['EMBED_SIZE'],
		image_features_dim=512,
		vocab_size=len(train_split.vocab.word_to_index),
		attention_dim=hparams['ATTENTION_DIM'],
		caption_max_length=hparams['MAX_LENGTH']
	)


	train_loader = DataLoader(train_split, shuffle=True, batch_size=hparams['BATCH_SIZE'], collate_fn=CapsCollate(pad_idx=train_split.vocab.word_to_index['<PAD>'],batch_first=True))
	
	## En el collate_fn he de posar-hi train_split??
	eval_loader = DataLoader(eval_split,batch_size=hparams['BATCH_SIZE'], collate_fn=CapsCollate(pad_idx=train_split.vocab.word_to_index['<PAD>'],batch_first=True))
	test_loader = DataLoader(test_split,batch_size=hparams['BATCH_SIZE'], collate_fn=CapsCollate(pad_idx=train_split.vocab.word_to_index['<PAD>'],batch_first=True))
	
	optimizer = optim.Adam(model.parameters(),lr=hparams['LEARNING_RATE'])
	
	## En el ignore_index de posar-hi train_split??
	criterion = nn.CrossEntropyLoss(ignore_index=train_split.vocab.word_to_index['<PAD>'])

	train(
		num_epochs=hparams['NUM_EPOCHS'],
		model=model,
		train_loader=train_loader,
		test_loader=test_loader,
		optimizer= optimizer,
		criterion=criterion,
		device=hparams['device'],
		log_interval=hparams['LOG_INTERVAL']
	)

if __name__ == "__main__":
	main()
	