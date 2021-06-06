import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pandas as pd

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

    ### dataset = Flickr8kDataset(transform=transform,reduce=True, vocab_max_size=hparams['VOCAB_SIZE'])
	#	train_split, test_split = random_split(dataset,[32364,8091]) #80% train, 20% test
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


def main2():
	# Choosing image transformations
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Resize((hparams['IMAGE_SIZE'],hparams['IMAGE_SIZE'])),
		# The normalize parameters depends on the model we're gonna use
		# If we apply transfer learning from a model that used ImageNet, then
		# we should use the ImageNet values to normalize the dataset.
		# Otherwise we could just normalize the values between -1 and 1 using the 
		# standard mean and standard deviation
		transforms.Normalize(mean=hparams['IMAGE_NET_MEANS'],std=hparams['IMAGE_NET_STDS']),
	])

	# Create captions files for train and test
	dataset_df = pd.read_csv(open(os.path.join('data','captions.txt'),'r'))

	for set in ['train', 'test']:
		if set == 'train':
			# Open file with the images for train
			imgs_file = open(os.path.join('data','Flickr_8k.trainImages.txt'), "r")
		if set == 'test':
			# Open file with the images for test
			imgs_file = open(os.path.join('data','Flickr_8k.testImages.txt'), "r")

	    # Load train images file names to a list
		imgs = []
		for linea in imgs_file:
			imgs.append(linea.rstrip('\n'))
		imgs_file.close()
            
		# Convert the dataframe to list
		#	llista = list(zip(self.dataframe['image'],self.dataframe['caption']))
		llista = dataset_df.values.tolist()
    	# Create a list with the image and captions of the images requested (train, test or eval)
		c=[]
		for a,b in llista:
			if a in imgs:
				c.append([a,b])
		# Convert the list to a dataframe
		new_dataframe = pd.DataFrame(c, columns = ['image','caption'])
		# Write the dataframe to file
		if set == 'train':
			new_dataframe.to_csv(os.path.join('data','train_captions.txt'),index=False)
		if set == 'test':
			new_dataframe.to_csv(os.path.join('data','test_captions.txt'),index=False)

#	train_split = Flickr8kDataset(transform=transform,reduce=True, vocab_max_size=hparams['VOCAB_SIZE'], split='train')
	dataset = Flickr8kDataset(transform=transform,reduce=True, vocab_max_size=hparams['VOCAB_SIZE'],split='test')

    #dataset = Flickr8kDataset(transform=transform,reduce=True, vocab_max_size=hparams['VOCAB_SIZE'])

	# Test the dataloader
	model = ImageCaptioningModel(
		embed_size=hparams['EMBED_SIZE'],
		image_features_dim=512,
		vocab_size=len(dataset.vocab.word_to_index),
		attention_dim=hparams['ATTENTION_DIM'],
		caption_max_length=hparams['MAX_LENGTH']
	)

#	train_split, test_split = random_split(dataset,[32364,8091]) #80% train, 20% test
#	train_loader = DataLoader(train_split, shuffle=True, batch_size=hparams['BATCH_SIZE'], collate_fn=CapsCollate(pad_idx=dataset.vocab.word_to_index['<PAD>'],batch_first=True))
	
	train_loader = DataLoader(dataset,batch_size=hparams['BATCH_SIZE'], collate_fn=CapsCollate(pad_idx=dataset.vocab.word_to_index['<PAD>'],batch_first=True))
	test_loader = DataLoader(dataset,batch_size=hparams['BATCH_SIZE'], collate_fn=CapsCollate(pad_idx=dataset.vocab.word_to_index['<PAD>'],batch_first=True))
	
	optimizer = optim.Adam(model.parameters(),lr=hparams['LEARNING_RATE'])
	criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.word_to_index['<PAD>'])

	model.encoder.eval()
	a = model.encoder(dataset[0][0].unsqueeze(0))

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
	main2()
	