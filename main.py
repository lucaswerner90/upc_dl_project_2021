import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms

from dataset.main import Flickr8kDataset
from dataset.caps_collate import CapsCollate
from model.main import ImageCaptioningModel,ViTImageCaptioningModel
from train import train, split_subsets
import json
import pickle

CONFIGURATION = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cwd = os.getcwd()

with open(os.path.join(cwd, 'config.json')) as f:
	CONFIGURATION = json.load(f)

hparams = CONFIGURATION['HPARAMS']
hparams['DEVICE'] = device

use_ViT_Enc = True

def main():

	if use_ViT_Enc:
		transform = None
		with open("feature_extractor.pickle", "rb") as f:
			feature_extractor = pickle.load(f)  
	else:
		feature_extractor = None
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

	dataset = Flickr8kDataset(dataset_folder=CONFIGURATION['DATASET_FOLDER'], transform=transform,
								reduce=hparams['REDUCE_VOCAB'], vocab_max_size=hparams['VOCAB_SIZE'],feature_extractor=feature_extractor)

	if use_ViT_Enc:
		model = ViTImageCaptioningModel(
			image_features_dim=hparams['IMAGE_FEATURES_DIM'],
			embed_size=hparams['EMBED_SIZE'],
			vocab = dataset.vocab,
			caption_max_length=hparams['MAX_LENGTH'],
			attention_dim=hparams['ATTENTION_DIM']
		).to(hparams['DEVICE'])
	else:
		model = ImageCaptioningModel(
			image_features_dim=hparams['IMAGE_FEATURES_DIM'],
			embed_size=hparams['EMBED_SIZE'],
			vocab = dataset.vocab,
			caption_max_length=hparams['MAX_LENGTH'],
			attention_dim=hparams['ATTENTION_DIM']
		).to(hparams['DEVICE'])

	## Perform the split of the dataset
	
	train_split, test_split = split_subsets(dataset,all_captions=True)
	
#  Aquesta línia em provocava errors quan corria el codi desde GPU
#	if (torch.cuda.is_available()):
#		torch.set_default_tensor_type('torch.cuda.FloatTensor')

	train_loader = DataLoader(train_split, shuffle=True, batch_size=hparams['BATCH_SIZE'], collate_fn=CapsCollate(
		pad_idx=dataset.vocab.word_to_index['<PAD>'], batch_first=True))
	test_loader = DataLoader(test_split, batch_size=hparams['BATCH_SIZE'], collate_fn=CapsCollate(
		pad_idx=dataset.vocab.word_to_index['<PAD>'], batch_first=True))

	optimizer = optim.Adam(model.parameters(), lr=hparams['LEARNING_RATE'])

	criterion = nn.CrossEntropyLoss(
		ignore_index=dataset.vocab.word_to_index['<PAD>'])

	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
#	scheduler = optim.lr_scheduler.StepLR
#	scheduler = None
#	print(optimizer.param_groups[0]['lr'])

	train(
		num_epochs=hparams['NUM_EPOCHS'],
		model=model,
		train_loader=train_loader,
		test_loader=test_loader,
		optimizer=optimizer,
		criterion=criterion,
		device=hparams['DEVICE'],
		log_interval=hparams['LOG_INTERVAL'],
		vocab=dataset.vocab,
		scheduler=scheduler
	)


if __name__ == "__main__":
	main()
