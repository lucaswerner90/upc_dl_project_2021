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
from torch.utils.data.dataset import Subset
import random

CONFIGURATION = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cwd = os.getcwd()

with open(os.path.join(cwd, 'config.json')) as f:
	CONFIGURATION = json.load(f)

hparams = CONFIGURATION['HPARAMS']
hparams['DEVICE'] = device

use_ViT_Enc = False
random_subsets = False
one_batch = False
do_training = False

print("Defined parameters:")
print("Use_Vit_Enc: ",use_ViT_Enc)
print("random_subsets: ",random_subsets)
print("one_batch: ",one_batch)
print("do_training: ",do_training)

def main():

	if use_ViT_Enc:
		print("It is using ViT encoder!!!!")
		transform = None
		with open("feature_extractor.pickle", "rb") as f:
			feature_extractor = pickle.load(f)  
#			feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

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
	
	if random_subsets==False:
		print("It is using fixed subsets!!!!")
		train_list = [*range(0,int(len(dataset)*0.8))]
		test_list = [*range(int(len(dataset)*0.8),len(dataset))]
		random.shuffle(train_list)
		train_split =  Subset(dataset,train_list)
		test_split =  Subset(dataset,test_list)
	else:
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


#	Si volem entrenar només amb un batch

	if one_batch==True:
		print("Only uses one batch for training!!!!!")
		list = [*range(0,30,5)]
		train_split =  Subset(dataset,list)
		train_loader = DataLoader(train_split, shuffle=True, batch_size=hparams['BATCH_SIZE'], collate_fn=CapsCollate(
		pad_idx=dataset.vocab.word_to_index['<PAD>'], batch_first=True))	
	else:
		print("We use the whole dataset for training!!!!!")

	if do_training == True:
		print("We are training")
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
	else:
		print("We are not training")
		model_epoch1 = ImageCaptioningModel(
			image_features_dim=hparams['IMAGE_FEATURES_DIM'],
			embed_size=hparams['EMBED_SIZE'],
			vocab = dataset.vocab,
			caption_max_length=hparams['MAX_LENGTH'],
			attention_dim=hparams['ATTENTION_DIM']
		).to(hparams['DEVICE'])

		epoch = 1
		filename = os.path.join('model','checkpoints','Epoch_'+str(epoch)+'_model_Pere_Transf_state.pth')
		model_state = torch.load(filename,map_location=torch.device('cpu'))

		model_epoch1.load_state_dict(model_state['model'])
		print("ok")

if __name__ == "__main__":
	main()
