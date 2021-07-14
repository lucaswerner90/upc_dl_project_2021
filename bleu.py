import numpy as np
import torch
import os
import sacrebleu
from torchvision import transforms
from dataset.main import Flickr8kDataset
from torch.utils.data import DataLoader
from model.main import ImageCaptioningModel, ViTImageCaptioningModel
from model.visualization import Visualization
from transformers import ViTFeatureExtractor
from dataset.caps_collate import CapsCollate
from PIL import Image
from train import split_subsets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

generate_for_vit = False

# Load the model
if not generate_for_vit:
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Resize((224, 224)),
		# The normalize parameters depends on the model we're gonna use
		# If we apply transfer learning from a model that used ImageNet, then
		# we should use the ImageNet values to normalize the dataset.
		# Otherwise we could just normalize the values between -1 and 1 using the
		# standard mean and standard deviation
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
			0.229, 0.224, 0.225]),
	])
	dataset = Flickr8kDataset(dataset_folder='/local/Datasets/8KFlicker/archive', transform=transform,
							  reduce=True, vocab_max_size=5000, feature_extractor=None)
else:
	feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
	dataset = Flickr8kDataset(dataset_folder='/local/Datasets/8KFlicker/archive', reduce=True,
							  vocab_max_size=5000, feature_extractor=feature_extractor)

# Create the model
if not generate_for_vit:
	model = ImageCaptioningModel(
		image_features_dim=512,
		embed_size=256,
		vocab=dataset.vocab,
		caption_max_length=28,
	).to(device)
else:
	model = ViTImageCaptioningModel(
		embed_size=256,
		vocab=dataset.vocab,
		caption_max_length=28,
	).to(device)

if generate_for_vit:
	checkpoint_file = os.path.join(os.getcwd(), 'model', 'trained_models', 'ViT_model_epoch_35.pth')
else:
	checkpoint_file = os.path.join(os.getcwd(),'model','trained_models','transformer_model_epoch_85.pth')

checkpoint = torch.load(checkpoint_file, map_location=torch.device('cuda'))

# BE CAREFUL WITH THIS!
# If you save a model that YOU KNOW the architecture,
# load it using strict=True
model.load_state_dict(checkpoint['model'], strict=True)
model.eval()


np.random.seed(42)

train_split, test_split = split_subsets(dataset,all_captions=True)

test_loader = DataLoader(test_split, shuffle=False, batch_size=1, collate_fn=CapsCollate(
		pad_idx=dataset.vocab.word_to_index['<PAD>'], batch_first=True))

refs=[]
preds=[]
for i,batch in enumerate(test_loader):
	img, target= batch
	

	refs.append(model.vocab.generate_caption(target[:,1:-1].squeeze(0)).strip().split())
	caption = model.generate(img)
	preds.append(model.vocab.generate_caption(model.generate(img)[1:]).strip().split())
refs=[refs]

bleu = sacrebleu.corpus_bleu(preds,refs)
print(bleu.score)
