import torch
import os
from torchvision import transforms
from dataset.main import Flickr8kDataset
from model.main import ImageCaptioningModel, ViTImageCaptioningModel
from model.visualization import Visualization
from transformers import ViTFeatureExtractor
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import cv2 

generate_for_vit = True

# Load the model
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize((224, 224)),
	# The normalize parameters depends on the model we're gonna use
	# If we apply transfer learning from a model that used ImageNet, then
	# we should use the ImageNet values to normalize the dataset.
	# Otherwise we could just normalize the values between -1 and 1 using the 
	# standard mean and standard deviation
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
dataset = Flickr8kDataset(dataset_folder='data', reduce=True, vocab_max_size=5000, feature_extractor=feature_extractor)
# dataset = Flickr8kDataset(dataset_folder='data', transform=transform, reduce=True, vocab_max_size=5000, feature_extractor=feature_extractor)

# Create the model
# model = ImageCaptioningModel(
# 	image_features_dim=512,
# 	embed_size=256,
# 	vocab = dataset.vocab,
# 	caption_max_length=28,
# ).to(device)
model = ViTImageCaptioningModel(
	embed_size=256,
	vocab = dataset.vocab,
	caption_max_length=28,
).to(device)

checkpoint_file = os.path.join(os.getcwd(),'model','trained_models','vit_5.pth')
# checkpoint_file = os.path.join(os.getcwd(),'model','trained_models','transformer_25.pth')
checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))

# BE CAREFUL WITH THIS!
# If you save a model that YOU KNOW the architecture, 
# load it using strict=True
# model.load_state_dict(torch.load(model_file), strict=True)
model.load_state_dict(checkpoint['model'], strict=True)
model.eval()

images_list = ['0.jpg','1.jpg','2.jpg','3.jpg','4.jpg', 'group_image.jpg']


cam = cv2.VideoCapture(0)
while True:
	ret_val, img = cam.read()
	frame = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
	frame = cv2.flip(frame, 1)
	image = feature_extractor(images=img, return_tensors="pt").data['pixel_values'].squeeze()
	caption = model.generate(image.unsqueeze(0))
	cv2.putText(frame,model.vocab.generate_phrase(caption), (20,20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255), 2)
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) == 27: 
		break  # esc to quit
cv2.destroyAllWindows()
# for index,path in enumerate(images_list):
# 	image = Image.open(path)
# 	# image = transform(image)
# 	image = feature_extractor(images=image, return_tensors="pt").data['pixel_values'].squeeze()
# 	caption = model.generate(image.unsqueeze(0))
# 	Visualization.show_image(image, model.vocab.generate_phrase(caption), f'vit_{index}.png')
# 	Visualization.show_image(image, model.vocab.generate_phrase(caption), f'transformer_{index}.png')
