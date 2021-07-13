import numpy as np
import torch
import os
from torchvision import transforms
from dataset.main import Flickr8kDataset
from model.main import ImageCaptioningModel, ViTImageCaptioningModel
from model.visualization import Visualization
from transformers import ViTFeatureExtractor
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


generate_for_vit = True

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
    dataset = Flickr8kDataset(dataset_folder='data', transform=transform,
                              reduce=True, vocab_max_size=5000, feature_extractor=None)
else:
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    dataset = Flickr8kDataset(dataset_folder='data', reduce=True,
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
	checkpoint_file = os.path.join(os.getcwd(), 'model', 'trained_models', 'vit_5.pth')
else:
	checkpoint_file = os.path.join(os.getcwd(),'model','trained_models','transformer_25.pth')

checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))

# BE CAREFUL WITH THIS!
# If you save a model that YOU KNOW the architecture,
# load it using strict=True
model.load_state_dict(checkpoint['model'], strict=True)
model.eval()


np.random.seed(42)
images_selected = np.random.randint(0,len(dataset),size=50)
images_list = [dataset[i][0] for i in images_selected ]
# images_list = ['0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg', 'group_image.jpg']

for index,image in enumerate(images_list):

	caption = model.generate(image.unsqueeze(0))
	if generate_for_vit:
		filename = os.path.join(os.getcwd(),'generated_images','vit',f'vit_{index}.png')
		Visualization.show_image(image, model.vocab.generate_phrase(caption), filename, True)
	else:
		filename = os.path.join(os.getcwd(),'generated_images','transformer',f'transformer_{index}.png')
		Visualization.show_image(image, model.vocab.generate_phrase(caption), filename)
