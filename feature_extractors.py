# In this file the class Encoder is defined. The class is an nn module that implements feature extraction from images by means of 
# the convolutional base of a pre-trained VGG-16 Neural Network. 
# The feature extraction module, given as input a tensor with a badge of cropped and normalized RGB images, returns a batch of tensors of 
# dimesion (bsz,L,num_feat) where L are the number of embeddings of the image (each one roughly corresponding to a patch of the image)
#  
# The argument num_emb of the constructor defines the number embeddings that will be extracted from the image. It can only take only the values 49 or 196.
# Default value is 196.    
# 
# The parameters of the encoder are frozen. In case of the need of fine tuning, the lastest layers of the network should be unfrozen.   
# 
#   Example for using for inference:
#   encoder = Encoder(num_emb_196=True)
#	encoder.eval()      
#   encoder.to(device)
#   for param in encoder.parameters():  # Freeze All layers 
#          param.requires_grad = False
#

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

## Import per fer proves

from dataset import Flickr8kDataset
from torch.utils.data import Dataset, DataLoader
from caps_collate import CapsCollate

class Encoder(nn.Module):

    def __init__(self,num_emb_196=True):
        super().__init__()

    ####   If we wanted to use the resnet50
    #   resnet = models.resnet50(pretrained=True)
    #   modules = list(resnet.children())[:-2]
    #   self.resnet = nn.Sequential(*modules)

   #####    We use a vgg16 network for the  feature extraction
        pretrained_model = models.vgg16(pretrained=True)
        if num_emb_196==True:                                   # In this case we just keep the first 24 layers of the Vgg
            self.conv_base = nn.Sequential(*list(pretrained_model.features)[0:24])
            print("Extracting 196 feat vect from the image")
        else:
            self.conv_base = pretrained_model.features         # In this case we keep all the layers of the convolutional base of the Vgg
            print("Extracting 49 feat vect from the image")

    # Freeze All layers as it will be mainly used for inference

        for i,param in enumerate(self.conv_base.parameters()):  
            param.requires_grad = False
        #    print(param.name, param.requires_grad)
        # print("numero de grupos de parametros:", i+1)

     # Flaten layer that flatten the dimensions 2 and 3 (H and W of the feature maps respectively)
        self.flat = nn.Flatten(2,3)      ####   MAYBE BETTER USE RESHAPE ???

    def forward(self, x):
        feat = self.conv_base(x)          # (batch_size, feat_maps=512, H=7 , W=7) or (batch_size, 512, 14 , 14)
        feat = self.flat(feat)               # (batch_size, 512, 7x7=49) or (batch_size, 512, 14x14=196)
        feat =  feat.transpose(1,2)          # (batch_size, 49, 512) or (batch_size, 196, 512)
        return feat


if __name__ == "__main__":

    num_emb_196=False     ## If this is false the number of embeddings per image is 49
    image_size = 224   ## this is to keep the aspect relation. Otherwised we could set image_size to (224,224) and then there is no need to crop the image.
    batch_size = 16
    MAX_SIZE = 5000
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),   ## only meaningful when we are keeping the aspect ratio when resizing. Otherwhise it has no effect
        # The normalize parameters depends on the model we're gonna use
        # If we apply transfer learning from a model that used ImageNet, then
        # we should use the ImageNet values to normalize the dataset.
        # Otherwise we could just normalize the values between -1 and 1 using the 
        # standard mean and standard deviation
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = Flickr8kDataset(transform=transform,reduce=True,max_size=MAX_SIZE)

    # split the database    TODO
    # print(len(dataset))

    # Set up the data loader for test
    dataloader = DataLoader(dataset=dataset,batch_size=batch_size, collate_fn=CapsCollate(pad_idx=dataset.vocab.word_to_index['<PAD>'],batch_first=True))
    batch_sample = next(iter(dataloader))
    images, captions = batch_sample

    # Load the encoder
    encoder = Encoder(num_emb_196=False)
    encoder.eval()

##    encoder.to(device)

    print(encoder) 

    print("Shape of the images tensor:",images.shape)

    img_emb = encoder(images)

    print("Shape of the extracted features:",img_emb.shape)

    # print an example of some items of the features tensor
    print(img_emb[0][0][:14])
